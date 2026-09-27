"""
inference_engine.py
Fetal Ultrasound Facial Measurement Inference Engine

Ported directly from:
  - keypoint_output_function.py  (SimplePoseModel, TTA wrapper, get_dark_preds)
  - keypoint_output_evaluater.py (clinical indicators, geometry drawing, ensemble loop)
  - test.ipynb                   (YOLO ensemble, pipeline)

All model / inference logic is kept identical to the source notebooks.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
KPT_NAMES = ['forehead_1', 'forehead_2', 'nasion',
             'palate_1', 'palate_2', 'upper_lip',
             'mandible', 'mentum_tip']

CLI_NAMES = ['FMF', 'FMA', 'IFA', 'PL_Dist']

# Clinical normal ranges (literature-based)
CLINICAL_RANGES = {
    'FMF': {
        'normal': (75.0, 85.0), 'unit': 'deg',
        'low_risk':  'FMF angle is within normal range. No obvious frontal facial profile abnormality.',
        'high_risk': 'FMF angle is below normal range. May be associated with Trisomy 21 (flat face). Further evaluation recommended.',
        'very_high': 'FMF angle is significantly abnormal. Highly suspicious of anterior skull base developmental anomaly. Chromosomal workup recommended.',
    },
    'FMA': {
        'normal': (60.0, 75.0), 'unit': 'deg',
        'low_risk':  'FMA angle is within normal range. Mandible-palate relationship appears normal.',
        'high_risk': 'FMA angle is abnormal. May be related to micrognathia. Mandibular development evaluation recommended.',
        'very_high': 'FMA angle is significantly abnormal. Highly suspicious of mandibular hypoplasia. Specialist referral recommended.',
    },
    'IFA': {
        'normal': (50.0, 65.0), 'unit': 'deg',
        'low_risk':  'IFA angle is within normal range. Lower facial profile appears normal.',
        'high_risk': 'IFA angle is below normal (< 50 deg). May be associated with chromosomal abnormalities. Further evaluation recommended.',
        'very_high': 'IFA angle is markedly low. Trisomy 13/18/21 should be ruled out as a priority.',
    },
    'PL_Dist': {
        'normal': (0.0, 15.0), 'unit': 'px',
        'low_risk':  'PL distance is within normal range. Frontal bone projection appears normal.',
        'high_risk': 'PL distance is elevated. Possible frontal bossing or facial profile abnormality. Evaluation recommended.',
        'very_high': 'PL distance is significantly abnormal. Further assessment of frontal bone structure recommended.',
    },
}

# Drawing colors (BGR)
COLOR_PRED = (255, 200, 0)   # bright blue (predictions)
COLOR_GT   = (0, 165, 255)   # orange-red  (ground truth, kept for completeness)


# ─────────────────────────────────────────────
# SimplePoseModel
# Source: keypoint_output_function.py — SimplePoseModel
# ─────────────────────────────────────────────
class SimplePoseModel(nn.Module):
    """
    Pose estimation model supporting HRNet and ViTPose backbones.
    Identical to the definition in keypoint_output_function.py.
    """

    def __init__(self, model_type: str, ratio: int, num_keypoints: int = 8, img_size: int = 640):
        super().__init__()
        self.ratio = ratio
        if model_type == 'hrnet':
            self.backbone = timm.create_model('hrnet_w32', pretrained=True, features_only=True)
            ch = self.backbone.feature_info.channels()[0]
            self.head = nn.Conv2d(ch, num_keypoints, 1)
        elif model_type == 'vitpose':
            self.backbone = timm.create_model(
                'vit_small_patch16_224', pretrained=True,
                num_classes=0, global_pool='', img_size=img_size
            )
            self.head = nn.Sequential(
                nn.ConvTranspose2d(384, 128, 4, 2, 1),
                nn.ReLU(),
                nn.ConvTranspose2d(128, num_keypoints, 4, 2, 1),
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def forward(self, x):
        feat = self.backbone(x)
        if isinstance(feat, list):
            out = self.head(feat[0])
        else:
            B, N, C = feat.shape
            # Drop the CLS token if present
            if int(N ** 0.5) ** 2 != N:
                feat = feat[:, 1:, :]
                N = N - 1
            H = W = int(N ** 0.5)
            out = self.head(feat.transpose(1, 2).reshape(B, C, H, W))

        target_h = int(x.shape[2] // self.ratio)
        target_w = int(x.shape[3] // self.ratio)
        if out.shape[2] != target_h or out.shape[3] != target_w:
            out = torch.nn.functional.interpolate(
                out, size=(target_h, target_w), mode='bilinear', align_corners=False)
        return out


# ─────────────────────────────────────────────
# Coordinate Decoding
# Source: keypoint_output_function.py
# ─────────────────────────────────────────────
def get_max_preds(batch_heatmaps):
    """Argmax-based coordinate decoding."""
    batch_size, num_joints, h, w = batch_heatmaps.shape
    hm_flat = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx     = torch.argmax(hm_flat, 2).unsqueeze(-1)
    preds   = torch.cat([idx % w, idx // w], dim=2).float()
    return preds


def get_dark_preds(batch_heatmaps):
    """
    DARK sub-pixel refinement.
    Source: keypoint_output_function.py — get_dark_preds
    Offset acceptance threshold: <= 0.5 (identical to source).
    """
    batch_size, num_joints, h, w = batch_heatmaps.shape
    hm_flat = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx     = torch.argmax(hm_flat, 2).unsqueeze(-1)
    preds   = torch.cat([idx % w, idx // w], dim=2).float()

    preds_np = preds.cpu().numpy()          # NOTE: no .copy() — matches source
    hm_np    = batch_heatmaps.cpu().numpy()

    for b in range(batch_size):
        for j in range(num_joints):
            px, py = int(preds_np[b, j, 0]), int(preds_np[b, j, 1])
            if 1 < px < w - 1 and 1 < py < h - 1:
                hm  = hm_np[b, j]
                l   = np.log(np.maximum(hm[py,   px],   1e-6))
                lx  = np.log(np.maximum(hm[py,   px+1], 1e-6))
                lnx = np.log(np.maximum(hm[py,   px-1], 1e-6))
                ly  = np.log(np.maximum(hm[py+1, px],   1e-6))
                lny = np.log(np.maximum(hm[py-1, px],   1e-6))
                dx  = 0.5  * (lx - lnx)
                dy  = 0.5  * (ly - lny)
                dxx = lx   - 2 * l + lnx
                dyy = ly   - 2 * l + lny
                dxy = 0.25 * (np.log(np.maximum(hm[py+1, px+1], 1e-6))
                            - np.log(np.maximum(hm[py+1, px-1], 1e-6))
                            - np.log(np.maximum(hm[py-1, px+1], 1e-6))
                            + np.log(np.maximum(hm[py-1, px-1], 1e-6)))
                hessian    = np.array([[dxx, dxy], [dxy, dyy]])
                derivative = np.array([dx, dy])
                if np.linalg.det(hessian) != 0:
                    offset = -np.linalg.inv(hessian) @ derivative   # matches source (inv, not solve)
                    if np.all(np.abs(offset) <= 0.5):                # threshold = 0.5, matches source
                        preds_np[b, j] += offset

    return torch.from_numpy(preds_np).to(batch_heatmaps.device)


# ─────────────────────────────────────────────
# TTA Wrapper
# Source: keypoint_output_function.py — FetalTimmModelWrapper_albumentations_TTA
# ─────────────────────────────────────────────
class FetalTimmModelWrapper_albumentations_TTA:
    """
    Multi-scale TTA inference wrapper with ViT positional embedding interpolation.
    Identical to FetalTimmModelWrapper_albumentations_TTA in keypoint_output_function.py.
    """

    def __init__(self, model_type: str, ratio: int, checkpoint_path: str,
                 gray: bool, imgsz: int = 640, device: str = 'cuda'):
        self.device     = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.base_imgsz = imgsz
        self.model_type = model_type
        self.ratio      = ratio
        self.gray       = gray

        self.model = SimplePoseModel(model_type, ratio, num_keypoints=8,
                                     img_size=imgsz).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        state = ckpt.get('model_state_dict', ckpt)
        self.model.load_state_dict(state)
        self.model.eval()

        # Unlock ViT backbone for dynamic input sizes and back up positional embedding
        if model_type == 'vitpose':
            if hasattr(self.model.backbone, 'patch_embed'):
                self.model.backbone.patch_embed.strict_img_size = False
            if hasattr(self.model.backbone, 'pos_embed'):
                self.orig_pos_embed = self.model.backbone.pos_embed.clone().detach()
                print("Positional embedding backed up.")
            print("ViT strict size check disabled — multi-scale TTA enabled.")

        print(f"[TTA] {model_type.upper()} weights loaded: {checkpoint_path}")

    @torch.no_grad()
    def predict(self, img_bgr: np.ndarray) -> dict:
        orig_h, orig_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        test_scales    = [512, 640, 768]
        all_orig_preds = []
        all_dark_preds = []

        for scale in test_scales:
            # ── Interpolate positional embedding for this scale ──
            if self.model_type == 'vitpose' and hasattr(self.model.backbone, 'pos_embed'):
                channels        = self.orig_pos_embed.shape[2]
                orig_grid_h = orig_grid_w = 40   # 640 / 16 = 40
                num_extra_tokens = self.orig_pos_embed.shape[1] - (orig_grid_h * orig_grid_w)

                cls_token_embed = self.orig_pos_embed[:, :num_extra_tokens, :]
                src_grid_embed  = self.orig_pos_embed[:, num_extra_tokens:, :]
                src_grid_embed  = src_grid_embed.reshape(
                    1, orig_grid_h, orig_grid_w, channels).permute(0, 3, 1, 2)

                tgt_grid_h = tgt_grid_w = scale // 16
                tgt_grid_embed = torch.nn.functional.interpolate(
                    src_grid_embed,
                    size=(tgt_grid_h, tgt_grid_w),
                    mode='bicubic',
                    align_corners=False
                )
                tgt_grid_embed = tgt_grid_embed.permute(0, 2, 3, 1).reshape(
                    1, tgt_grid_h * tgt_grid_w, channels)
                new_pos_embed = torch.cat([cls_token_embed, tgt_grid_embed], dim=1)
                self.model.backbone.pos_embed = torch.nn.Parameter(new_pos_embed)

            # ── Albumentations preprocessing ──
            if self.gray:
                scale_transform = A.Compose([
                    A.LongestMaxSize(max_size=scale),
                    A.PadIfNeeded(min_height=scale, min_width=scale,
                                  border_mode=cv2.BORDER_CONSTANT, fill=[128, 128, 128]),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
            else:
                scale_transform = A.Compose([
                    A.LongestMaxSize(max_size=scale),
                    A.PadIfNeeded(min_height=scale, min_width=scale,
                                  border_mode=cv2.BORDER_REPLICATE),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

            aug    = scale_transform(image=img_rgb, keypoints=[[0.0, 0.0]])
            tensor = aug['image'].to(self.device).unsqueeze(0)
            hm     = self.model(tensor)

            preds_orig = get_max_preds(hm)
            preds_dark = get_dark_preds(hm)

            r           = min(scale / orig_h, scale / orig_w)
            new_unpad_w = int(round(orig_w * r))
            new_unpad_h = int(round(orig_h * r))
            dw          = (scale - new_unpad_w) / 2.0
            dh          = (scale - new_unpad_h) / 2.0

            def rescale_to_absolute_orig(pts):
                pts_np        = pts.cpu().numpy()[0]
                coords_scale  = pts_np * self.ratio   # heatmap -> padded-image coords
                coords_abs    = np.zeros_like(coords_scale)
                coords_abs[:, 0] = (coords_scale[:, 0] - dw) / r
                coords_abs[:, 1] = (coords_scale[:, 1] - dh) / r
                return coords_abs

            all_orig_preds.append(rescale_to_absolute_orig(preds_orig))
            all_dark_preds.append(rescale_to_absolute_orig(preds_dark))

        return {
            'orig': np.mean(all_orig_preds, axis=0),
            'dark': np.mean(all_dark_preds, axis=0),
        }


# ─────────────────────────────────────────────
# YOLO Ensemble
# Source: test.ipynb — run_yolo_ensemble_testing_with_val /
#         ensemble_voting_boxes_detailed
# ─────────────────────────────────────────────
def _compute_iou(box1, box2):
    """Compute IoU between two [x1, y1, x2, y2] bounding boxes."""
    x1 = max(box1[0], box2[0]);  y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]);  y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def ensemble_voting_boxes(all_boxes, iou_thresh=0.6, min_votes=3):
    """
    5-Fold YOLO ensemble with majority-vote box fusion.
    Greedy IoU clustering → confidence-weighted average → require >= min_votes distinct folds.
    """
    if not all_boxes:
        return None

    used     = [False] * len(all_boxes)
    clusters = []

    for i, b in enumerate(all_boxes):
        if used[i]:
            continue
        cluster = [i]
        used[i] = True
        for j in range(i + 1, len(all_boxes)):
            if not used[j] and _compute_iou(b['box'], all_boxes[j]['box']) >= iou_thresh:
                cluster.append(j)
                used[j] = True
        clusters.append(cluster)

    best_box   = None
    best_votes = 0
    best_conf  = 0.0

    for cluster in clusters:
        folds = set(all_boxes[i]['fold'] for i in cluster)
        votes = len(folds)
        if votes < min_votes:
            continue
        confs   = np.array([all_boxes[i]['conf'] for i in cluster])
        boxes   = np.array([all_boxes[i]['box']  for i in cluster])
        avg_box   = (boxes * confs[:, None]).sum(axis=0) / confs.sum()
        mean_conf = confs.mean()
        if votes > best_votes or (votes == best_votes and mean_conf > best_conf):
            best_votes = votes
            best_conf  = mean_conf
            best_box   = avg_box.tolist()

    return best_box   # [x1, y1, x2, y2] or None


# ─────────────────────────────────────────────
# Geometric Helper Functions
# Source: keypoint_output_evaluater.py — FetalAcademicEvaluator._get_line_intersection
# ─────────────────────────────────────────────
def _get_line_intersection(p1, p2, p3, p4):
    """
    Return the intersection of the infinite lines through (p1,p2) and (p3,p4).
    Uses determinant-based Cramer's Rule — identical to source.
    Returns None when lines are parallel (|div| < 1e-9).
    """
    xdiff = np.array([p1[0] - p2[0], p3[0] - p4[0]])
    ydiff = np.array([p1[1] - p2[1], p3[1] - p4[1]])

    def det(v1, v2): return v1[0] * v2[1] - v1[1] * v2[0]

    div = det(xdiff, ydiff)
    if abs(div) < 1e-9:
        return None

    d = np.array([det(p1, p2), det(p3, p4)])
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return np.array([x, y], dtype=np.float32)


def angle_pure(v1, v2):
    """Angle (0–180 deg) between two vectors. Matches source implementation."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 * n2 < 1e-9:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))))


# ─────────────────────────────────────────────
# Clinical Indicator Calculation
# Source: keypoint_output_evaluater.py — _get_clinical_indicators
# ─────────────────────────────────────────────
def get_clinical_indicators(coords: np.ndarray, vis_mask: np.ndarray) -> dict:
    """
    Compute FMF, FMA, IFA, PL_Dist.
    Logic is identical to FetalAcademicEvaluator._get_clinical_indicators in
    keypoint_output_evaluater.py (farthest-endpoint method for all three angles).

    Args:
        coords  : (8, 2) float32 pixel coordinates.
        vis_mask: (8,)  float; > 0 means the keypoint is valid.
    Returns:
        dict keyed by CLI_NAMES; nan when a keypoint is missing.
    """
    res = {k: np.nan for k in CLI_NAMES}

    v_profile = None
    if all(vis_mask[[5, 7]] > 0):
        v_profile = coords[7] - coords[5]   # upper_lip -> mentum_tip

    # ── 1. FMF ──────────────────────────────────────────────────────
    if all(vis_mask[[0, 1, 3, 4]] > 0):
        p0, p1 = coords[0], coords[1]   # forehead_1, forehead_2
        p3, p4 = coords[3], coords[4]   # palate_1,   palate_2
        p_cross = _get_line_intersection(p0, p1, p3, p4)
        if p_cross is not None:
            v_fore = (p1 - p_cross if np.linalg.norm(p1 - p_cross) > np.linalg.norm(p0 - p_cross)
                      else p0 - p_cross)
            v_pal  = (p4 - p_cross if np.linalg.norm(p4 - p_cross) > np.linalg.norm(p3 - p_cross)
                      else p3 - p_cross)
            res['FMF'] = angle_pure(v_fore, v_pal)

    # ── 2. FMA ──────────────────────────────────────────────────────
    if all(vis_mask[[3, 4, 5, 7]] > 0) and v_profile is not None:
        p3, p4 = coords[3], coords[4]
        p5, p7 = coords[5], coords[7]
        p_cross = _get_line_intersection(p3, p4, p5, p7)
        if p_cross is not None:
            v_pal  = (p4 - p_cross if np.linalg.norm(p4 - p_cross) > np.linalg.norm(p3 - p_cross)
                      else p3 - p_cross)
            v_face = (p7 - p_cross if np.linalg.norm(p7 - p_cross) > np.linalg.norm(p5 - p_cross)
                      else p5 - p_cross)
            res['FMA'] = angle_pure(v_pal, v_face)

    # ── 3. IFA ──────────────────────────────────────────────────────
    if all(vis_mask[[0, 1, 2, 5, 7]] > 0) and v_profile is not None:
        p0, p1, p2 = coords[0], coords[1], coords[2]
        p5, p7 = coords[5], coords[7]
        v_fore = p1 - p0
        v_norm = np.array([-v_fore[1], v_fore[0]], dtype=float)
        v_norm_len = np.linalg.norm(v_norm)
        if v_norm_len > 1e-6:
            v_norm = v_norm / v_norm_len
            ref_dir = coords[7] - p2   # ensure normal points toward mentum
            if np.dot(v_norm, ref_dir) < 0:
                v_norm = -v_norm
            p_norm_virtual = p2 + v_norm
            p_cross = _get_line_intersection(p2, p_norm_virtual, p5, p7)
            if p_cross is not None:
                v_to_nasion = p2 - p_cross
                v_to_lip = (p5 - p_cross if np.linalg.norm(p5 - p_cross) > np.linalg.norm(p7 - p_cross)
                            else p7 - p_cross)
                res['IFA'] = angle_pure(v_to_nasion, v_to_lip)

    # ── 4. PL_Dist ──────────────────────────────────────────────────
    if all(vis_mask[[0, 2, 6]] > 0):
        p0, p2, p6 = coords[0], coords[2], coords[6]
        v_line   = p6 - p2
        line_len = np.linalg.norm(v_line)
        if line_len > 1e-6:
            v_point = p0 - p2
            area = abs(v_point[0] * v_line[1] - v_point[1] * v_line[0])
            res['PL_Dist'] = float(area / line_len)

    return res


# ─────────────────────────────────────────────
# Diagnostic Suggestions
# ─────────────────────────────────────────────
def get_diagnosis(clinical_values: dict) -> str:
    """Generate a diagnostic suggestion string from clinical indicators."""
    lines = []
    for k, v in clinical_values.items():
        if np.isnan(v):
            lines.append(f"{k}: N/A (keypoint detection failed)")
            continue
        info = CLINICAL_RANGES[k]
        lo, hi = info['normal']
        unit = info['unit']
        lines.append(f"{k}: {v:.2f} {unit}")
        if lo <= v <= hi:
            lines.append(f"  -> {info['low_risk']}")
        elif abs(v - lo) > 15 or abs(v - hi) > 15:
            lines.append(f"  [!] {info['very_high']}")
        else:
            lines.append(f"  [!] {info['high_risk']}")
    return "\n".join(lines) if lines else "No diagnostic information available."


# ─────────────────────────────────────────────
# Drawing Functions
# Source: keypoint_output_evaluater.py — _draw_clinical_geometry
#         test.ipynb                   — image annotation block
# ─────────────────────────────────────────────
def draw_roi(img: np.ndarray, box, color=(0, 100, 255), thickness=2) -> np.ndarray:
    """Draw the ROI bounding box."""
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    return img


def draw_keypoints_and_geometry(img: np.ndarray, coords: np.ndarray,
                                vis_mask: np.ndarray,
                                color=COLOR_PRED, thickness=1) -> np.ndarray:
    """
    Draw keypoints and clinical geometry lines.
    Matches _draw_clinical_geometry in keypoint_output_evaluater.py exactly.
    """
    h, w = img.shape[:2]

    def to_pt(pt):
        return (int(round(float(pt[0]))), int(round(float(pt[1]))))

    def draw_line(idx1, idx2, should_extend=True):
        if all(vis_mask[[idx1, idx2]] > 0):
            p1, p2 = coords[idx1], coords[idx2]
            if should_extend:
                direction = p2 - p1
                norm = np.linalg.norm(direction)
                if norm > 1e-6:
                    ext_p1 = p1 - (direction / norm) * 2000
                    ext_p2 = p2 + (direction / norm) * 2000
                    cv2.line(img, to_pt(ext_p1), to_pt(ext_p2), color, thickness, cv2.LINE_AA)
            else:
                cv2.line(img, to_pt(p1), to_pt(p2), color, thickness, cv2.LINE_AA)

    draw_line(0, 1)   # forehead line
    draw_line(3, 4)   # palate line
    draw_line(5, 7)   # face profile line (intersects nasion normal perfectly)
    draw_line(2, 6)   # PL baseline (nasion – mandible)

    # Nasion perpendicular normal line (core for IFA)
    if all(vis_mask[[0, 1, 2]] > 0):
        p_nasion = coords[2]
        v_fore   = coords[1] - coords[0]
        v_norm   = np.array([-v_fore[1], v_fore[0]])
        v_norm_len = np.linalg.norm(v_norm)
        if v_norm_len > 1e-6:
            ext_p_norm1 = p_nasion - (v_norm / v_norm_len) * 2000
            ext_p_norm2 = p_nasion + (v_norm / v_norm_len) * 2000
            cv2.line(img, to_pt(ext_p_norm1), to_pt(ext_p_norm2), color, thickness, cv2.LINE_AA)

    # PL perpendicular projection line
    if all(vis_mask[[0, 2, 6]] > 0):
        p0, p2, p6 = coords[0], coords[2], coords[6]
        v26    = p6 - p2
        v26_sq = np.dot(v26, v26)
        if v26_sq > 1e-9:
            proj_pt = p2 + (np.dot(p0 - p2, v26) / v26_sq) * v26
            cv2.line(img, to_pt(p0), to_pt(proj_pt), color, 1, cv2.LINE_AA)

    return img


def draw_keypoint_circles(img: np.ndarray, coords: np.ndarray,
                          vis_mask: np.ndarray, color=COLOR_PRED, radius=3) -> np.ndarray:
    """Draw filled circles at each valid keypoint. Radius=3 matches source."""
    for i in range(8):
        if vis_mask[i] > 0:
            cv2.circle(img,
                       (int(round(float(coords[i, 0]))), int(round(float(coords[i, 1])))),
                       radius, color, -1, cv2.LINE_AA)
    return img


def draw_measurements_on_image(img: np.ndarray, clinical_values: dict,
                                coords: np.ndarray, vis_mask: np.ndarray,
                                img_name: str = "") -> np.ndarray:
    """
    Overlay measurement text on the image.
    Font scale = 0.35 and layout match the source annotation block in test.ipynb.
    """
    h, w = img.shape[:2]

    # Determine text position (away from forehead keypoints)
    if vis_mask[0] > 0 and vis_mask[1] > 0:
        f1_x = coords[0][0]
        f2_x = coords[1][0]
    else:
        f1_x, f2_x = 10, 20
    start_x = w - 150 if f1_x > f2_x else 15

    font = cv2.FONT_HERSHEY_SIMPLEX
    ts   = 0.35   # matches test.ipynb
    iy   = 25

    cv2.putText(img, f"IMG: {img_name}",
                (start_x, iy), font, ts, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "5-Fold Ensemble Mean",
                (start_x, iy + 15), font, ts, (255, 255, 255), 1, cv2.LINE_AA)
    iy += 35

    for k in CLI_NAMES:
        val = clinical_values.get(k, np.nan)
        if not np.isnan(val):
            unit = "deg" if k != 'PL_Dist' else "px"
            cv2.putText(img, f"{k}: {val:.1f}{unit}",
                        (start_x, iy), font, ts, (0, 255, 0), 1, cv2.LINE_AA)
            iy += 15
    return img


# ─────────────────────────────────────────────
# Main Inference Engine
# ─────────────────────────────────────────────
class FetalMeasurementEngine:
    """
    End-to-end pipeline:
      1. YOLO 5-Fold Ensemble ROI detection
      2. ViTPose 5-Fold + TTA (gray=True) keypoint detection with mean-ensemble
      3. Clinical indicator calculation
      4. Diagnostic suggestion generation

    Inference logic follows test.ipynb / FetalClinicalEnsembleSystem.evaluate_folder_ensemble
    (has_ground_truth=False branch).
    """

    def __init__(self,
                 yolo_model_paths: list,
                 vitpose_model_paths: list,
                 device: Optional[str] = None,
                 yolo_conf: float = 0.1,
                 yolo_iou: float = 0.6,
                 yolo_min_votes: int = 3,
                 kpt_min_votes: int = 3,
                 ratio: int = 2,
                 gray: bool = True):
        """
        Args:
            yolo_model_paths   : Paths to 5 YOLO best.pt weights (fold 1–5).
            vitpose_model_paths: Paths to 5 ViTPose best_*.pth weights (fold 1–5).
            device             : 'cuda' / 'cpu'. Auto-selected when None.
            yolo_conf          : YOLO confidence threshold (default 0.1).
            yolo_iou           : IoU threshold for ensemble clustering (default 0.6).
            yolo_min_votes     : Min folds required to accept a detection (default 3).
            kpt_min_votes      : Min folds required to accept a keypoint (default 3).
            ratio              : Heatmap-to-input downscale ratio (default 2, matches training).
            gray               : Use BORDER_CONSTANT gray padding for TTA (default True,
                                 matches vitpose_Augmentation_new_4_IP_4_dysigma_gray).
        """
        device_str = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Engine] Device: {device_str}")

        self.yolo_conf     = yolo_conf
        self.yolo_iou      = yolo_iou
        self.yolo_min_votes = yolo_min_votes
        self.kpt_min_votes  = kpt_min_votes

        # Load YOLO models
        self.yolo_models = []
        for path in yolo_model_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"[Engine] YOLO model not found: {path}")
            from ultralytics import YOLO
            self.yolo_models.append(YOLO(str(path)))
        print(f"[Engine] YOLO: loaded {len(self.yolo_models)} fold models")

        # Load ViTPose TTA wrappers (identical to test.ipynb model loading)
        self.vitpose_wrappers = []
        for path in vitpose_model_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"[Engine] ViTPose model not found: {path}")
            wrapper = FetalTimmModelWrapper_albumentations_TTA(
                model_type      = 'vitpose',
                ratio           = ratio,
                checkpoint_path = str(path),
                gray            = gray,
                imgsz           = 640,
                device          = device_str,
            )
            self.vitpose_wrappers.append(wrapper)
        print(f"[Engine] ViTPose: loaded {len(self.vitpose_wrappers)} fold TTA wrappers")

    # ── YOLO inference ────────────────────────
    def _run_yolo_single(self, img_bgr: np.ndarray):
        """Run 5-Fold YOLO ensemble. Returns [x1,y1,x2,y2] or None."""
        all_boxes = []
        for fold_idx, model in enumerate(self.yolo_models):
            results = model.predict(img_bgr, conf=self.yolo_conf, verbose=False)
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    all_boxes.append({'box': xyxy, 'conf': conf, 'fold': fold_idx})
        return ensemble_voting_boxes(all_boxes, self.yolo_iou, self.yolo_min_votes)

    # ── ViTPose inference ─────────────────────
    def _run_vitpose_single(self, img_bgr: np.ndarray, roi_box):
        """
        Run 5-Fold TTA ViTPose on the full image cropped to roi_box.
        Follows the evaluate_folder_ensemble loop in test.ipynb:
          - Each wrapper returns {'orig': ..., 'dark': ...} in original image space
          - Accumulate dark coords; accept keypoint if >= kpt_min_votes folds agree
        Returns (pred_coords_dark, vis_mask) in full-image pixel coords.
        """
        h_orig, w_orig = img_bgr.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in roi_box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_orig, x2), min(h_orig, y2)
        roi_bgr = img_bgr[y1:y2, x1:x2]

        sum_dark   = np.zeros((8, 2), dtype=np.float32)
        point_votes = np.zeros(8, dtype=np.float32)

        for wrapper in self.vitpose_wrappers:
            res_dict = wrapper.predict(roi_bgr)
            p_px_dark = res_dict['dark']          # (8, 2) in ROI coords
            p_vis     = np.ones(8, dtype=np.float32)  # TTA wrapper always predicts all points

            for i in range(8):
                if p_vis[i] > 0:
                    sum_dark[i]     += p_px_dark[i]
                    point_votes[i]  += 1

        pred_dark = np.zeros((8, 2), dtype=np.float32)
        vis_mask  = np.zeros(8, dtype=np.float32)
        for i in range(8):
            if point_votes[i] >= self.kpt_min_votes:   # majority vote >= 3
                pred_dark[i] = sum_dark[i] / point_votes[i]
                vis_mask[i]  = 1.0

        # Convert ROI-space coords to full-image coords
        pred_dark[:, 0] += x1
        pred_dark[:, 1] += y1

        return pred_dark, vis_mask

    # ── Main prediction method ────────────────
    def predict(self, img_bgr: np.ndarray) -> dict:
        """
        Full pipeline for a single BGR image.

        Returns dict with keys:
            'roi_box'     : [x1,y1,x2,y2] or None
            'kpt_coords'  : (8, 2) dark-refined coords in full-image space
            'kpt_vis'     : (8,) visibility mask
            'clinical'    : {FMF, FMA, IFA, PL_Dist}
            'diagnosis'   : str
            'vis_img'     : annotated BGR image
            'det_success' : bool
        """
        img_name   = ""   # caller may populate via result dict if needed
        h, w       = img_bgr.shape[:2]
        roi_box    = self._run_yolo_single(img_bgr)
        det_success = roi_box is not None

        kpt_coords = np.zeros((8, 2), dtype=np.float32)
        vis_mask   = np.zeros(8, dtype=np.float32)
        clinical   = {k: np.nan for k in CLI_NAMES}

        if det_success:
            kpt_coords, vis_mask = self._run_vitpose_single(img_bgr, roi_box)
            if vis_mask.sum() == 0:
                det_success = False
            else:
                clinical = get_clinical_indicators(kpt_coords, vis_mask)

        diagnosis = get_diagnosis(clinical)

        # Render annotated image — mirrors evaluate_folder_ensemble blind-test branch
        vis_img = img_bgr.copy()
        if roi_box is not None:
            draw_roi(vis_img, roi_box)
        if vis_mask.sum() > 0:
            draw_keypoints_and_geometry(vis_img, kpt_coords, vis_mask, color=COLOR_PRED)
            draw_keypoint_circles(vis_img, kpt_coords, vis_mask, color=COLOR_PRED, radius=3)
        # img_name is populated by the caller via result['img_name'] if needed;
        # we pass empty string here since the engine has no filename context.
        draw_measurements_on_image(vis_img, clinical, kpt_coords, vis_mask, img_name="")

        return {
            'roi_box':     roi_box,
            'kpt_coords':  kpt_coords,
            'kpt_vis':     vis_mask,
            'clinical':    clinical,
            'diagnosis':   diagnosis,
            'vis_img':     vis_img,
            'det_success': det_success,
        }


# ─────────────────────────────────────────────
# LabelMe JSON Export
# ─────────────────────────────────────────────
def build_labelme_json(img_path: str, img_shape: tuple,
                       roi_box, kpt_coords: np.ndarray,
                       vis_mask: np.ndarray, clinical: dict,
                       img_base64: str = None) -> dict:
    """
    Build a LabelMe-compatible JSON annotation.
    Label names follow the reference sample (2.json).
    """
    import base64
    import os

    shapes = []
    h, w = img_shape[:2]

    if roi_box is not None:
        x1, y1, x2, y2 = roi_box
        shapes.append({"label": "able_ROI",
                        "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
                        "group_id": None, "description": "",
                        "shape_type": "rectangle", "flags": {}})

    for i, name in enumerate(KPT_NAMES):
        if vis_mask[i] > 0:
            shapes.append({"label": name,
                            "points": [[float(kpt_coords[i, 0]), float(kpt_coords[i, 1])]],
                            "group_id": None, "description": "",
                            "shape_type": "point", "flags": {}})

    def _add_extend_line(i1, i2, label):
        p1, p2 = kpt_coords[i1], kpt_coords[i2]
        d = p2 - p1;  nd = np.linalg.norm(d)
        if nd < 1e-6: return
        d = d / nd;   ext = max(h, w) * 2
        ep1 = p1 - d * ext;  ep2 = p2 + d * ext
        shapes.append({"label": label,
                        "points": [[float(ep1[0]), float(ep1[1])],
                                   [float(ep2[0]), float(ep2[1])]],
                        "group_id": None, "description": "",
                        "shape_type": "line", "flags": {}})

    if all(vis_mask[[0, 1]] > 0): _add_extend_line(0, 1, "FMF_1")
    if all(vis_mask[[3, 4]] > 0): _add_extend_line(3, 4, "FMF_2")
    if all(vis_mask[[5, 7]] > 0): _add_extend_line(5, 7, "FMA_2")
    if all(vis_mask[[0, 1, 2]] > 0):
        p2 = kpt_coords[2]
        v_fore = kpt_coords[1] - kpt_coords[0]
        v_norm = np.array([-v_fore[1], v_fore[0]], dtype=float)
        vl = np.linalg.norm(v_norm)
        if vl > 1e-6:
            v_norm = v_norm / vl
            ep1 = p2 - v_norm * min(h, w);  ep2 = p2 + v_norm * min(h, w)
            shapes.append({"label": "IFA_2",
                            "points": [[float(ep1[0]), float(ep1[1])],
                                       [float(ep2[0]), float(ep2[1])]],
                            "group_id": None, "description": "",
                            "shape_type": "line", "flags": {}})
    if all(vis_mask[[2, 6]] > 0): _add_extend_line(2, 6, "PL_distance_1")
    if all(vis_mask[[0, 2, 6]] > 0):
        p0, p2_, p6 = kpt_coords[0], kpt_coords[2], kpt_coords[6]
        v26 = p6 - p2_;  v26_sq = np.dot(v26, v26)
        if v26_sq > 1e-9:
            proj_pt = p2_ + (np.dot(p0 - p2_, v26) / v26_sq) * v26
            shapes.append({"label": "PL_distance",
                            "points": [[float(p0[0]), float(p0[1])],
                                       [float(proj_pt[0]), float(proj_pt[1])]],
                            "group_id": None, "description": "",
                            "shape_type": "line", "flags": {}})

    if img_base64 is None:
        try:
            with open(img_path, 'rb') as f:
                img_base64 = base64.b64encode(f.read()).decode('utf-8')
        except Exception:
            img_base64 = None

    return {"version": "5.3.1", "flags": {}, "shapes": shapes,
            "imagePath": os.path.basename(img_path), "imageData": img_base64,
            "imageHeight": h, "imageWidth": w}


# ─────────────────────────────────────────────
# Normal range strings (from keypoint_output_evaluater.py)
# ─────────────────────────────────────────────
NORMAL_RANGE = {
    'FMF':     '78.8~87.0 deg',
    'FMA':     '68.01~78.03 deg',
    'IFA':     '74.75~82.89 deg',
    'PL_Dist': 'N/A',
}


# ─────────────────────────────────────────────
# Info Image Export
# Layout reference: keypoint_output_evaluater.py (blind-test mode)
# The annotated image is placed on top; a dark-grey info panel is
# appended BELOW the image so nothing is occluded.  The panel height
# expands automatically to fit all text.
# ─────────────────────────────────────────────
def build_info_image(img_bgr: np.ndarray, vis_img: np.ndarray,
                     clinical: dict, diagnosis: str,
                     img_name: str = "") -> np.ndarray:
    """
    Compose an output image:
      TOP    — annotated prediction image (vis_img)
      BOTTOM — dark-grey information panel with measurements + diagnosis

    Each clinical value is shown as:
        FMF: 80.3 deg  (normal: 78.8~87.0 deg)
    Diagnosis text is word-wrapped to fit the panel width.
    Panel height auto-expands when there is not enough room.
    """
    h, w = vis_img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    ts   = 0.38        # font scale — same order of magnitude as source
    lh   = 18          # line height in pixels
    pad  = 10          # left/top padding
    bg   = (45, 45, 45)   # dark grey, matches keypoint_output_evaluater.py

    # ── Build text lines ──────────────────────────────────────────
    text_lines = []   # list of (text, color)

    white  = (240, 240, 240)
    yellow = (150, 255, 255)   # clinical values — light yellow, matches source
    green  = (100, 255, 100)
    red    = (100, 100, 255)   # BGR: reddish

    text_lines.append((f"IMG: {img_name}", white))
    text_lines.append(("", white))

    # Measurements with normal range in parentheses
    text_lines.append(("[ Measurements ]", yellow))
    for k in CLI_NAMES:
        val  = clinical.get(k, np.nan)
        unit = "deg" if k != 'PL_Dist' else "px"
        nr   = NORMAL_RANGE.get(k, '')
        if np.isnan(val):
            text_lines.append((f"  {k}: N/A", white))
        else:
            if k == 'PL_Dist':
                text_lines.append((f"  {k}: {val:.2f} {unit}", yellow))
            else:
                text_lines.append((f"  {k}: {val:.2f} {unit}  (normal: {nr})", yellow))

    text_lines.append(("", white))

    # Diagnosis — word-wrap each line to fit panel width
    text_lines.append(("[ Diagnosis ]", yellow))
    max_chars_per_line = max(40, w // 7)   # rough character budget per line
    for raw_line in diagnosis.split('\n'):
        raw_line = raw_line.strip()
        if not raw_line:
            text_lines.append(("", white))
            continue
        # Determine colour by content
        if raw_line.startswith('[!]'):
            c = red
        elif raw_line.startswith('->'):
            c = green
        else:
            c = white

        # Word-wrap using cv2.getTextSize for pixel-accurate breaking
        words = raw_line.split(' ')
        current = ""
        for word in words:
            test = (current + " " + word).strip()
            (tw, _), _ = cv2.getTextSize(test, font, ts, 1)
            if tw <= w - pad * 2:
                current = test
            else:
                if current:
                    text_lines.append(("  " + current, c))
                current = word
        if current:
            text_lines.append(("  " + current, c))

    # ── Calculate panel height ─────────────────────────────────────
    n_lines   = len(text_lines)
    panel_h   = max(120, n_lines * lh + pad * 2)

    # ── Draw panel ────────────────────────────────────────────────
    panel = np.full((panel_h, w, 3), bg, dtype=np.uint8)

    iy = pad + lh
    for line_text, color in text_lines:
        if line_text == "":
            iy += lh // 2
            continue
        ts_use = ts + 0.02 if line_text.startswith("[ ") else ts
        cv2.putText(panel, line_text, (pad, iy), font, ts_use, color, 1, cv2.LINE_AA)
        iy += lh
        if iy > panel_h - pad:
            # Panel ran out of space — extend it
            extra = np.full((lh * 4, w, 3), bg, dtype=np.uint8)
            panel = np.vstack([panel, extra])
            panel_h += lh * 4

    # ── Stack image (top) and panel (bottom) ──────────────────────
    return np.vstack([vis_img, panel])


# ─────────────────────────────────────────────
# Info Image Export
# Layout reference: keypoint_output_evaluater.py (blind-test mode)
# The annotated image is placed on top; a dark-grey info panel is
# appended BELOW the image so nothing is occluded.  The panel height
# expands automatically to fit all text.
# ─────────────────────────────────────────────
def build_info_image(img_bgr: np.ndarray, vis_img: np.ndarray,
                     clinical: dict, diagnosis: str,
                     img_name: str = "") -> np.ndarray:
    """
    Compose an output image:
      TOP    — annotated prediction image (vis_img)
      BOTTOM — dark-grey information panel with measurements + diagnosis

    Each clinical value is shown as:
        FMF: 80.3 deg  (normal: 78.8~87.0 deg)
    Diagnosis text is word-wrapped to fit the panel width.
    Panel height auto-expands when there is not enough room.
    """
    h, w = vis_img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    ts   = 0.38        # font scale — same order of magnitude as source
    lh   = 18          # line height in pixels
    pad  = 10          # left/top padding
    bg   = (45, 45, 45)      # dark grey, matches keypoint_output_evaluater.py

    # ── Colour palette ────────────────────────────────────────────
    white  = (240, 240, 240)
    yellow = (150, 255, 255)   # light yellow — clinical values
    green  = (100, 255, 100)   # normal / positive
    red    = (100, 100, 255)   # abnormal / warning (BGR)

    # ── Build text line list: (text_string, colour) ──────────────
    text_lines = []

    text_lines.append((f"IMG: {img_name}", white))
    text_lines.append(("", white))

    # --- Measurements section ---
    text_lines.append(("[ Measurements ]", yellow))
    for k in CLI_NAMES:
        val  = clinical.get(k, np.nan)
        unit = "deg" if k != 'PL_Dist' else "px"
        nr   = NORMAL_RANGE.get(k, '')
        if np.isnan(val):
            text_lines.append((f"  {k}: N/A", white))
        else:
            if k == 'PL_Dist':
                text_lines.append((f"  {k}: {val:.2f} {unit}", yellow))
            else:
                text_lines.append(
                    (f"  {k}: {val:.2f} {unit}  (normal: {nr})", yellow))

    text_lines.append(("", white))

    # --- Diagnosis section — word-wrap each raw line ---------------
    text_lines.append(("[ Diagnosis ]", yellow))
    for raw_line in diagnosis.split('\n'):
        raw_line = raw_line.strip()
        if not raw_line:
            text_lines.append(("", white))
            continue

        # Pick colour by content prefix
        if raw_line.startswith('[!]'):
            c = red
        elif raw_line.startswith('->'):
            c = green
        else:
            c = white

        # Pixel-accurate word-wrap using cv2.getTextSize
        words   = raw_line.split(' ')
        current = ""
        for word in words:
            test = (current + " " + word).strip()
            (tw, _), _ = cv2.getTextSize(test, font, ts, 1)
            if tw <= w - pad * 2:
                current = test
            else:
                if current:
                    text_lines.append(("  " + current, c))
                current = word
        if current:
            text_lines.append(("  " + current, c))

    # ── Calculate initial panel height ───────────────────────────
    n_lines = len(text_lines)
    panel_h = max(120, n_lines * lh + pad * 2)

    # ── Draw panel ───────────────────────────────────────────────
    panel = np.full((panel_h, w, 3), bg, dtype=np.uint8)

    iy = pad + lh
    for line_text, color in text_lines:
        if line_text == "":
            iy += lh // 2
            continue
        ts_use = ts + 0.02 if line_text.startswith("[ ") else ts
        # Auto-expand panel if we are about to overflow
        if iy > panel.shape[0] - pad:
            extra = np.full((lh * 6, w, 3), bg, dtype=np.uint8)
            panel = np.vstack([panel, extra])
        cv2.putText(panel, line_text, (pad, iy), font, ts_use, color, 1, cv2.LINE_AA)
        iy += lh

    # ── Stack image (top) + info panel (bottom) ──────────────────
    return np.vstack([vis_img, panel])
