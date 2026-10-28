"""
ai_season.py

Optional AI-based replacement for season.py's decide_season() decision step,
loading the tf_efficientnetv2_l checkpoint trained in the sibling
personal_color_ai/ project (53.7% TEST accuracy on the real Deep Armocromia
benchmark, no single-class collapse -- vs. decide_season()'s measured 27.7%
with a severe Spring-collapse, see
backend_auramatch/evaluation_dataset/raw_validation/RAW_PIPELINE_VALIDATION.md).

Deliberately self-contained -- does NOT import anything from personal_color_ai/
(a sibling repo, not part of this repo's git tree; won't exist in a Railway
checkout). Only the minimal model-class definition and inference logic are
vendored here, not personal_color_ai's training/config/recommendation code.

Gated behind PERSONAL_COLOR_ENGINE=ai (see personal_color_service.py) --
default behavior is completely unaffected; this module is only imported when
that flag is on.
"""
from __future__ import annotations

import os
import time

import numpy as np

MACRO_SEASONS = ("spring", "summer", "autumn", "winter")
MICRO_SUBCLASSES = ("light", "soft", "deep", "warm", "cool", "bright")

# The 12 real (season, subclass) pairs that actually occur in Deep Armocromia
# (see personal_color_ai/recommendation/recommender.py's VALID_MICRO_SEASONS,
# copied here verbatim -- not re-derived, not invented).
VALID_MICRO_BY_SEASON = {
    "spring": ("light", "warm", "bright"),
    "summer": ("light", "cool", "soft"),
    "autumn": ("warm", "soft", "deep"),
    "winter": ("bright", "cool", "deep"),
}

_MODEL = None
_MODEL_BACKBONE = "tf_efficientnetv2_l.in21k_ft_in1k"


def _build_model():
    """Builds the (untrained-weights) nn.Module. A plain function, not a
    module-level class, so `import torch`/`timm` only happens when this
    module is actually used (PERSONAL_COLOR_ENGINE=ai), not merely imported."""
    import torch.nn as nn
    import timm

    class DualHeadSeasonClassifier(nn.Module):
        """Trimmed copy of personal_color_ai/models/classifier.py's class --
        architecture must match the trained checkpoint exactly (shared
        backbone + two linear heads), see that file for full design notes."""

        def __init__(self):
            super().__init__()
            # pretrained=False: the checkpoint's own weights immediately
            # overwrite anything ImageNet-pretrained init would set, so
            # pretrained=True here would just be a wasted HuggingFace Hub
            # fetch (and a possible failure point if the Hub isn't reachable).
            self.backbone = timm.create_model(_MODEL_BACKBONE, pretrained=False, num_classes=0)
            feat_dim = self.backbone.num_features
            self.dropout = nn.Dropout(0.2)
            self.macro_head = nn.Linear(feat_dim, len(MACRO_SEASONS))
            self.micro_head = nn.Linear(feat_dim, len(MICRO_SUBCLASSES))

        def forward(self, pixel_values):
            features = self.backbone(pixel_values)
            features = self.dropout(features)
            return {
                "macro_logits": self.macro_head(features),
                "micro_logits": self.micro_head(features),
            }

    return DualHeadSeasonClassifier()


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    import torch

    checkpoint_path = os.getenv("PERSONAL_COLOR_AI_CHECKPOINT_PATH")
    if not checkpoint_path:
        raise FileNotFoundError(
            "PERSONAL_COLOR_AI_CHECKPOINT_PATH is not set -- required when "
            "PERSONAL_COLOR_ENGINE=ai"
        )
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"AI checkpoint not found: {checkpoint_path}")

    print(f"🧠 loading AI personal-color model from {checkpoint_path} ...")
    t0 = time.time()
    device = torch.device("cpu")
    model = _build_model()
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    _MODEL = model
    print(f"✅ AI personal-color model loaded in {time.time() - t0:.1f}s")
    return _MODEL


def _eval_transform(image_rgb):
    """Resize(224,224) + ToTensor + ImageNet-normalize -- same 4 steps as
    personal_color_ai/data/transforms.py's get_eval_transforms(), reimplemented
    inline rather than imported (that module lives in the sibling repo)."""
    import torch
    from PIL import Image

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    img = Image.fromarray(image_rgb).resize((224, 224), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
    tensor = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    return torch.from_numpy(tensor).unsqueeze(0).float()


def predict_ai(image_bgr: np.ndarray) -> dict:
    """Runs the trained CNN on one BGR image (as already decoded by
    personal_color_service.py's cv2.imdecode -- no re-decode here). Returns
    macro season + micro subclass predictions with confidence, the micro
    prediction masked to the 12 valid (season, subclass) pairs so it's
    always coherent with the predicted season (same guarantee
    decide_season()'s sub_tone already gives).
    """
    import torch

    model = _load_model()
    image_rgb = image_bgr[:, :, ::-1]  # BGR -> RGB
    pixel_values = _eval_transform(image_rgb)

    with torch.no_grad():
        out = model(pixel_values)
        macro_probs = torch.softmax(out["macro_logits"], dim=-1)[0]
        micro_probs = torch.softmax(out["micro_logits"], dim=-1)[0]

    macro_idx = int(macro_probs.argmax())
    season = MACRO_SEASONS[macro_idx]
    season_confidence = float(macro_probs[macro_idx])

    valid_micro = VALID_MICRO_BY_SEASON[season]
    masked_micro = [
        float(micro_probs[i]) if MICRO_SUBCLASSES[i] in valid_micro else -1.0
        for i in range(len(MICRO_SUBCLASSES))
    ]
    micro_idx = int(np.argmax(masked_micro))
    subclass = MICRO_SUBCLASSES[micro_idx]

    ranked = sorted(
        ({"season": s, "probability": round(float(p), 4)} for s, p in zip(MACRO_SEASONS, macro_probs.tolist())),
        key=lambda d: -d["probability"],
    )

    return {
        "season": season,
        "season_confidence": round(season_confidence, 4),
        "subclass": subclass,
        "ranked_seasons": ranked,
    }
