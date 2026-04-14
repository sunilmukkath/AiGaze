# ============================================================
# AI GAZE — Predictive Eye Tracker
# Powered by Elastic Tree | Built with Python + Streamlit
# Features: Heatmap · Hotspot · Gaze Sequence · AOI · PDF Export
# ============================================================

import io
import os
import string
import tempfile
import base64

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image
from scipy.ndimage import gaussian_filter, maximum_filter

try:
    import requests
except ImportError:
    requests = None

# ── Compatibility patch: streamlit-drawable-canvas uses a moved + changed API ──
# New Streamlit expects a layout_config object; canvas passes a plain int for width.
try:
    import streamlit.elements.image as _st_img_module
    from streamlit.elements.lib.image_utils import image_to_url as _real_iurl
    try:
        from streamlit.elements.lib.layout_utils import LayoutConfig as _LayoutConfig
        def _compat_image_to_url(image, width, *args, **kwargs):
            lc = _LayoutConfig(width=width)
            return _real_iurl(image, lc, *args, **kwargs)
    except ImportError:
        # Fallback for older Streamlit versions
        def _compat_image_to_url(image, width, *args, **kwargs):
            class _LC:
                pass
            lc = _LC()
            lc.width = width
            lc.height = None
            lc.text_alignment = None
            return _real_iurl(image, lc, *args, **kwargs)
    _st_img_module.image_to_url = _compat_image_to_url
except Exception:
    pass

# ── Optional dependencies ──────────────────────────────────
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    CANVAS_AVAILABLE = False

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

try:
    import torch
    import deepgaze_pytorch
    DEEPGAZE_AVAILABLE = True
except ImportError:
    DEEPGAZE_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Gaze",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── App password ─────────────────────────────────────────────
_APP_PASSWORD = "elastic2026"

# ── Elastic Tree brand palette ───────────────────────────────
ET_PURPLE  = "#3D3587"
ET_TEAL    = "#3CBFBF"
ET_BLUE    = "#5B8DD9"
ET_GOLD    = "#F5A623"
ET_GREEN   = "#44BB77"

# Optional remote saliency inference endpoint (for DeepGaze/SALICON/ViT ensemble).
SALIENCY_API_URL = os.getenv("SALIENCY_API_URL", "").strip()

# ── Custom CSS ───────────────────────────────────────────────
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Playfair+Display:wght@600;700;800&family=Space+Grotesk:wght@500;600;700;800&display=swap');

* { box-sizing: border-box; }
html, body, .stApp {
    background: radial-gradient(1200px 500px at 12% -10%, #1a1f39 0%, #0b0f24 36%, #070a1b 70%, #060815 100%) !important;
    color: #e4e4f4;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
.block-container {
    padding-top: 2.6rem !important;
    padding-bottom: 2rem !important;
    max-width: 1280px;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #07071c; }
::-webkit-scrollbar-thumb { background: #1c1c3a; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #2c2c5a; }

/* ── Tabs — underline style ── */
.stTabs [data-baseweb="tab-list"] {
    background: linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.015)) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 16px !important;
    padding: 6px !important;
    gap: 6px !important;
    display: flex !important;
    flex-wrap: wrap !important;
    overflow: visible !important;
    align-items: stretch !important;
    box-shadow: 0 6px 22px rgba(0,0,0,0.16) !important;
    backdrop-filter: none !important;
}
.stTabs [data-baseweb="tab"] {
    color: #a2a8ce;
    border-radius: 12px !important;
    padding: 10px 14px !important;
    font-weight: 600;
    font-size: 0.81em;
    letter-spacing: 0.35px;
    text-transform: uppercase;
    border: none !important;
    margin-bottom: 0;
    background: rgba(255,255,255,0.015) !important;
    transition: color 0.2s, background 0.2s, border-color 0.2s;
    border: 1px solid rgba(255,255,255,0.06) !important;
    flex: 1 1 155px !important;
    justify-content: center !important;
    text-align: center !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #e9ecff;
    background: rgba(255,255,255,0.06) !important;
    border-color: rgba(255,255,255,0.20) !important;
}
.stTabs [aria-selected="true"] {
    color: #ffffff !important;
    background: linear-gradient(120deg, rgba(245,166,35,0.22), rgba(245,166,35,0.09)) !important;
    border: 1px solid rgba(245,166,35,0.50) !important;
    border-radius: 12px !important;
    box-shadow: 0 0 0 1px rgba(245,166,35,0.16) inset, 0 6px 18px rgba(245,166,35,0.12) !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 24px; }

/* ── Primary button ── */
.stButton > button {
    background: linear-gradient(135deg, #f5a623 0%, #ffbf63 100%) !important;
    color: #131728 !important;
    border: none !important;
    border-radius: 9px;
    font-weight: 700;
    font-size: 0.92em;
    letter-spacing: 0.3px;
    padding: 12px 28px;
    transition: all 0.15s ease;
    width: 100%;
    box-shadow: 0 8px 22px rgba(245,166,35,0.20);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #ffb84d 0%, #ffd18b 100%) !important;
    transform: translateY(-1px);
    box-shadow: 0 12px 26px rgba(245,166,35,0.30) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Download button ── */
.stDownloadButton > button {
    background: rgba(255,255,255,0.04) !important;
    color: #c8c8e0 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px;
    font-weight: 500;
    font-size: 0.84em;
    padding: 8px 18px;
    transition: all 0.15s;
    width: auto !important;
}
.stDownloadButton > button:hover {
    background: rgba(255,255,255,0.08) !important;
    border-color: rgba(255,255,255,0.2) !important;
    color: #ffffff !important;
}

/* ── Metrics ── */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 16px 14px;
    transition: border-color 0.2s;
}
div[data-testid="metric-container"]:hover { border-color: rgba(255,255,255,0.13); }
div[data-testid="stMetricLabel"] {
    color: #8888b0 !important;
    font-size: 0.7em !important;
    text-transform: uppercase;
    letter-spacing: 1.1px;
    font-weight: 600;
}
div[data-testid="stMetricValue"] {
    color: #e8e8f8 !important;
    font-size: 1.6em !important;
    font-weight: 700 !important;
    letter-spacing: -0.3px;
}

/* ── File uploader ── */
div[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1.5px dashed rgba(255,255,255,0.1) !important;
    border-radius: 14px !important;
    transition: border-color 0.2s, background 0.2s;
}
div[data-testid="stFileUploader"]:hover {
    border-color: rgba(245,166,35,0.35) !important;
    background: rgba(245,166,35,0.015) !important;
}
div[data-testid="stFileUploaderDropzoneInstructions"] { color: #8888b0; }

/* ── Alerts ── */
div[data-testid="stAlert"] { border-radius: 10px; }
.stInfo    { background: rgba(91,141,217,0.05) !important; border-color: rgba(91,141,217,0.18) !important; }
.stWarning { background: rgba(245,166,35,0.05) !important; border-color: rgba(245,166,35,0.18) !important; }
.stSuccess { background: rgba(68,187,119,0.05) !important; border-color: rgba(68,187,119,0.18) !important; }

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.06) !important; margin: 4px 0 !important; }

/* ── Caption ── */
.stCaption, small { color: #7070a0 !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] { background: #05050f; }

/* ── Card (glass-card kept for compat) ── */
.glass-card {
    background: linear-gradient(160deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    padding: 24px;
    transition: border-color 0.2s, background 0.2s, transform 0.15s;
    box-shadow: 0 8px 28px rgba(0,0,0,0.22);
}
.glass-card:hover {
    border-color: rgba(255,255,255,0.13);
    background: linear-gradient(160deg, rgba(255,255,255,0.07), rgba(255,255,255,0.03));
    transform: translateY(-1px);
}
.top-note {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 14px;
    padding: 16px 18px;
}

/* ── Tier badge ── */
.tier-high   { color: #ff5e5e; background: rgba(255,94,94,0.08); border: 1px solid rgba(255,94,94,0.22); border-radius: 5px; padding: 2px 8px; font-size: 0.72em; font-weight: 700; letter-spacing: 0.8px; text-transform: uppercase; }
.tier-medium { color: #F5A623; background: rgba(245,166,35,0.08); border: 1px solid rgba(245,166,35,0.22); border-radius: 5px; padding: 2px 8px; font-size: 0.72em; font-weight: 700; letter-spacing: 0.8px; text-transform: uppercase; }
.tier-low    { color: #5B8DD9; background: rgba(91,141,217,0.08); border: 1px solid rgba(91,141,217,0.22); border-radius: 5px; padding: 2px 8px; font-size: 0.72em; font-weight: 700; letter-spacing: 0.8px; text-transform: uppercase; }

/* ── Action point ── */
.action-point {
    border-left: 2px solid rgba(245,166,35,0.7);
    background: rgba(245,166,35,0.03);
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    margin: 14px 0;
    font-size: 0.86em;
    color: #9898c0;
    line-height: 1.65;
}
.action-point strong { color: #F5A623; }

/* ── Section heading ── */
.section-title {
    font-family: 'Space Grotesk', 'Inter', sans-serif;
    font-size: 0.93em;
    font-weight: 700;
    color: #f0f0fa;
    letter-spacing: 0.2px;
    margin: 0 0 2px;
}
.section-sub {
    font-size: 0.78em;
    color: #7070a0;
    margin: 0 0 14px;
}

/* ── Feature chip ── */
.feature-chip {
    display: inline-block;
    background: rgba(91,141,217,0.12);
    border: 1px solid rgba(91,141,217,0.24);
    color: #d8e2ff;
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.79em;
    font-weight: 500;
    margin: 3px 2px;
}

/* ── Gaze card ── */
.gaze-card {
    border-radius: 12px;
    padding: 14px 16px;
    margin: 6px 0;
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    transition: transform 0.15s, border-color 0.2s;
}
.gaze-card:hover { transform: translateX(3px); border-color: rgba(255,255,255,0.11); }

/* ── Text input ── */
div[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 9px !important;
    color: #e8e8f8 !important;
    padding: 12px 16px !important;
    font-size: 0.95em !important;
    transition: border-color 0.2s;
}
div[data-testid="stTextInput"] input[type="password"] {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: rgba(245,166,35,0.5) !important;
    box-shadow: 0 0 0 3px rgba(245,166,35,0.07) !important;
}

/* ── Dataframe / table theme ── */
div[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    background: rgba(10, 14, 34, 0.80) !important;
}
div[data-testid="stDataFrame"] [role="grid"] {
    background: rgba(10, 14, 34, 0.80) !important;
    color: #e7ebff !important;
}
div[data-testid="stDataFrame"] [role="columnheader"] {
    background: linear-gradient(180deg, rgba(91,141,217,0.24), rgba(91,141,217,0.10)) !important;
    color: #f3f5ff !important;
    border-bottom: 1px solid rgba(255,255,255,0.16) !important;
    font-weight: 700 !important;
}
div[data-testid="stDataFrame"] [role="gridcell"] {
    background: rgba(8, 12, 30, 0.75) !important;
    color: #d9ddf8 !important;
    border-bottom: 1px solid rgba(255,255,255,0.07) !important;
}
div[data-testid="stDataFrame"] [role="row"]:hover [role="gridcell"] {
    background: rgba(20, 28, 60, 0.86) !important;
}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# DEEPGAZE SALIENCY ENGINE
# Primary: DeepGaze IIE (deep learning, ~90% accuracy)
# Fallback: Itti-Koch inspired (rule-based)
# ══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _load_deepgaze_model():
    """Load DeepGaze IIE once and cache for the session. Downloads weights on first run."""
    try:
        # Use MPS on Apple Silicon if available, else CPU
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(device)
        model.eval()
        return model, device, True
    except Exception as e:
        return None, None, False


def _norm(arr):
    mn, mx = arr.min(), arr.max()
    return np.zeros_like(arr) if mx - mn < 1e-8 else (arr - mn) / (mx - mn)


def _decode_saliency_blob(blob, target_hw):
    """Decode base64/bytes saliency image into normalized 2D map."""
    H, W = target_hw
    if blob is None:
        return None
    try:
        if isinstance(blob, str):
            raw = base64.b64decode(blob)
        elif isinstance(blob, bytes):
            raw = blob
        else:
            return None
        arr = np.frombuffer(raw, dtype=np.uint8)
        im = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if im is None:
            return None
        if im.shape[:2] != (H, W):
            im = cv2.resize(im, (W, H), interpolation=cv2.INTER_LINEAR)
        return _norm(im.astype(np.float32))
    except Exception:
        return None


def _remote_ensemble_saliency(img):
    """
    Optional high-accuracy remote inference path.
    Expected response keys (any subset):
      - saliency_map (single)
      - deepgaze_map, salicon_map, vit_map (ensemble components)
      - weights: {"deepgaze":0.55,"salicon":0.30,"vit":0.15}
    """
    if not SALIENCY_API_URL or requests is None:
        return None
    H, W = img.shape[:2]
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        return None

    try:
        resp = requests.post(
            SALIENCY_API_URL,
            files={"image": ("image.jpg", buf.tobytes(), "image/jpeg")},
            data={"models": "deepgaze,salicon,vit"},
            timeout=45,
        )
        if resp.status_code != 200:
            return None
        payload = resp.json()
    except Exception:
        return None

    # Single-map mode
    if "saliency_map" in payload:
        sm = _decode_saliency_blob(payload.get("saliency_map"), (H, W))
        if sm is not None:
            return sm, {"face_found": False, "engine": payload.get("model_used", "Remote Saliency")}

    # Ensemble mode
    dg = _decode_saliency_blob(payload.get("deepgaze_map"), (H, W))
    sc = _decode_saliency_blob(payload.get("salicon_map"), (H, W))
    vt = _decode_saliency_blob(payload.get("vit_map"), (H, W))
    if dg is None and sc is None and vt is None:
        return None

    w = payload.get("weights", {}) if isinstance(payload.get("weights"), dict) else {}
    wd = float(w.get("deepgaze", 0.55))
    ws = float(w.get("salicon", 0.30))
    wv = float(w.get("vit", 0.15))

    maps = []
    weights = []
    if dg is not None:
        maps.append(dg); weights.append(max(wd, 0.0))
    if sc is not None:
        maps.append(sc); weights.append(max(ws, 0.0))
    if vt is not None:
        maps.append(vt); weights.append(max(wv, 0.0))
    if not maps:
        return None

    total_w = sum(weights) if sum(weights) > 1e-8 else float(len(weights))
    sal = np.zeros((H, W), np.float32)
    for m, ww in zip(maps, weights):
        sal += m * (ww / total_w)
    sal = _norm(gaussian_filter(sal, sigma=max(H, W) / 120))
    sal = _sharpen_sal(sal)
    return sal, {"face_found": False, "engine": "Remote Ensemble (DeepGaze+SALICON+ViT)"}


def _center_bias(H, W):
    """Gaussian log-density center bias (people look at image centers)."""
    from scipy.special import logsumexp
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cy, cx = H / 2.0, W / 2.0
    sigma_y, sigma_x = H / 2.8, W / 2.8
    cb = -0.5 * ((yy - cy) ** 2 / sigma_y ** 2 + (xx - cx) ** 2 / sigma_x ** 2)
    cb -= logsumexp(cb)   # normalise: sum of exp = 1
    return cb.astype(np.float32)


def _face_boost(img, sal, H, W, weight=0.22):
    """Detect faces and blend a soft attention boost into saliency."""
    faces = _detect_faces(img)
    if len(faces) == 0:
        return sal, False
    face_map = np.zeros((H, W), np.float32)
    for fx, fy, fw, fh in faces:
        face_map[fy:fy+fh, fx:fx+fw] = 1.0
    face_map = gaussian_filter(face_map, sigma=max(H, W) / 22)
    face_map = _norm(face_map)
    sal = _norm((1 - weight) * sal + weight * face_map)
    return sal, True


def _detect_faces(img):
    """Best-effort face detection: MediaPipe first, Haar fallback."""
    H, W = img.shape[:2]
    faces = []

    if MEDIAPIPE_AVAILABLE:
        try:
            with mp.solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.45,
            ) as detector:
                res = detector.process(img)
            if res.detections:
                for d in res.detections:
                    b = d.location_data.relative_bounding_box
                    x = max(0, int(b.xmin * W))
                    y = max(0, int(b.ymin * H))
                    w = int(b.width * W)
                    h = int(b.height * H)
                    if w > 14 and h > 14:
                        faces.append((x, y, min(w, W - x), min(h, H - y)))
        except Exception:
            faces = []

    if not faces:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        for fx, fy, fw, fh in face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(20, 20)):
            faces.append((int(fx), int(fy), int(fw), int(fh)))
    return faces


@st.cache_resource(show_spinner=False)
def _load_yolo_model():
    """Load YOLO model once for person/object prior maps."""
    if not YOLO_AVAILABLE:
        return None
    try:
        return YOLO("yolov8n.pt")
    except Exception:
        return None


def _person_object_boost(img, sal, H, W, person_weight=0.12):
    """Boost saliency around detected people/objects when YOLO is available."""
    model = _load_yolo_model()
    if model is None:
        return sal, False
    try:
        res = model.predict(img, verbose=False, conf=0.28, iou=0.5, imgsz=min(960, max(H, W)))[0]
    except Exception:
        return sal, False

    if not hasattr(res, "boxes") or res.boxes is None or len(res.boxes) == 0:
        return sal, False

    box_map = np.zeros((H, W), np.float32)
    person_found = False
    for b in res.boxes:
        cls_id = int(b.cls.item()) if hasattr(b, "cls") else -1
        conf = float(b.conf.item()) if hasattr(b, "conf") else 0.0
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        # Stronger prior for person class (COCO class 0), lighter for others.
        w = (0.55 if cls_id == 0 else 0.22) * max(0.0, min(conf, 1.0))
        if cls_id == 0:
            person_found = True
        box_map[y1:y2, x1:x2] = np.maximum(box_map[y1:y2, x1:x2], w)

    if box_map.max() <= 0:
        return sal, False

    box_map = gaussian_filter(box_map, sigma=max(H, W) / 26)
    box_map = _norm(box_map)
    sal = _norm((1 - person_weight) * sal + person_weight * box_map)
    return sal, person_found


def _sharpen_sal(sal):
    """Balanced contrast preset: keep dark regions clearer, hotspots readable."""
    p2, p98 = np.percentile(sal, 2), np.percentile(sal, 98)
    sal = np.clip((sal - p2) / (p98 - p2 + 1e-8), 0, 1)
    # Less aggressive than before (0.65): preserves low-attention contrast.
    sal = np.power(sal, 0.90)
    return _norm(sal)


def compute_saliency(img):
    """
    Returns (sal_map [0,1], meta_dict).
    Uses DeepGaze IIE when available, Itti-Koch fallback otherwise.
    """
    H, W = img.shape[:2]

    # ── Remote ensemble path (DeepGaze/SALICON/ViT) ─────────
    remote = _remote_ensemble_saliency(img)
    if remote is not None:
        return remote

    # ── DeepGaze path ──────────────────────────────────────
    if DEEPGAZE_AVAILABLE:
        model, device, ok = _load_deepgaze_model()
        if ok:
            try:
                # Run at full res up to 1024px — more detail than 768
                max_dim = 1024
                scale   = min(max_dim / H, max_dim / W, 1.0)
                rH, rW  = int(H * scale), int(W * scale)
                img_r   = cv2.resize(img, (rW, rH), interpolation=cv2.INTER_AREA)

                img_t = torch.tensor(
                    img_r.transpose(2, 0, 1)[np.newaxis], dtype=torch.float32
                ).to(device)
                cb_t  = torch.tensor(
                    _center_bias(rH, rW)[np.newaxis, np.newaxis]
                ).to(device)

                with torch.no_grad():
                    log_den = model(img_t, cb_t)

                sal = log_den.squeeze().cpu().numpy()
                sal = np.exp(sal - sal.max())

                if scale < 1.0:
                    from scipy.ndimage import zoom
                    sal = zoom(sal, (H / rH, W / rW), order=1)

                # Lighter smoothing preserves spatial sharpness
                sal = gaussian_filter(sal, sigma=max(H, W) / 110)
                sal = _norm(sal)

                # AI priors: stronger face detection + optional person/object prior.
                sal, face_found = _face_boost(img, sal, H, W, weight=0.16)
                sal, person_found = _person_object_boost(img, sal, H, W, person_weight=0.10)

                # Sharpen: percentile clip + gamma to make hot-spots pop
                sal = _sharpen_sal(sal)

                return sal, {"face_found": face_found or person_found, "engine": "DeepGaze IIE+"}
            except Exception:
                pass   # fall through to fallback

    # ── Fallback: Itti-Koch (enhanced) ────────────────────
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Multi-scale intensity centre-surround
    intensity_maps = []
    for s1, s2 in [(1, 8), (2, 16), (3, 28)]:
        d = np.abs(
            gaussian_filter(gray.astype(np.float32) / 255, s1) -
            gaussian_filter(gray.astype(np.float32) / 255, s2)
        )
        intensity_maps.append(_norm(d))
    intensity = _norm(sum(intensity_maps) / 3)

    # Opponent colour contrast (R/G + B/Y)
    f = img.astype(np.float32) / 255
    R, G, B = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    rg = np.abs(R - G) / (R + G + 1e-8)
    by = np.abs(B - 0.5 * (R + G)) / (B + 0.5 * (R + G) + 1e-8)
    color = _norm(gaussian_filter(_norm(rg + by), 5))

    # HSV saturation channel — pops vibrant objects
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    sat = _norm(gaussian_filter(hsv[:, :, 1] / 255.0, 4))

    # Multi-threshold Canny edges
    e1 = cv2.Canny(gray, 30,  90) / 255.0
    e2 = cv2.Canny(gray, 60, 160) / 255.0
    e3 = cv2.Canny(gray, 100, 220) / 255.0
    edges = _norm(gaussian_filter(np.maximum(np.maximum(e1, e2), e3), 4))

    # Face detection
    faces = _detect_faces(img)
    face_map  = np.zeros((H, W), np.float32)
    face_found = len(faces) > 0
    for fx, fy, fw, fh in faces:
        face_map[fy:fy+fh, fx:fx+fw] = 1.0
    if face_found:
        face_map = gaussian_filter(face_map, sigma=max(H, W) / 16)
        face_map = _norm(face_map)

    if face_found:
        sal = 0.12*intensity + 0.18*color + 0.14*sat + 0.10*edges + 0.46*face_map
    else:
        sal = 0.26*intensity + 0.28*color + 0.24*sat + 0.22*edges

    # Optional person/object prior improves fallback quality on ads/people visuals.
    sal, person_found = _person_object_boost(img, sal, H, W, person_weight=0.12)
    sal = gaussian_filter(sal, sigma=max(H, W) / 65)
    sal = _sharpen_sal(sal)
    return sal, {"face_found": face_found or person_found, "engine": "Itti-Koch+"}


# ══════════════════════════════════════════════════════════════
# VISUALIZATION FUNCTIONS
# ══════════════════════════════════════════════════════════════

def _attention_colormap():
    """Custom RGBA colormap: transparent navy → cyan/green → yellow → red."""
    import matplotlib.colors as mcolors
    # (R, G, B, A): tuned to the provided spectrum with stronger transparency.
    # Low-saliency stays mostly transparent so the base image remains visible.
    colors = [
        (0.00, 0.00, 0.02, 0.00),   # 0%   transparent
        (0.00, 0.00, 0.12, 0.00),   # 14%  deep navy, still transparent
        (0.00, 0.06, 0.78, 0.16),   # 32%  blue
        (0.04, 0.74, 0.98, 0.30),   # 50%  cyan
        (0.56, 0.95, 0.24, 0.46),   # 66%  green-yellow
        (1.00, 0.95, 0.00, 0.58),   # 80%  yellow
        (1.00, 0.52, 0.00, 0.72),   # 90%  orange
        (0.92, 0.00, 0.00, 0.84),   # 100% red
    ]
    positions = [0.0, 0.14, 0.32, 0.50, 0.66, 0.80, 0.90, 1.0]
    return mcolors.LinearSegmentedColormap.from_list(
        "attention", list(zip(positions, colors))
    )

ATTENTION_CMAP = _attention_colormap()


def generate_heatmap(img, sal_map):
    """
    True attention heatmap: transparent where no attention,
    navy/blue → cyan/green → yellow/red where eyes are drawn.
    """
    h, w = img.shape[:2]
    base = img.astype(np.float32) / 255.0

    # Apply custom RGBA colormap
    rgba = ATTENTION_CMAP(sal_map)          # shape (H, W, 4), values [0,1]
    heat_rgb = rgba[:, :, :3]
    alpha    = rgba[:, :, 3:4]              # per-pixel alpha

    # Alpha-composite heat over image
    blended = base * (1 - alpha) + heat_rgb * alpha
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
    return blended


def generate_hotspot(img, sal_map):
    """
    More accurate hotspot rendering:
    - adaptive tier thresholds per image
    - lighter smoothing so regions follow true saliency structure
    - labels anchored to local maxima (not geometric centers)
    """
    result = img.copy().astype(np.float32)
    H, W = img.shape[:2]
    min_dim = min(H, W)

    # Light smoothing keeps peaks but reduces single-pixel noise.
    smooth = cv2.GaussianBlur(sal_map.astype(np.float32), (0, 0), sigmaX=max(1.2, min_dim / 260))

    # Adaptive thresholds make tiers more stable across very flat vs very peaky maps.
    q40 = float(np.quantile(smooth, 0.55))
    q70 = float(np.quantile(smooth, 0.82))
    t_low = max(0.08, min(0.40, q40))
    t_mid = max(t_low + 0.06, min(0.70, q70))
    flat_smooth = np.sort(smooth.flatten())

    # Smaller morphology footprint preserves shape accuracy.
    k = max(5, (min_dim // 120) | 1)  # odd
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    min_area = max(120, int(H * W * 0.0008))

    tiers = [
        (t_low, t_mid, np.array([20, 100, 245], np.float32), 0.24),   # blue
        (t_mid, 0.86, np.array([0, 210, 130], np.float32), 0.34),     # green/cyan
        (0.86, 1.01, np.array([235, 35, 0], np.float32), 0.50),       # red
    ]

    for lo, hi, color_f, base_alpha in tiers:
        raw_mask = ((smooth >= lo) & (smooth < hi)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Soft alpha follows saliency intensity inside the tier for better fidelity.
        tier_strength = np.clip((smooth - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=max(1.0, k * 0.8)) / 255.0
        alpha_map = (0.45 + 0.55 * tier_strength) * soft * base_alpha

        for c in range(3):
            result[:, :, c] = result[:, :, c] * (1 - alpha_map) + color_f[c] * alpha_map

        contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        res_uint = np.clip(result, 0, 255).astype(np.uint8)

        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue
            cv2.drawContours(res_uint, [cnt], -1, tuple(color_f.astype(int).tolist()), 2)

            # Label at local max inside contour; score uses robust local stats
            # + global percentile calibration so 100% is rare.
            contour_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.drawContours(contour_mask, [cnt], -1, 255, -1)
            ys, xs = np.where(contour_mask > 0)
            if len(xs) == 0:
                continue
            local_vals = smooth[ys, xs]
            p_idx = int(np.argmax(local_vals))
            cx_, cy_ = int(xs[p_idx]), int(ys[p_idx])
            local_peak = float(local_vals[p_idx])
            local_p90 = float(np.percentile(local_vals, 90))
            # Percentile rank in the whole image makes labels comparable.
            global_rank = float(np.searchsorted(flat_smooth, local_peak, side="right")) / max(len(flat_smooth), 1)
            prob = (
                0.55 * (global_rank * 100.0) +
                0.30 * (local_peak * 100.0) +
                0.15 * (local_p90 * 100.0)
            )
            prob = int(np.clip(prob, 1, 98))
            label = f"{prob}%"
            fs = max(0.46, min_dim / 1200)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, fs, 1)
            pad = 4
            x1 = max(0, cx_ - tw // 2 - pad)
            y1 = max(0, cy_ - th - pad * 2)
            x2 = min(W - 1, cx_ + tw // 2 + pad)
            y2 = min(H - 1, cy_ + pad)
            cv2.rectangle(res_uint, (x1, y1), (x2, y2), (10, 10, 10), -1)
            cv2.putText(
                res_uint, label, (x1 + pad, y2 - pad),
                cv2.FONT_HERSHEY_DUPLEX, fs,
                tuple(color_f.astype(int).tolist()), 1, cv2.LINE_AA
            )
        result = res_uint.astype(np.float32)

    return np.clip(result, 0, 255).astype(np.uint8)


def _calibrated_prob(sal_map, value, local_values=None):
    """Map raw saliency to calibrated percentage; keeps 100% rare."""
    vals = np.clip(sal_map.astype(np.float32).flatten(), 0.0, 1.0)
    vals_sorted = np.sort(vals)
    rank = float(np.searchsorted(vals_sorted, float(value), side="right")) / max(len(vals_sorted), 1)
    base = rank * 100.0
    if local_values is not None and len(local_values) > 0:
        lp = float(np.max(local_values)) * 100.0
        lq = float(np.percentile(local_values, 90)) * 100.0
        base = 0.55 * base + 0.30 * lp + 0.15 * lq
    return float(np.clip(base, 1.0, 98.0))


def get_gaze_sequence(sal_map, n=5):
    """
    Non-maximum suppression peak detection with Gaussian suppression.
    min_dist scales with image size so all resolutions work equally well.
    Returns list of (x, y, probability%) in viewing order.
    """
    h, w = sal_map.shape
    min_dist = max(50, int(min(w, h) * 0.13))

    # Pre-smooth slightly so we find region centres, not pixel-noise peaks
    smooth = gaussian_filter(sal_map, sigma=max(h, w) / 120)
    canvas = smooth.copy()
    points = []

    for _ in range(n):
        yx = np.unravel_index(np.argmax(canvas), canvas.shape)
        y, x = yx
        # Use local neighborhood + global calibration for realistic percentages.
        x1 = max(0, x - 8)
        x2 = min(w, x + 9)
        y1 = max(0, y - 8)
        y2 = min(h, y + 9)
        local = sal_map[y1:y2, x1:x2].flatten()
        prob = _calibrated_prob(sal_map, sal_map[y, x], local)
        points.append((x, y, prob))

        # Gaussian suppression: soft falloff around found peak
        yy, xx = np.mgrid[0:h, 0:w]
        dist_sq = (yy - y) ** 2 + (xx - x) ** 2
        suppress = np.exp(-dist_sq / (2 * (min_dist * 0.6) ** 2))
        canvas = canvas * (1 - suppress)

    return points


def draw_gaze_sequence(img, points):
    """
    Gaze path with anti-aliased circles (PIL) + arrows (OpenCV).
    """
    from PIL import Image as PImage, ImageDraw as PDraw

    result  = img.copy().astype(np.uint8)
    H, W    = result.shape[:2]
    r       = max(18, min(W, H) // 30)   # circle radius scales with image

    pt_colors = [
        (230, 50,  50,  230),   # 1 — red
        ( 50, 210, 80,  230),   # 2 — green
        ( 50, 100, 240, 230),   # 3 — blue
        (240, 165,   0, 230),   # 4 — orange
        (160, 100, 240, 230),   # 5 — violet
    ]
    coords = [(x, y) for x, y, _ in points]

    # ── Arrows (OpenCV, drawn first so circles sit on top) ──
    for i in range(1, len(coords)):
        x0, y0 = coords[i - 1]
        x1, y1 = coords[i]
        # Shorten line so it ends at circle edge
        dx, dy = x1 - x0, y1 - y0
        dist   = max((dx**2 + dy**2) ** 0.5, 1)
        ux, uy = dx / dist, dy / dist
        sx = int(x0 + ux * (r + 2))
        sy = int(y0 + uy * (r + 2))
        ex = int(x1 - ux * (r + 6))
        ey = int(y1 - uy * (r + 6))
        cv2.arrowedLine(result, (sx, sy), (ex, ey),
                        (255, 255, 255), max(1, r // 10),
                        tipLength=max(0.2, 12 / max(dist, 1)),
                        line_type=cv2.LINE_AA)

    # ── Circles (PIL for smooth anti-aliasing) ──────────────
    pil = PImage.fromarray(result)
    draw = PDraw.Draw(pil, "RGBA")

    for i, (x, y, prob) in enumerate(points):
        cr, cg, cb, ca = pt_colors[i % len(pt_colors)]
        # Glow ring
        draw.ellipse([x - r - 6, y - r - 6, x + r + 6, y + r + 6],
                     outline=(cr, cg, cb, 80), width=4)
        # Filled circle with slight transparency
        draw.ellipse([x - r, y - r, x + r, y + r],
                     fill=(cr, cg, cb, ca), outline=(255, 255, 255, 200), width=2)

    result = np.array(pil.convert("RGB"))

    # ── Numbers (OpenCV on top) ─────────────────────────────
    font_scale = max(0.55, r / 22)
    for i, (x, y, _) in enumerate(points):
        num = str(i + 1)
        (tw, th), _ = cv2.getTextSize(num, cv2.FONT_HERSHEY_DUPLEX, font_scale, 1)
        cv2.putText(result, num, (x - tw // 2, y + th // 2),
                    cv2.FONT_HERSHEY_DUPLEX, font_scale,
                    (255, 255, 255), 1, cv2.LINE_AA)

    return result


def calculate_aoi(sal_map, boxes, scale_x=1.0, scale_y=1.0):
    """
    boxes: list of (x1,y1,x2,y2) in canvas coordinates.
    Returns list of dicts with label, scaled box, and seen probability.
    """
    H, W   = sal_map.shape
    results = []

    for raw_box in boxes:
        x1, y1, x2, y2 = raw_box
        sx1 = max(0, min(int(x1 * scale_x), W - 1))
        sy1 = max(0, min(int(y1 * scale_y), H - 1))
        sx2 = max(0, min(int(x2 * scale_x), W))
        sy2 = max(0, min(int(y2 * scale_y), H))

        if sx2 <= sx1 or sy2 <= sy1:
            continue

        roi  = sal_map[sy1:sy2, sx1:sx2]
        if roi.size == 0:
            continue

        # AOI "Seen %" should represent how much of the selected region is likely noticed,
        # not only the single hottest pixel. We combine area coverage at multiple saliency
        # levels with average saliency for a more stable, human-intuitive score.
        roi = roi.astype(np.float32)
        mean_sal = float(np.mean(roi))                 # overall attention density
        cov_low  = float(np.mean(roi >= 0.35))         # broad visibility
        cov_mid  = float(np.mean(roi >= 0.55))         # meaningful visibility
        cov_high = float(np.mean(roi >= 0.75))         # strong visibility

        prob = (
            0.45 * mean_sal +
            0.20 * cov_low +
            0.20 * cov_mid +
            0.15 * cov_high
        ) * 100.0
        prob = float(np.clip(prob, 0.0, 100.0))
        peak = _calibrated_prob(sal_map, float(np.max(roi)), roi.flatten())

        results.append({
            "label": string.ascii_uppercase[len(results)],
            "box":   (sx1, sy1, sx2, sy2),
            "prob":  round(prob, 1),
            "peak":  round(peak, 1),
        })

    return results


def draw_aoi_regions(img, aoi_results):
    """Color-coded bounding boxes with probability labels."""
    result = img.copy().astype(np.uint8)

    for r in aoi_results:
        x1, y1, x2, y2 = r["box"]
        p = r["prob"]
        color = (255, 55, 55) if p >= 70 else (255, 215, 0) if p >= 40 else (0, 160, 255)

        cv2.rectangle(result, (x1, y1), (x2, y2), color, 3)

        text = f"{r['label']}: {p:.0f}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(result, (x1, y1 - th - 12), (x1 + tw + 10, y1), color, -1)
        cv2.putText(result, text, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    return result


def compute_clarity_score(sal_map):
    """Return clarity score (0-100) + supporting components."""
    sal = np.clip(sal_map.astype(np.float32), 0.0, 1.0)
    flat = sal.flatten()
    if flat.size == 0:
        return {"score": 0.0, "focus_ratio": 0.0, "contrast": 0.0, "peak": 0.0}

    focus_ratio = float(np.mean(flat >= np.percentile(flat, 90)))
    peak = float(np.max(flat))
    q50 = float(np.percentile(flat, 50))
    contrast = float(np.clip((peak - q50) / max(peak, 1e-6), 0.0, 1.0))
    entropy = float(-np.sum((flat / (flat.sum() + 1e-8)) * np.log(flat / (flat.sum() + 1e-8) + 1e-8)))
    entropy_norm = float(np.clip(entropy / max(np.log(flat.size + 1e-8), 1e-8), 0.0, 1.0))

    score = (
        0.40 * contrast +
        0.30 * (1.0 - focus_ratio) +
        0.20 * peak +
        0.10 * (1.0 - entropy_norm)
    ) * 100.0
    return {
        "score": round(float(np.clip(score, 0.0, 100.0)), 1),
        "focus_ratio": round(focus_ratio * 100.0, 1),
        "contrast": round(contrast * 100.0, 1),
        "peak": round(peak * 100.0, 1),
    }


def detect_top_elements(sal_map, max_items=5):
    """Detect top attention regions and rank by combined peak+mass."""
    sal = np.clip(sal_map.astype(np.float32), 0.0, 1.0)
    thr = max(0.35, float(np.percentile(sal, 80)))
    mask = (sal >= thr).astype(np.uint8) * 255
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = []
    total_mass = float(sal.sum()) + 1e-8
    for i in range(1, num):
        x, y, w, h, a = stats[i]
        if a < max(80, int(0.0007 * sal.size)):
            continue
        comp_mask = labels == i
        vals = sal[comp_mask]
        peak = float(vals.max()) * 100.0
        mass = float(vals.sum() / total_mass) * 100.0
        score = 0.65 * peak + 0.35 * mass
        areas.append({
            "box": (int(x), int(y), int(x + w), int(y + h)),
            "peak": round(peak, 1),
            "share": round(mass, 1),
            "score": round(score, 1),
        })
    areas.sort(key=lambda d: d["score"], reverse=True)
    for idx, item in enumerate(areas[:max_items], start=1):
        item["rank"] = idx
    return areas[:max_items]


def draw_top_elements_overlay(img, elements):
    out = img.copy().astype(np.uint8)
    colors = [(235, 35, 0), (245, 166, 35), (60, 191, 191), (91, 141, 217), (68, 187, 119)]
    for i, e in enumerate(elements):
        x1, y1, x2, y2 = e["box"]
        color = colors[i % len(colors)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"#{e['rank']}  {e['peak']:.0f}%"
        cv2.rectangle(out, (x1, max(0, y1 - 24)), (x1 + 130, y1), color, -1)
        cv2.putText(out, label, (x1 + 6, y1 - 7), cv2.FONT_HERSHEY_DUPLEX, 0.52, (10, 10, 10), 1, cv2.LINE_AA)
    return out


def compute_face_pull(img, sal_map):
    """Return attention share captured by detected faces/person proxies."""
    faces = _detect_faces(img)
    h, w = sal_map.shape
    face_mask = np.zeros((h, w), dtype=np.uint8)
    for (x, y, w, h) in faces:
        cv2.rectangle(face_mask, (x, y), (x + w, y + h), 255, -1)
    sal = np.clip(sal_map.astype(np.float32), 0.0, 1.0)
    total = float(sal.sum()) + 1e-8
    face_share = float(sal[face_mask > 0].sum() / total) * 100.0
    return {
        "count": int(len(faces)),
        "share": round(face_share, 1),
        "faces": [tuple(map(int, f)) for f in faces],
    }


def compare_variant_metrics(metrics_a, metrics_b, name_a="A", name_b="B"):
    """Build compact comparison rows and overall winner."""
    keys = [
        ("Clarity", metrics_a["clarity"]["score"], metrics_b["clarity"]["score"]),
        ("Peak Attention", metrics_a["peak"], metrics_b["peak"]),
        ("Face Pull", metrics_a["face_pull"]["share"], metrics_b["face_pull"]["share"]),
    ]
    rows = []
    wins = {name_a: 0, name_b: 0}
    for label, va, vb in keys:
        winner = name_a if va > vb else name_b if vb > va else "Tie"
        if winner in wins:
            wins[winner] += 1
        rows.append({"Metric": label, name_a: f"{va:.1f}", name_b: f"{vb:.1f}", "Winner": winner})
    overall = name_a if wins[name_a] > wins[name_b] else name_b if wins[name_b] > wins[name_a] else "Tie"
    return rows, overall


def compute_attention_balance(sal_map):
    """Compute high-level composition balance and distraction metrics."""
    sal = np.clip(sal_map.astype(np.float32), 0.0, 1.0)
    h, w = sal.shape
    total = float(sal.sum()) + 1e-8

    # Center window (middle 50% x 50%)
    y1, y2 = int(0.25 * h), int(0.75 * h)
    x1, x2 = int(0.25 * w), int(0.75 * w)
    center = float(sal[y1:y2, x1:x2].sum())
    center_share = center / total * 100.0
    edge_share = 100.0 - center_share

    left_share = float(sal[:, : w // 2].sum() / total) * 100.0
    right_share = 100.0 - left_share
    top_share = float(sal[: h // 2, :].sum() / total) * 100.0
    bottom_share = 100.0 - top_share

    # Distraction index: attention outside top 3 strongest connected regions.
    mask = (sal >= np.percentile(sal, 80)).astype(np.uint8) * 255
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    masses = []
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] < max(50, int(0.0006 * sal.size)):
            continue
        masses.append(float(sal[labels == i].sum()))
    masses.sort(reverse=True)
    top_mass = sum(masses[:3]) if masses else 0.0
    distraction = (1.0 - (top_mass / total)) * 100.0

    return {
        "center_share": round(center_share, 1),
        "edge_share": round(edge_share, 1),
        "left_share": round(left_share, 1),
        "right_share": round(right_share, 1),
        "top_share": round(top_share, 1),
        "bottom_share": round(bottom_share, 1),
        "distraction": round(float(np.clip(distraction, 0.0, 100.0)), 1),
    }


def draw_attention_balance_overlay(img):
    """Overlay compositional guides to interpret balance metrics."""
    out = img.copy().astype(np.uint8)
    h, w = out.shape[:2]
    # Rule-of-thirds
    c = (130, 150, 215)
    cv2.line(out, (w // 3, 0), (w // 3, h), c, 1, cv2.LINE_AA)
    cv2.line(out, (2 * w // 3, 0), (2 * w // 3, h), c, 1, cv2.LINE_AA)
    cv2.line(out, (0, h // 3), (w, h // 3), c, 1, cv2.LINE_AA)
    cv2.line(out, (0, 2 * h // 3), (w, 2 * h // 3), c, 1, cv2.LINE_AA)
    # Center box
    cv2.rectangle(out, (int(0.25 * w), int(0.25 * h)), (int(0.75 * w), int(0.75 * h)), (245, 166, 35), 2)
    return out


# ── Helpers ────────────────────────────────────────────────

def arr_to_png_bytes(arr):
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _pdf_safe(txt):
    """Replace Unicode chars not supported by PDF standard Helvetica font."""
    return (txt
        .replace("\u2013", "-").replace("\u2014", "-")
        .replace("\u2018", "'").replace("\u2019", "'")
        .replace("\u201c", '"').replace("\u201d", '"')
        .replace("\u00b7", ".").replace("\u2022", "-")
        .replace("\u00a0", " ").replace("\u2026", "...")
    )


def colorbar_figure():
    """Vertical colorbar matching the attention heatmap palette."""
    fig, ax = plt.subplots(figsize=(1.1, 5))
    grad = np.linspace(1, 0, 256).reshape(256, 1)
    ax.imshow(grad, aspect="auto", cmap=ATTENTION_CMAP)
    ax.set_xticks([])
    ax.set_yticks([0, 64, 128, 192, 255])
    ax.set_yticklabels(["100%", "75%", "50%", "25%", "0%"],
                       color="white", fontsize=9)
    for spine in ax.spines.values():
        spine.set_color("#1a1a2e")
    fig.patch.set_facecolor("#080810")
    ax.set_facecolor("#080810")
    return fig


# ══════════════════════════════════════════════════════════════
# PDF EXPORT
# ══════════════════════════════════════════════════════════════

def export_pdf(original, heatmap_img, hotspot_img, gaze_img,
               aoi_img, aoi_results, gaze_points):
    if not FPDF_AVAILABLE:
        return None

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=14)

    # Save images to temp files
    tmp = []
    def _save(arr):
        f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        Image.fromarray(arr.astype(np.uint8)).save(f.name)
        tmp.append(f.name)
        return f.name

    orig_p  = _save(original)
    heat_p  = _save(heatmap_img)
    hot_p   = _save(hotspot_img)
    gaze_p  = _save(gaze_img)
    aoi_p   = _save(aoi_img) if aoi_img is not None else None

    def clean_page():
        pdf.add_page()
        # Top accent and simple header band.
        pdf.set_fill_color(245, 166, 35)
        pdf.rect(0, 0, 210, 3, "F")
        pdf.set_fill_color(248, 250, 252)
        pdf.rect(0, 3, 210, 17, "F")

    def heading(title, subtitle=""):
        pdf.set_xy(12, 8)
        pdf.set_font("Helvetica", "B", 15)
        pdf.set_text_color(24, 30, 42)
        pdf.cell(0, 6, _pdf_safe(title), ln=True)
        if subtitle:
            pdf.set_x(12)
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(96, 106, 122)
            pdf.cell(0, 5, _pdf_safe(subtitle), ln=True)
        pdf.ln(4)

    def body(txt):
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(65, 74, 86)
        pdf.multi_cell(0, 5.5, _pdf_safe(txt))

    def image_block(path, y=55, h=130):
        pdf.set_draw_color(226, 232, 240)
        pdf.set_line_width(0.3)
        pdf.rect(10, y - 2, 190, h + 4)
        pdf.image(path, x=12, y=y, w=186, h=h)

    # Cover
    clean_page()
    heading("AI GAZE REPORT", "Predictive visual attention analysis")
    body("Generated by Elastic Tree AI Gaze. This report summarizes expected first-glance attention in the first 3-5 seconds.")
    image_block(orig_p, y=46, h=140)

    # Heat map
    clean_page()
    heading("HEAT MAP", "Probability overlay of visual fixation")
    body(
        "Higher intensity colors indicate stronger expected attention.\n"
        "Use this view to verify whether the key brand elements sit in high-visibility zones."
    )
    image_block(heat_p, y=56, h=132)

    # Hot spot
    clean_page()
    heading("HOT SPOT", "Tiered attention zones")
    body(
        "Hot Spot simplifies attention into low, medium, and high regions.\n"
        "Key assets such as logo, product, and CTA should ideally appear in high-attention regions."
    )
    image_block(hot_p, y=56, h=132)

    # Gaze sequence
    clean_page()
    heading("GAZE SEQUENCE", "Predicted first viewing order")
    seq_lines = []
    for i, (x, y, prob) in enumerate(gaze_points):
        tier = "HIGH" if prob >= 70 else "MEDIUM" if prob >= 40 else "LOW"
        seq_lines.append(f"Point {i+1}: ({x}, {y})  -  {prob:.0f}% ({tier})")
    body("Top 5 predicted fixation points in the first 3-5 seconds:\n" + "\n".join(seq_lines))
    image_block(gaze_p, y=70, h=118)

    # AOI (optional)
    if aoi_p is not None and aoi_results:
        clean_page()
        heading("AREA OF ATTENTION", "Manually selected regions and estimated visibility")
        body("User-defined areas with estimated visibility percentages:")
        pdf.ln(1)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(45, 55, 72)
        pdf.cell(30, 6, "Region", border=1)
        pdf.cell(26, 6, "Seen %", border=1)
        pdf.cell(24, 6, "Peak %", border=1)
        pdf.cell(30, 6, "Tier", border=1)
        pdf.cell(95, 6, "Coordinates (x1,y1,x2,y2)", border=1, ln=True)
        pdf.set_font("Helvetica", "", 9)
        for r in aoi_results:
            tier = "HIGH" if r["prob"] >= 70 else "MEDIUM" if r["prob"] >= 40 else "LOW"
            coords = ",".join(map(str, r["box"]))
            pdf.cell(30, 6, f"Region {r['label']}", border=1)
            pdf.cell(26, 6, f"{r['prob']:.1f}%", border=1)
            pdf.cell(24, 6, f"{r['peak']:.1f}%", border=1)
            pdf.cell(30, 6, tier, border=1)
            pdf.cell(95, 6, coords, border=1, ln=True)
        image_block(aoi_p, y=106, h=84)

    # Cleanup
    for f in tmp:
        try:
            os.unlink(f)
        except OSError:
            pass

    return bytes(pdf.output())


# ══════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════

def _et_wordmark(size="1.5em", align="left"):
    """Elastic Tree logo image as inline HTML."""
    logo_candidates = [
        os.path.join(os.path.dirname(__file__), "elastic_tree_logo.png"),
        os.path.join(os.path.dirname(__file__), "assets", "elastic_tree_logo.png"),
    ]
    for logo_path in logo_candidates:
        if not os.path.exists(logo_path):
            continue
        try:
            with open(logo_path, "rb") as fh:
                raw = fh.read()
            # Validate image bytes so we don't render broken <img> tags.
            arr = np.frombuffer(raw, dtype=np.uint8)
            if len(arr) < 64 or cv2.imdecode(arr, cv2.IMREAD_UNCHANGED) is None:
                continue
            encoded = base64.b64encode(raw).decode("ascii")
            return (
                f"<span style='display:inline-block;text-align:{align};'>"
                f"<img src='data:image/png;base64,{encoded}' "
                f"style='height:{size};width:auto;display:inline-block;vertical-align:middle;' "
                f"alt='Elastic Tree logo'>"
                "</span>"
            )
        except OSError:
            continue

    # Fallback to text logo if image cannot be loaded
    return (
        f"<span style='font-size:{size};font-weight:900;font-family:Arial,sans-serif;"
        f"text-align:{align};display:block;letter-spacing:-0.5px;'>"
        "<span style='color:#3D3587;'>E</span>"
        "<span style='color:#3CBFBF;'>l</span>"
        "<span style='color:#5B8DD9;'>a</span>"
        "<span style='color:#3D3587;'>s</span>"
        "<span style='color:#3CBFBF;'>t</span>"
        "<span style='color:#5B8DD9;'>i</span>"
        "<span style='color:#3D3587;'>c</span>"
        "<span style='color:#F5A623;'>&thinsp;T</span>"
        "<span style='color:#44BB77;'>r</span>"
        "<span style='color:#3D3587;'>e</span>"
        "<span style='color:#F5A623;'>e</span>"
        "</span>"
    )


# ══════════════════════════════════════════════════════════════
# LANDING PAGE  (pre-auth)
# ══════════════════════════════════════════════════════════════

def _landing_page():
    st.markdown("""
    <style>
    html, body, .stApp { background: #07071c !important; }
    .lp-feat {
        background: rgba(255,255,255,0.025);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px;
        padding: 14px 14px;
        height: 100%;
        transition: border-color 0.2s, background 0.2s;
    }
    .lp-feat:hover { border-color: rgba(255,255,255,0.14); background: rgba(255,255,255,0.04); }
    .lp-tight { max-width: 980px; margin: 0 auto; }
    </style>
    """, unsafe_allow_html=True)

    # ── Hero ─────────────────────────────────────────────────
    st.markdown(
        "<div class='lp-tight' style='text-align:center;padding:22px 12px 14px;position:relative;'>"
        # Radial glow behind title
        "<div style='position:absolute;top:0;left:50%;transform:translateX(-50%);"
        "width:560px;height:240px;background:radial-gradient(ellipse at 50% 30%,"
        "rgba(245,166,35,0.055) 0%,rgba(61,53,135,0.04) 45%,transparent 72%);"
        "pointer-events:none;'></div>"
        # ET wordmark (subtle)
        f"<div style='margin-bottom:14px;opacity:0.75;position:relative;z-index:1;'>{_et_wordmark('1.25em','center')}</div>"
        # Badge pill
        "<div style='position:relative;z-index:1;display:inline-flex;align-items:center;"
        "gap:6px;background:rgba(245,166,35,0.07);border:1px solid rgba(245,166,35,0.18);"
        "border-radius:24px;padding:4px 12px;margin-bottom:14px;"
        "font-size:0.68em;color:#F5A623;font-weight:600;letter-spacing:1px;'>"
        "&#x2726; Predictive Eye Tracking"
        "</div><br style='line-height:0;'>"
        # Gradient headline
        "<div style='position:relative;z-index:1;font-family:Playfair Display,Space Grotesk,Inter,serif;"
        "font-size:3.8em;font-weight:800;line-height:0.9;letter-spacing:-2px;margin-bottom:10px;"
        "background:linear-gradient(130deg,#ffffff 25%,#ededff 50%,#F5A623 80%);"
        "-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
        "background-clip:text;'>AI GAZE"
        "<sup style='font-size:0.22em;letter-spacing:0;vertical-align:super;"
        "background:none;-webkit-text-fill-color:#F5A623;'>&#8482;</sup></div>"
        # Sub-label
        "<div style='position:relative;z-index:1;font-size:0.72em;color:#3CBFBF;"
        "font-weight:600;letter-spacing:3px;text-transform:uppercase;margin-bottom:8px;'>"
        "Predictive Eye Tracking</div>"
        # Tagline
        "<div style='position:relative;z-index:1;font-size:0.9em;color:#8888b0;"
        "max-width:500px;margin:0 auto;line-height:1.5;'>"
        "See what gets attention in the first 3&#8211;5 seconds before launch.</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Compact feature grid (6) ─────────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    f4, f5, f6 = st.columns(3)
    feat_data = [
        ("HM", ET_GOLD, "Heat Map", "See strongest attention zones."),
        ("HS", ET_TEAL, "Hot Spot", "Check low/medium/high impact."),
        ("GS", ET_BLUE, "Gaze Sequence", "Preview first-glance path."),
        ("CS", "#F5A623", "Clarity Score", "Measure visual focus quality."),
        ("T5", "#8fd0ff", "Top 5 Elements", "Rank key attention regions."),
        ("AB", "#8fffb3", "Attention Balance", "Check center-edge distribution."),
    ]
    for col, (icon, color, title, desc) in zip([f1, f2, f3, f4, f5, f6], feat_data):
        with col:
            st.markdown(
                f"<div class='lp-feat'>"
                f"<div style='width:36px;height:36px;border-radius:10px;"
                f"background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.09);"
                f"display:flex;align-items:center;justify-content:center;"
                f"font-size:0.82em;font-weight:700;letter-spacing:0.8px;margin-bottom:8px;'>{icon}</div>"
                f"<div style='font-family:Space Grotesk,Inter,sans-serif;"
                f"font-weight:700;color:{color};font-size:0.84em;margin-bottom:4px;'>{title}</div>"
                f"<div style='color:#9090c0;font-size:0.75em;line-height:1.45;'>{desc}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── Sign-in ───────────────────────────────────────────────
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    _, sc, _ = st.columns([1, 2, 1])
    with sc:
        st.markdown(
            "<div style='background:rgba(255,255,255,0.025);"
            "border:1px solid rgba(255,255,255,0.1);"
            "border-radius:20px;padding:26px 24px;text-align:center;'>"
            "<div style='font-family:Space Grotesk,Inter,sans-serif;"
            "font-size:1.0em;font-weight:700;color:#f0f0fa;margin-bottom:4px;'>"
            "Access AI Gaze&#8482;</div>"
            "<div style='color:#8888b0;font-size:0.78em;margin-bottom:14px;'>"
            "Enter password to continue</div>",
            unsafe_allow_html=True,
        )
        pwd = st.text_input(
            "Password",
            type="password",
            placeholder="Enter password",
            label_visibility="collapsed",
        )
        if st.button("Sign In  →", type="primary", use_container_width=True):
            if pwd == _APP_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password. Please try again.")
        st.markdown(
            "<div style='color:#44446a;font-size:0.72em;margin-top:20px;'>"
            "Powered by " + _et_wordmark("0.85em", "center") + "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Footer ────────────────────────────────────────────────
    st.markdown("<div style='height:52px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align:center;color:#44446a;font-size:0.72em;"
        "border-top:1px solid rgba(255,255,255,0.05);padding-top:20px;'>"
        "&#169; 2025 Elastic Tree. All Rights Reserved &nbsp;&#183;&nbsp; "
        "AI Gaze&#8482; is a proprietary research product of Elastic Tree</div>",
        unsafe_allow_html=True,
    )


def main():
    # ── Auth gate ─────────────────────────────────────────────
    if not st.session_state.get("authenticated", False):
        _landing_page()
        return

    # ── Header ────────────────────────────────────────────────
    hdr_l, hdr_r = st.columns([3, 1])
    with hdr_l:
        st.markdown(
            "<div style='padding:28px 4px 20px;'>"
            "<div style='display:flex;align-items:baseline;gap:14px;'>"
            "<div style='font-family:Playfair Display,Space Grotesk,Inter,serif;font-size:2.1em;"
            "font-weight:800;color:#f0f0fa;letter-spacing:-1px;line-height:1;'>"
            "AI GAZE"
            "<sup style='font-size:0.3em;vertical-align:super;color:#F5A623;"
            "letter-spacing:0;'>&#8482;</sup></div>"
            "<div style='width:5px;height:5px;border-radius:50%;background:#F5A623;"
            "margin-bottom:3px;flex-shrink:0;'></div>"
            "<div style='font-size:0.72em;color:#55557a;letter-spacing:3px;"
            "font-weight:500;text-transform:uppercase;'>"
            "Predictive Eye Tracking</div>"
            "</div></div>",
            unsafe_allow_html=True,
        )
    with hdr_r:
        st.markdown(
            "<div style='padding:28px 4px 20px;text-align:right;'>"
            + _et_wordmark("1.4em", "right")
            + "<div style='font-size:0.62em;color:#7070a0;letter-spacing:2px;"
            "font-weight:500;margin-top:3px;'>RESEARCH &amp; INNOVATION</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Upload zone + options row ─────────────────────────────
    up_l, up_r = st.columns([3, 2], vertical_alignment="bottom")
    with up_l:
        st.markdown(
            "<p class='section-title' style='margin-bottom:8px;'>Upload Creative</p>",
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "Upload Creative File",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            label_visibility="collapsed",
        )
        st.markdown(
            "<div style='color:#7070a0;font-size:0.75em;margin-top:2px;'>JPG · PNG · WebP · BMP</div>",
            unsafe_allow_html=True,
        )
    with up_r:
        st.markdown(
            "<div class='top-note' style='text-align:left;color:#8e94bc;font-size:0.8em;'>"
            "<strong style='color:#e6e9ff;font-size:0.95em;'>Workflow</strong><br>"
            "Upload once, then review each analysis from the tabs below."
            "<br>Export a polished PDF from the final tab."
            "</div>",
            unsafe_allow_html=True,
        )

    if not uploaded:
        st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
        _, hero_col, _ = st.columns([1, 3, 1])
        with hero_col:
            st.markdown(
                "<div style='text-align:center;padding:52px 28px 44px;"
                "border:1px solid rgba(255,255,255,0.07);border-radius:20px;"
                "background:rgba(255,255,255,0.02);'>"
                "<div style='font-size:2.2em;margin-bottom:14px;opacity:0.75;color:#8ba7d9;'>▣</div>"
                "<div style='font-family:Space Grotesk,Inter,sans-serif;font-size:1.25em;"
                "font-weight:700;color:#f0f0fa;letter-spacing:-0.3px;margin-bottom:8px;'>"
                "Upload an Image to Analyse</div>"
                "<div style='color:#8888a8;font-size:0.86em;margin-bottom:26px;"
                "line-height:1.65;'>Predicts where people look in the first 3&#8211;5 seconds</div>"
                "<div style='display:flex;justify-content:center;flex-wrap:wrap;gap:6px;'>"
                "<span class='feature-chip'>Heat Map</span>"
                "<span class='feature-chip'>Hot Spot</span>"
                "<span class='feature-chip'>Gaze Sequence</span>"
                "<span class='feature-chip'>Clarity Score</span>"
                "<span class='feature-chip'>Top 5 Elements</span>"
                "<span class='feature-chip'>Face Pull</span>"
                "<span class='feature-chip'>Attention Balance</span>"
                "<span class='feature-chip'>Variant Compare</span>"
                "<span class='feature-chip'>PDF Export</span>"
                "</div></div>",
                unsafe_allow_html=True,
            )
        return

    # ── Load & process ────────────────────────────────────────
    pil_img = Image.open(uploaded).convert("RGB")
    img_np  = np.array(pil_img)
    H, W    = img_np.shape[:2]

    if SALIENCY_API_URL:
        engine_label = "Remote Ensemble"
    else:
        engine_label = "DeepGaze IIE" if DEEPGAZE_AVAILABLE else "Itti-Koch"
    with st.spinner(f"Running {engine_label} analysis…"):
        if DEEPGAZE_AVAILABLE:
            _load_deepgaze_model()
        sal_map, components = compute_saliency(img_np)
        engine_label = components.get("engine", engine_label)
        heatmap_img  = generate_heatmap(img_np, sal_map)
        hotspot_img  = generate_hotspot(img_np, sal_map)
        gaze_points  = get_gaze_sequence(sal_map)
        gaze_img     = draw_gaze_sequence(img_np, gaze_points)
        clarity      = compute_clarity_score(sal_map)
        top_elements = detect_top_elements(sal_map, max_items=5)
        top_overlay  = draw_top_elements_overlay(img_np, top_elements) if top_elements else img_np
        face_pull    = compute_face_pull(img_np, sal_map)
        balance      = compute_attention_balance(sal_map)
        balance_grid = draw_attention_balance_overlay(img_np)

    # ── Vertical metrics sidebar + Tabs ──────────────────────
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    peak_pct  = int(round(_calibrated_prob(sal_map, float(sal_map.max()), sal_map.flatten())))
    top_tier  = "HIGH" if peak_pct >= 70 else "MEDIUM" if peak_pct >= 40 else "LOW"
    tier_color = "#ff4c4c" if peak_pct >= 70 else "#F5A623" if peak_pct >= 40 else "#5B8DD9"

    metric_items = [
        ("Peak Attention", f"{peak_pct}%", tier_color),
        ("Clarity Score", f"{clarity['score']:.1f}", "#F5A623"),
        ("Face Pull", f"{face_pull['share']:.1f}%", "#3CBFBF"),
        ("Distraction", f"{balance['distraction']:.1f}%", "#F08A8A"),
        ("Faces", "Yes" if components["face_found"] else "No", "#44BB77" if components["face_found"] else "#8888aa"),
        ("Size", f"{W}×{H}", "#5B8DD9"),
    ]
    mcols = st.columns(len(metric_items))
    for col, (lbl, val, color) in zip(mcols, metric_items):
        with col:
            st.markdown(
                f"<div class='glass-card' style='padding:12px 14px;'>"
                f"<div style='color:#8b90b8;font-size:0.63em;text-transform:uppercase;letter-spacing:1px;font-weight:600;'>{lbl}</div>"
                f"<div style='color:{color};font-size:1.28em;font-weight:800;line-height:1.25;margin-top:2px;'>{val}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    with st.container():
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
            "01 Heat Map",
            "02 Hot Spot",
            "03 Gaze Path",
            "04 Clarity",
            "05 Top Elements",
            "06 Face Pull",
            "07 Balance",
            "08 Compare",
            "09 Report",
        ])

        # ── TAB 1 — HEATMAP ───────────────────────────────────
        with tab1:
            left, right = st.columns([5, 1])

            with left:
                top_l, top_r = st.columns(2)
                with top_l:
                    st.markdown(
                        "<p class='section-title'>Original Image</p>"
                        "<p class='section-sub'>Uploaded reference</p>",
                        unsafe_allow_html=True,
                    )
                    st.image(img_np, width="stretch")
                with top_r:
                    st.markdown(
                        "<p class='section-title'>Heat Map</p>"
                        "<p class='section-sub'>Attention probability overlay</p>",
                        unsafe_allow_html=True,
                    )
                    st.image(heatmap_img, width="stretch")

                st.markdown(
                    "<div class='action-point'><strong>Action Point &nbsp;—</strong> "
                    "The hottest colours (red &rarr; orange &rarr; yellow) should cover the "
                    "elements you want noticed in the first 3&#8211;5 seconds. "
                    "Cold/transparent areas will likely be missed on first glance.</div>",
                    unsafe_allow_html=True,
                )
                _, dl_col, _ = st.columns([2, 1, 2])
                with dl_col:
                    st.download_button(
                        "⬇  Download Heat Map",
                        arr_to_png_bytes(heatmap_img),
                        "aigaze_heatmap.png", "image/png",
                    )

            with right:
                st.markdown(
                    "<p class='section-title' style='margin-top:26px;'>Scale</p>",
                    unsafe_allow_html=True,
                )
                fig = colorbar_figure()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                st.markdown(
                    "<div style='font-size:0.75em;color:#7070a0;margin-top:6px;line-height:1.8;'>"
                    "<span style='color:#cc0000;'>&#9646;</span> High<br>"
                    "<span style='color:#ff6600;'>&#9646;</span> Medium<br>"
                    "<span style='color:#cccc00;'>&#9646;</span> Low<br>"
                    "<span style='color:#44446a;'>&#9646;</span> None</div>",
                    unsafe_allow_html=True,
                )

        # ── TAB 2 — HOTSPOT ───────────────────────────────────
        with tab2:
            leg1, leg2, leg3 = st.columns(3)
            legend_items = [
                ("#1464f5", "rgba(20,100,245,0.08)",   "0 – 40%",   "Low Attention",
                 "Background, margins — likely unnoticed."),
                ("#00d282", "rgba(0,210,130,0.08)",    "40 – 70%", "Medium Attention",
                 "Supporting elements — seen by most viewers."),
                ("#eb2300", "rgba(235,35,0,0.08)",     "70 – 100%", "High Attention",
                 "Hero zones — logo, headline, CTA should sit here."),
            ]
            for col, (border, bg, rng, label, desc) in zip([leg1, leg2, leg3], legend_items):
                with col:
                    st.markdown(
                        f"<div style='border:1px solid {border};background:{bg};"
                        f"border-radius:12px;padding:16px;text-align:center;'>"
                        f"<div style='font-size:1.3em;font-weight:800;color:{border};'>{rng}</div>"
                        f"<div style='font-weight:700;color:#ddd;font-size:0.88em;"
                        f"margin:4px 0 6px;'>{label}</div>"
                        f"<div style='color:#9090c0;font-size:0.76em;line-height:1.5;'>{desc}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

            img_l, img_r = st.columns(2)
            with img_l:
                st.markdown(
                    "<p class='section-title'>Original Image</p>"
                    "<p class='section-sub'>Uploaded reference</p>",
                    unsafe_allow_html=True,
                )
                st.image(img_np, width="stretch")
            with img_r:
                st.markdown(
                    "<p class='section-title'>Hot Spot</p>"
                    "<p class='section-sub'>Three-tier attention zones</p>",
                    unsafe_allow_html=True,
                )
                st.image(hotspot_img, width="stretch")

            st.markdown(
                "<div class='action-point'><strong>Action Point &nbsp;—</strong> "
                "Key elements (logo, product shot, headline, CTA button) should sit inside "
                "a <span style='color:#ff4c4c;font-weight:700;'>red zone</span> (&gt;70%). "
                "If they fall in blue, rethink placement or visual weight.</div>",
                unsafe_allow_html=True,
            )
            _, dl_col, _ = st.columns([2, 1, 2])
            with dl_col:
                st.download_button(
                    "⬇  Download Hot Spot",
                    arr_to_png_bytes(hotspot_img),
                    "aigaze_hotspot.png", "image/png",
                )

        # ── TAB 3 — GAZE SEQUENCE ─────────────────────────────
        with tab3:
            img_col, info_col = st.columns([3, 1])

            with img_col:
                st.markdown(
                    "<p class='section-title'>Gaze Sequence</p>"
                    "<p class='section-sub'>Top 5 predicted fixation points in the first 3&#8211;5 seconds</p>",
                    unsafe_allow_html=True,
                )
                st.image(gaze_img, width="stretch")
                st.markdown(
                    "<div class='action-point'><strong>Action Point &nbsp;—</strong> "
                    "Your brand / key message should appear at positions "
                    "<strong style='color:#fff;'>1 or 2</strong>. "
                    "The eye naturally follows this path — design your visual hierarchy accordingly.</div>",
                    unsafe_allow_html=True,
                )
                _, dl_col, _ = st.columns([2, 1, 2])
                with dl_col:
                    st.download_button(
                        "⬇  Download Gaze Sequence",
                        arr_to_png_bytes(gaze_img),
                        "aigaze_sequence.png", "image/png",
                    )

            with info_col:
                st.markdown(
                    "<p class='section-title' style='margin-top:26px;'>Viewing Order</p>",
                    unsafe_allow_html=True,
                )
                hex_colors = ["#E84040", "#40C860", "#4060E8", "#E8A000", "#9B7BFF"]
                for i, (x, y, prob) in enumerate(gaze_points):
                    c    = hex_colors[i % len(hex_colors)]
                    tier = "HIGH" if prob >= 70 else "MEDIUM" if prob >= 40 else "LOW"
                    tcls = "tier-high" if prob >= 70 else "tier-medium" if prob >= 40 else "tier-low"
                    st.markdown(
                        f"<div class='gaze-card' style='border-left:3px solid {c};'>"
                        f"<div style='display:flex;justify-content:space-between;"
                        f"align-items:center;margin-bottom:6px;'>"
                        f"<span style='font-weight:800;color:{c};font-size:0.9em;'>"
                        f"&#9679; Point {i+1}</span>"
                        f"<span class='{tcls}'>{tier}</span></div>"
                        f"<div style='font-size:2em;font-weight:900;color:#fff;"
                        f"line-height:1;'>{prob:.0f}%</div>"
                        f"<div style='color:#7070a0;font-size:0.72em;margin-top:4px;'>"
                        f"x&thinsp;{x} &nbsp; y&thinsp;{y}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        # ── TAB 4 — CLARITY SCORE ──────────────────────────────
        with tab4:
            left, right = st.columns([3, 2])
            with left:
                st.markdown(
                    "<p class='section-title'>Original + Heat Map</p>"
                    "<p class='section-sub'>Visual context for clarity scoring</p>",
                    unsafe_allow_html=True,
                )
                a, b = st.columns(2)
                with a:
                    st.image(img_np, width="stretch")
                with b:
                    st.image(heatmap_img, width="stretch")
            with right:
                c1, c2 = st.columns(2)
                c3, c4 = st.columns(2)
                c1.metric("Clarity Score", f"{clarity['score']:.1f}/100")
                c2.metric("Peak", f"{clarity['peak']:.1f}%")
                c3.metric("Contrast", f"{clarity['contrast']:.1f}%")
                c4.metric("Focused Area", f"{clarity['focus_ratio']:.1f}%")
                st.markdown(
                    "<div class='action-point'><strong>Interpretation &nbsp;—</strong> "
                    "Higher clarity means attention is concentrated on clear focal regions rather than scattered.</div>",
                    unsafe_allow_html=True,
                )

        # ── TAB 5 — TOP 5 ELEMENTS ────────────────────────────
        with tab5:
            lcol, rcol = st.columns([3, 2])
            with lcol:
                st.markdown(
                    "<p class='section-title'>Original + Top 5 Overlay</p>"
                    "<p class='section-sub'>Highest-ranked attention regions detected from saliency structure</p>",
                    unsafe_allow_html=True,
                )
                o1, o2 = st.columns(2)
                with o1:
                    st.image(img_np, width="stretch")
                with o2:
                    st.image(top_overlay, width="stretch")
            with rcol:
                if top_elements:
                    rows = [{"Rank": e["rank"], "Peak %": e["peak"], "Share %": e["share"], "Score": e["score"]} for e in top_elements]
                    table_rows = "".join([
                        "<tr>"
                        f"<td style='padding:10px 12px;border-bottom:1px solid rgba(255,255,255,0.08);color:#e9ecff;font-weight:600;'>{r['Rank']}</td>"
                        f"<td style='padding:10px 12px;border-bottom:1px solid rgba(255,255,255,0.08);color:#ffb86b;'>{r['Peak %']}</td>"
                        f"<td style='padding:10px 12px;border-bottom:1px solid rgba(255,255,255,0.08);color:#8fd0ff;'>{r['Share %']}</td>"
                        f"<td style='padding:10px 12px;border-bottom:1px solid rgba(255,255,255,0.08);color:#8fffb3;font-weight:700;'>{r['Score']}</td>"
                        "</tr>"
                        for r in rows
                    ])
                    st.markdown(
                        "<div style='border:1px solid rgba(255,255,255,0.10);border-radius:12px;overflow:hidden;"
                        "background:linear-gradient(180deg, rgba(12,17,40,0.96), rgba(8,12,28,0.94));'>"
                        "<table style='width:100%;border-collapse:collapse;font-size:0.83em;'>"
                        "<thead>"
                        "<tr style='background:linear-gradient(180deg, rgba(91,141,217,0.28), rgba(91,141,217,0.12));'>"
                        "<th style='text-align:left;padding:10px 12px;color:#f5f7ff;border-bottom:1px solid rgba(255,255,255,0.16);'>Rank</th>"
                        "<th style='text-align:left;padding:10px 12px;color:#f5f7ff;border-bottom:1px solid rgba(255,255,255,0.16);'>Peak %</th>"
                        "<th style='text-align:left;padding:10px 12px;color:#f5f7ff;border-bottom:1px solid rgba(255,255,255,0.16);'>Share %</th>"
                        "<th style='text-align:left;padding:10px 12px;color:#f5f7ff;border-bottom:1px solid rgba(255,255,255,0.16);'>Score</th>"
                        "</tr>"
                        "</thead>"
                        "<tbody>"
                        + table_rows +
                        "</tbody>"
                        "</table>"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("No strong elements detected.")

        # ── TAB 6 — FACE / PERSON PULL ────────────────────────
        with tab6:
            f1, f2 = st.columns([2, 3])
            with f1:
                st.metric("Detected Faces", face_pull["count"])
                st.metric("Face Attention Share", f"{face_pull['share']:.1f}%")
                st.markdown(
                    "<div class='action-point'><strong>Interpretation &nbsp;—</strong> "
                    "Higher values mean faces are capturing a larger fraction of first-glance attention.</div>",
                    unsafe_allow_html=True,
                )
            with f2:
                face_overlay = img_np.copy()
                for (x, y, w_, h_) in face_pull["faces"]:
                    cv2.rectangle(face_overlay, (x, y), (x + w_, y + h_), (60, 191, 191), 2)
                fo1, fo2 = st.columns(2)
                with fo1:
                    st.image(img_np, width="stretch")
                with fo2:
                    st.image(face_overlay, width="stretch")

        # ── TAB 7 — ATTENTION BALANCE ─────────────────────────
        with tab7:
            lcol, rcol = st.columns([3, 2])
            with lcol:
                st.markdown(
                    "<p class='section-title'>Composition Guide Overlay</p>"
                    "<p class='section-sub'>Rule-of-thirds and center frame for layout balance checks</p>",
                    unsafe_allow_html=True,
                )
                g1, g2 = st.columns(2)
                with g1:
                    st.image(img_np, width="stretch")
                with g2:
                    st.image(balance_grid, width="stretch")
            with rcol:
                b1, b2 = st.columns(2)
                b3, b4 = st.columns(2)
                b1.metric("Center Share", f"{balance['center_share']:.1f}%")
                b2.metric("Edge Share", f"{balance['edge_share']:.1f}%")
                b3.metric("Left / Right", f"{balance['left_share']:.1f}% / {balance['right_share']:.1f}%")
                b4.metric("Top / Bottom", f"{balance['top_share']:.1f}% / {balance['bottom_share']:.1f}%")
                st.metric("Distraction Index", f"{balance['distraction']:.1f}%")
                st.markdown(
                    "<div class='action-point'><strong>Interpretation &nbsp;—</strong> "
                    "Higher distraction means attention is spread across many minor regions instead of core focal areas.</div>",
                    unsafe_allow_html=True,
                )

        # ── TAB 8 — VARIANT COMPARISON ────────────────────────
        with tab8:
            st.markdown(
                "<p class='section-title'>Variant Comparison (A/B)</p>"
                "<p class='section-sub'>Upload a second creative to compare key attention outcomes</p>",
                unsafe_allow_html=True,
            )
            variant_file = st.file_uploader(
                "Upload Variant B",
                type=["jpg", "jpeg", "png", "webp", "bmp"],
                key="variant_b_uploader",
                label_visibility="collapsed",
            )
            if variant_file:
                img_b = np.array(Image.open(variant_file).convert("RGB"))
                sal_b, comp_b = compute_saliency(img_b)
                heat_b = generate_heatmap(img_b, sal_b)
                clarity_b = compute_clarity_score(sal_b)
                face_b = compute_face_pull(img_b, sal_b)
                peak_b = float(np.max(sal_b) * 100.0)

                metrics_a = {"clarity": clarity, "peak": float(peak_pct), "face_pull": face_pull}
                metrics_b = {"clarity": clarity_b, "peak": peak_b, "face_pull": face_b}
                rows, overall = compare_variant_metrics(metrics_a, metrics_b, "A", "B")

                va, vb = st.columns(2)
                with va:
                    st.markdown("<p class='section-title'>Variant A</p>", unsafe_allow_html=True)
                    st.image(heatmap_img, width="stretch")
                with vb:
                    st.markdown("<p class='section-title'>Variant B</p>", unsafe_allow_html=True)
                    st.image(heat_b, width="stretch")

                st.dataframe(rows, width="stretch", hide_index=True)
                st.success(f"Overall winner: Variant {overall}" if overall in ["A", "B"] else "Overall result: Tie")
            else:
                st.info("Upload a second image to run A/B comparison.")

        # ── TAB 9 — PDF EXPORT ────────────────────────────────
        with tab9:
            _, pdf_col, _ = st.columns([1, 2, 1])
            with pdf_col:
                st.markdown(
                    "<p class='section-title'>Report Preview Context</p>"
                    "<p class='section-sub'>Original image included in cover page</p>",
                    unsafe_allow_html=True,
                )
                st.image(img_np, width="stretch")
                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
                st.markdown(
                    "<div class='glass-card' style='text-align:center;padding:36px 32px;'>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "<div style='font-size:2.6em;margin-bottom:12px;'>&#128196;</div>"
                    "<div style='font-size:1.15em;font-weight:800;color:#fff;"
                    "letter-spacing:1px;margin-bottom:8px;'>Full Analysis Report</div>"
                    "<div style='color:#8888a8;font-size:0.83em;line-height:1.8;margin-bottom:20px;'>"
                    "Includes original image, heat map, hot spot &amp; gaze sequence"
                    "</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "<div style='display:flex;flex-direction:column;gap:6px;"
                    "text-align:left;margin-bottom:24px;'>"
                    + "".join([
                        f"<div style='font-size:0.8em;color:#44BB77;'>"
                        f"&#10003; &nbsp; {label}</div>"
                        for label in [
                            "Original image overview",
                            "Heat map with colour scale",
                            "Hot spot tier breakdown",
                            "Gaze sequence &amp; coordinates",
                        ]
                    ])
                    + "</div>",
                    unsafe_allow_html=True,
                )
                if not FPDF_AVAILABLE:
                    st.error("fpdf2 not installed — run `pip install fpdf2`")
                else:
                    if st.button("Generate PDF Report", type="primary", use_container_width=True):
                        with st.spinner("Building report…"):
                            pdf_bytes = export_pdf(
                                img_np, heatmap_img, hotspot_img, gaze_img,
                                None, [], gaze_points,
                            )
                        if pdf_bytes:
                            st.success("Report ready!")
                            st.download_button(
                                "⬇  Download PDF Report",
                                pdf_bytes,
                                "aigaze_report.pdf",
                                "application/pdf",
                            )
                        else:
                            st.error("PDF generation failed.")
                st.markdown("</div>", unsafe_allow_html=True)

    # ── Footer ───────────────────────────────────────────────
    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    ft_l, ft_mid, ft_r = st.columns([2, 1, 1])
    with ft_l:
        st.markdown(
            "<div style='color:#7070a0;font-size:0.75em;padding:10px 0;letter-spacing:0.5px;'>"
            "AI Gaze&#8482; &nbsp;&#183;&nbsp; Predictive Eye Tracking &nbsp;&#183;&nbsp; "
            "Powered by <span style='color:#F5A623;font-weight:700;'>Elastic Tree</span>"
            "</div>",
            unsafe_allow_html=True,
        )
    with ft_r:
        if st.button("Sign Out", use_container_width=False):
            st.session_state.authenticated = False
            st.rerun()
    with ft_mid:
        st.markdown(
            "<div style='text-align:right;padding:8px 0;'>"
            + _et_wordmark("1.0em", "right")
            + "</div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
