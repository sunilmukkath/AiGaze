"""AI Gaze analysis engine (Streamlit-free)."""
from __future__ import annotations

import base64
import io
import os
import string
import tempfile
import time
from datetime import datetime, timezone
from functools import lru_cache

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, maximum_filter

try:
    import requests
except ImportError:
    requests = None

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF = None
    FPDF_AVAILABLE = False

try:
    import torch
    import deepgaze_pytorch
    DEEPGAZE_AVAILABLE = True
except ImportError:
    torch = None
    deepgaze_pytorch = None
    DEEPGAZE_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    mp = None
    MEDIAPIPE_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO = None
    YOLO_AVAILABLE = False

_APP_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Fast defaults for Railway CPU — DeepGaze/MediaPipe/YOLO can hang on first load.
SALIENCY_ENABLE_TTA = os.environ.get("AIGAZE_ENABLE_TTA", "").strip() == "1"
USE_DEEPGAZE = os.environ.get("AIGAZE_USE_DEEPGAZE", "").strip() == "1"
ENABLE_FACE_PRIOR = os.environ.get("AIGAZE_ENABLE_FACE_PRIOR", "").strip() == "1"


# ── Brand assets ──
def _find_elastic_tree_source_png():
    for path in (
        os.path.join(_APP_ROOT_DIR, "elastic_tree_logo.png"),
        os.path.join(_APP_ROOT_DIR, "assets", "elastic_tree_logo.png"),
        os.path.join(_APP_ROOT_DIR, "static", "logo.png"),
        os.path.join(_APP_ROOT_DIR, "static", "elastic-tree-logo.png"),
    ):
        if os.path.isfile(path):
            return path
    return None


def _find_aigaze_source_png():
    for path in (
        os.path.join(_APP_ROOT_DIR, "aigaze_logo.png"),
        os.path.join(_APP_ROOT_DIR, "assets", "aigaze_logo.png"),
        os.path.join(_APP_ROOT_DIR, "static", "aigaze-logo.png"),
    ):
        if os.path.isfile(path):
            return path
    return None


def _normalize_aigaze_logo_file() -> str | None:
    """
    Knock out flat white/light backgrounds, crop, resize height to match Elastic Tree logo.
    Cached under `.cache/` (already gitignored).
    """
    src = _find_aigaze_source_png()
    if src is None:
        return None
    cache_dir = os.path.join(_APP_ROOT_DIR, ".cache")
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except OSError:
        return src
    outp = os.path.join(cache_dir, "aigaze_logo_processed.png")
    try:
        src_mtime = os.path.getmtime(src)
        et_early = _find_elastic_tree_source_png()
        dep_mtime = src_mtime if et_early is None else max(src_mtime, os.path.getmtime(et_early))
        if os.path.isfile(outp) and os.path.getmtime(outp) >= dep_mtime:
            return outp

        pil = Image.open(src).convert("RGBA")
        # PIL-backed buffers can be read-only in NumPy; we mutate alpha in-place.
        arr = np.asarray(pil, dtype=np.uint8).copy()
        if arr.ndim < 3 or arr.shape[0] < 2 or arr.shape[1] < 2:
            return src

        r = arr[..., 0].astype(np.float32)
        g_c = arr[..., 1].astype(np.float32)
        b_c = arr[..., 2].astype(np.float32)
        a_in = arr[..., 3].astype(np.float32)

        mx = np.maximum(np.maximum(r, g_c), b_c)
        mn = np.minimum(np.minimum(r, g_c), b_c)
        chroma = mx - mn
        lum = 0.2126 * r + 0.7152 * g_c + 0.0722 * b_c

        matte = np.ones(arr.shape[:2], dtype=bool)
        matte &= chroma <= 22.0
        matte &= mn >= 210.0
        matte &= lum >= 235.0
        matte |= (lum >= 247.5) & (mx >= 237.0) & (chroma <= 14.0)

        a_new = np.where(matte, 0.0, a_in)
        alp = np.clip(np.round(a_new), 0, 255).astype(np.uint8)
        arr[..., 3] = np.minimum(arr[..., 3], alp)

        nz = np.argwhere(arr[..., 3] > 18)
        if nz.size > 0:
            y0, x0 = nz.min(axis=0)
            y1, x1 = nz.max(axis=0)
            pad = 1
            arr = arr[
                max(0, y0 - pad) : min(arr.shape[0], y1 + pad + 1),
                max(0, x0 - pad) : min(arr.shape[1], x1 + pad + 1),
                :,
            ]

        # Keep native cropped resolution for crisp UI scaling (height set in CSS).
        Image.fromarray(arr, mode="RGBA").save(outp, format="PNG")
        return outp
    except Exception:
        try:
            if os.path.isfile(outp):
                os.unlink(outp)
        except OSError:
            pass
        return src


def _aigaze_logo_path():
    """Prefer processed AiGaze PNG so UI/PDF show transparent edges vs Elastic Tree."""
    return _normalize_aigaze_logo_file() or _find_aigaze_source_png()




# ══════════════════════════════════════════════════════════════
# DEEPGAZE SALIENCY ENGINE
# Primary: DeepGaze IIE (deep learning, ~90% accuracy)
# ══════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def _load_deepgaze_model():
    """Load DeepGaze IIE once and cache for the session. Downloads weights on first run."""
    # Streamlit/cloud-safe cache path for Torch model weights.
    torch_home = os.path.join(tempfile.gettempdir(), "aigaze-torch-cache")
    os.makedirs(torch_home, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", torch_home)
    ckpt_dir = os.path.join(torch_home, "hub", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Use MPS on Apple Silicon if available, else CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    def _download_atomic(url, dst_path, timeout=180):
        if requests is None:
            return False, "requests dependency unavailable"
        tmp_path = f"{dst_path}.download"
        try:
            r = requests.get(url, stream=True, timeout=timeout)
            if r.status_code != 200:
                return False, f"HTTP {r.status_code} for {url}"
            with open(tmp_path, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1024 * 512):
                    if not chunk:
                        continue
                    fh.write(chunk)
            if os.path.getsize(tmp_path) < 1024 * 1024:
                return False, f"Downloaded file too small: {tmp_path}"
            os.replace(tmp_path, dst_path)
            return True, ""
        except Exception as e:
            return False, str(e)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass

    def _url_to_filename(url):
        name = url.split("?")[0].rstrip("/").split("/")[-1]
        return name or "checkpoint.pth"

    def _acquire_lock(lock_path, timeout_sec=240.0, poll_sec=0.25, stale_sec=300.0):
        start = time.time()
        while True:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                payload = f"{os.getpid()}|{time.time():.6f}"
                os.write(fd, payload.encode("ascii", errors="ignore"))
                return fd
            except FileExistsError:
                # Recover from stale locks left by crashed workers.
                try:
                    age = time.time() - os.path.getmtime(lock_path)
                    if age > stale_sec:
                        os.unlink(lock_path)
                        continue
                except OSError:
                    # If lock disappeared between checks, retry immediately.
                    continue
                if time.time() - start > timeout_sec:
                    raise TimeoutError(
                        f"Timed out waiting for checkpoint lock: {os.path.basename(lock_path)}. "
                        "Another worker may still be downloading checkpoints."
                    )
                time.sleep(poll_sec)

    def _release_lock(fd, lock_path):
        try:
            os.close(fd)
        except Exception:
            pass
        try:
            if os.path.exists(lock_path):
                os.unlink(lock_path)
        except Exception:
            pass

    def _robust_load_url(url, map_location=None, progress=True, **kwargs):
        del progress, kwargs
        filename = _url_to_filename(url)
        dst_path = os.path.join(ckpt_dir, filename)
        file_lock = f"{dst_path}.lock"
        lock_fd = None
        try:
            lock_fd = _acquire_lock(file_lock)
            for attempt in range(1, 5):
                # Happy path: existing file
                if os.path.exists(dst_path):
                    try:
                        if os.path.getsize(dst_path) >= 1024 * 1024:
                            return torch.load(dst_path, map_location=map_location)
                    except Exception:
                        try:
                            os.unlink(dst_path)
                        except OSError:
                            pass
                ok, err = _download_atomic(url, dst_path)
                if ok:
                    try:
                        return torch.load(dst_path, map_location=map_location)
                    except Exception as e:
                        err = str(e)
                # clear broken file and retry with backoff
                try:
                    if os.path.exists(dst_path):
                        os.unlink(dst_path)
                except OSError:
                    pass
                if attempt < 4:
                    time.sleep(1.1 * attempt)
                    continue
                raise RuntimeError(f"Failed to download/load checkpoint {filename}: {err}")
        finally:
            if lock_fd is not None:
                _release_lock(lock_fd, file_lock)

    def _construct():
        # Monkey-patch both model_zoo and torch.hub URL loading for robust checkpoint fetch.
        orig_mz = torch.utils.model_zoo.load_url
        orig_hub = torch.hub.load_state_dict_from_url
        torch.utils.model_zoo.load_url = _robust_load_url
        torch.hub.load_state_dict_from_url = _robust_load_url
        try:
            model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(device)
            model.eval()
            return model
        finally:
            torch.utils.model_zoo.load_url = orig_mz
            torch.hub.load_state_dict_from_url = orig_hub

    last_err = ""
    for attempt in range(1, 4):
        try:
            model = _construct()
            return model, device, True, ""
        except Exception as e:
            last_err = str(e)
            if attempt < 3:
                time.sleep(1.2 * attempt)
                continue
            break
    return None, None, False, last_err


def _norm(arr):
    mn, mx = arr.min(), arr.max()
    return np.zeros_like(arr) if mx - mn < 1e-8 else (arr - mn) / (mx - mn)


def _remote_ensemble_saliency(img):
    """Optional remote ensemble hook; returns None when unused."""
    del img
    return None


def _map_correlation(a, b):
    """Stable correlation score for two saliency maps."""
    aa = a.astype(np.float32).reshape(-1)
    bb = b.astype(np.float32).reshape(-1)
    aa -= aa.mean()
    bb -= bb.mean()
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb)) + 1e-8
    return float(np.clip(np.dot(aa, bb) / denom, -1.0, 1.0))


def _saliency_confidence_score(sal_map, agreement=None):
    """Confidence proxy (0-100): contrast + peak + concentration + agreement."""
    sal = np.clip(sal_map.astype(np.float32), 0.0, 1.0)
    flat = sal.reshape(-1)
    if flat.size == 0:
        return 0.0

    peak = float(np.max(flat))
    q50 = float(np.percentile(flat, 50))
    q90 = float(np.percentile(flat, 90))
    contrast = float(np.clip((q90 - q50) / max(q90, 1e-6), 0.0, 1.0))
    concentration = float(np.mean(flat >= q90))
    concentration_score = float(np.clip(1.0 - abs(concentration - 0.10) / 0.10, 0.0, 1.0))
    agreement_score = float(np.clip((agreement + 1.0) / 2.0, 0.0, 1.0)) if agreement is not None else 0.5

    score = (0.38 * contrast + 0.26 * peak + 0.18 * concentration_score + 0.18 * agreement_score) * 100.0
    return round(float(np.clip(score, 0.0, 100.0)), 1)


def _scene_complexity(img):
    """Estimate visual clutter to adapt semantic prior strengths."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.uint8)
    e1 = cv2.Canny(gray, 45, 130)
    e2 = cv2.Canny(gray, 80, 210)
    edge_density = float(np.mean(np.maximum(e1, e2) > 0))
    return float(np.clip((edge_density - 0.04) / 0.20, 0.0, 1.0))


def _classify_scene_type(img):
    """
    Lightweight scene classifier to drive model fusion presets.
    Returns one of: hero, product, social_busy, editorial.
    """
    complexity = _scene_complexity(img)
    h, w = img.shape[:2]
    total = max(h * w, 1)

    # Approximate text heaviness: many small high-contrast contours.
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 11)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    text_like = 0
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        if 12 <= cw <= max(12, int(0.18 * w)) and 8 <= ch <= max(8, int(0.08 * h)) and area < 0.004 * total:
            text_like += 1
    text_density = float(np.clip(text_like / 220.0, 0.0, 1.0))

    faces = _detect_faces(img)
    face_count = len(faces)
    face_area = 0.0
    for x, y, fw, fh in faces:
        face_area += float(fw * fh)
    face_ratio = float(np.clip(face_area / total, 0.0, 1.0))

    # Large centered object proxy (good for product cards).
    cx1, cx2 = int(0.25 * w), int(0.75 * w)
    cy1, cy2 = int(0.25 * h), int(0.75 * h)
    center = gray[cy1:cy2, cx1:cx2]
    outer = gray.copy()
    outer[cy1:cy2, cx1:cx2] = int(np.mean(outer))
    center_var = float(np.std(center)) if center.size else 0.0
    outer_var = float(np.std(outer)) if outer.size else 1.0
    center_focus = float(np.clip(center_var / max(outer_var, 1e-6), 0.0, 2.0))

    if complexity >= 0.62 or text_density >= 0.45:
        scene = "social_busy"
    elif center_focus >= 1.08 and complexity <= 0.58:
        scene = "product"
    elif face_count > 0 or face_ratio >= 0.03:
        scene = "hero"
    else:
        scene = "editorial"

    return {
        "scene_type": scene,
        "complexity": round(complexity, 3),
        "text_density": round(text_density, 3),
        "face_ratio": round(face_ratio, 3),
        "center_focus": round(center_focus, 3),
    }


SCENE_FUSION_PRESETS = {
    "hero": {"deepgaze": 1.0},
    "product": {"deepgaze": 1.0},
    "social_busy": {"deepgaze": 1.0},
    "editorial": {"deepgaze": 1.0},
}


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
    if not ENABLE_FACE_PRIOR:
        return []
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
        try:
            cascade_path = None
            cv_data = getattr(cv2, "data", None)
            if cv_data is not None and hasattr(cv_data, "haarcascades"):
                cascade_path = os.path.join(
                    cv_data.haarcascades, "haarcascade_frontalface_default.xml"
                )
            if not cascade_path or not os.path.isfile(cascade_path):
                # Some headless OpenCV builds omit cv2.data; resolve via package root.
                pkg_root = os.path.dirname(getattr(cv2, "__file__", "") or "")
                candidate = os.path.join(
                    pkg_root, "data", "haarcascade_frontalface_default.xml"
                )
                if os.path.isfile(candidate):
                    cascade_path = candidate
            classifier_cls = getattr(cv2, "CascadeClassifier", None)
            if cascade_path and os.path.isfile(cascade_path) and classifier_cls is not None:
                face_cascade = classifier_cls(cascade_path)
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                for fx, fy, fw, fh in face_cascade.detectMultiScale(
                    gray, 1.1, 4, minSize=(20, 20)
                ):
                    faces.append((int(fx), int(fy), int(fw), int(fh)))
        except Exception:
            faces = []
    return faces


@lru_cache(maxsize=1)
def _load_yolo_model():
    """Load YOLO model once for person/object prior maps."""
    # Off by default on Railway — first-request download blocks analyse for minutes.
    if os.environ.get("AIGAZE_ENABLE_YOLO", "").strip() != "1":
        return None
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


def _apply_semantic_priors(img, sal):
    """Adapt face/object prior strengths based on scene complexity."""
    H, W = sal.shape
    complexity = _scene_complexity(img)
    face_weight = float(np.clip(0.12 + 0.08 * (1.0 - complexity), 0.08, 0.24))
    person_weight = float(np.clip(0.08 + 0.08 * (1.0 - complexity), 0.06, 0.18))

    sal, face_found = _face_boost(img, sal, H, W, weight=face_weight)
    sal, person_found = _person_object_boost(img, sal, H, W, person_weight=person_weight)
    return sal, {
        "face_found": face_found,
        "person_found": person_found,
        "complexity": round(complexity, 3),
    }


def _deepgaze_forward(model, device, img_rgb):
    """Run DeepGaze once and return normalized saliency map."""
    H, W = img_rgb.shape[:2]
    img_t = torch.tensor(
        img_rgb.transpose(2, 0, 1)[np.newaxis], dtype=torch.float32
    ).to(device)
    cb_t = torch.tensor(
        _center_bias(H, W)[np.newaxis, np.newaxis]
    ).to(device)
    with torch.no_grad():
        log_den = model(img_t, cb_t)
    sal = log_den.squeeze().cpu().numpy()
    sal = np.exp(sal - sal.max())
    return _norm(sal)


def _sharpen_sal(sal):
    """Balanced contrast preset: keep dark regions clearer, hotspots readable."""
    p2, p98 = np.percentile(sal, 2), np.percentile(sal, 98)
    sal = np.clip((sal - p2) / (p98 - p2 + 1e-8), 0, 1)
    # Less aggressive than before (0.65): preserves low-attention contrast.
    sal = np.power(sal, 0.90)
    return _norm(sal)


def compute_saliency(img, enable_tta=None):
    """
    Returns (sal_map [0,1], meta_dict).
    Primary DeepGaze inference with robust fallback chain.
    """
    H, W = img.shape[:2]
    tta_enabled = SALIENCY_ENABLE_TTA if enable_tta is None else bool(enable_tta)
    scene_meta = _classify_scene_type(img)
    fallback_reason = ""

    # 1) Remote ensemble (if configured) - most reliable in cloud.
    remote = _remote_ensemble_saliency(img)
    if remote is not None:
        sal, meta = remote
        meta = dict(meta or {})
        meta.setdefault("scene_type", scene_meta.get("scene_type", "editorial"))
        return sal, meta

    # 2) Local DeepGaze path (opt-in — default off on Railway to avoid first-load hangs).
    if USE_DEEPGAZE and DEEPGAZE_AVAILABLE:
        model, device, ok, load_err = _load_deepgaze_model()
        if ok and model is not None and device is not None:
            try:
                # Run at full res up to 1024px — more detail than 768
                max_dim = 1024
                scale   = min(max_dim / H, max_dim / W, 1.0)
                rH, rW  = int(H * scale), int(W * scale)
                img_r   = cv2.resize(img, (rW, rH), interpolation=cv2.INTER_AREA)

                sal = _deepgaze_forward(model, device, img_r)

                # Multi-scale blend improves stability on very large creatives.
                small_scale = min(768 / H, 768 / W, 1.0)
                if small_scale < scale:
                    sH, sW = int(H * small_scale), int(W * small_scale)
                    img_s = cv2.resize(img, (sW, sH), interpolation=cv2.INTER_AREA)
                    sal_s = _deepgaze_forward(model, device, img_s)
                    sal_s = cv2.resize(sal_s, (rW, rH), interpolation=cv2.INTER_LINEAR)
                    sal = _norm(0.72 * sal + 0.28 * sal_s)

                # Test-time augmentation reduces left/right directional artifacts.
                if tta_enabled:
                    sal_flip = _deepgaze_forward(model, device, img_r[:, ::-1, :])[:, ::-1]
                    sal = _norm(0.78 * sal + 0.22 * sal_flip)

                if scale < 1.0:
                    sal = cv2.resize(sal, (W, H), interpolation=cv2.INTER_LINEAR)

                # Lighter smoothing preserves spatial sharpness
                sal = gaussian_filter(sal, sigma=max(H, W) / 110)
                sal = _norm(sal)

                # AI priors adapted to scene complexity.
                sal, prior_meta = _apply_semantic_priors(img, sal)

                # Sharpen: percentile clip + gamma to make hot-spots pop
                sal = _sharpen_sal(sal)

                return sal, {
                    "face_found": prior_meta["face_found"] or prior_meta["person_found"],
                    "engine": "DeepGaze IIE+",
                    "confidence": _saliency_confidence_score(sal),
                    "complexity": prior_meta["complexity"],
                    "scene_type": scene_meta.get("scene_type", "editorial"),
                }
            except Exception as exc:
                fallback_reason = f"DeepGaze inference failed: {exc}"
        else:
            fallback_reason = f"DeepGaze load failed: {load_err}" if load_err else "DeepGaze unavailable"
    elif not USE_DEEPGAZE:
        fallback_reason = "DeepGaze disabled (AIGAZE_USE_DEEPGAZE!=1)"
    else:
        fallback_reason = "DeepGaze dependencies unavailable"

    # 3) Local heuristic fallback to keep app operational.
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    intensity_maps = []
    for s1, s2 in [(1, 8), (2, 16), (3, 28)]:
        d = np.abs(
            gaussian_filter(gray.astype(np.float32) / 255, s1) -
            gaussian_filter(gray.astype(np.float32) / 255, s2)
        )
        intensity_maps.append(_norm(d))
    intensity = _norm(sum(intensity_maps) / 3)

    f = img.astype(np.float32) / 255
    R, G, B = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    rg = np.abs(R - G) / (R + G + 1e-8)
    by = np.abs(B - 0.5 * (R + G)) / (B + 0.5 * (R + G) + 1e-8)
    color = _norm(gaussian_filter(_norm(rg + by), 5))

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    sat = _norm(gaussian_filter(hsv[:, :, 1] / 255.0, 4))

    e1 = cv2.Canny(gray, 30, 90) / 255.0
    e2 = cv2.Canny(gray, 60, 160) / 255.0
    e3 = cv2.Canny(gray, 100, 220) / 255.0
    edges = _norm(gaussian_filter(np.maximum(np.maximum(e1, e2), e3), 4))

    sal = _norm(0.28 * intensity + 0.28 * color + 0.24 * sat + 0.20 * edges)
    sal, prior_meta = _apply_semantic_priors(img, sal)
    sal = gaussian_filter(sal, sigma=max(H, W) / 70)
    sal = _sharpen_sal(sal)
    return sal, {
        "face_found": prior_meta["face_found"] or prior_meta["person_found"],
        "engine": "Fallback Saliency",
        "confidence": _saliency_confidence_score(sal),
        "complexity": prior_meta["complexity"],
        "scene_type": scene_meta.get("scene_type", "editorial"),
        "fallback_reason": fallback_reason,
    }


def compute_saliency_high_confidence(img, target_confidence=85.0, enabled=False, strict_target=False, max_passes=5):
    """
    High-confidence mode:
    - runs baseline saliency
    - optionally tries extra inference variants
    - picks the highest-confidence candidate
    """
    base_sal, base_comp = compute_saliency(img, enable_tta=SALIENCY_ENABLE_TTA)
    base_comp = dict(base_comp or {})
    scene_type = base_comp.get("scene_type", "editorial")
    scene_floor = {
        "hero": 85.0,
        "product": 85.0,
        "social_busy": 82.0,
        "editorial": 84.0,
    }.get(scene_type, float(target_confidence))
    dynamic_target = float(target_confidence) if strict_target else max(float(target_confidence), scene_floor)
    base_conf = float(base_comp.get("confidence", _saliency_confidence_score(base_sal)))
    base_comp["confidence"] = round(base_conf, 1)
    base_comp["mode"] = "Standard"
    base_comp["target_confidence"] = float(dynamic_target)

    if not enabled or base_conf >= float(dynamic_target):
        return base_sal, base_comp

    candidates = [(base_sal, base_comp)]
    current_best = base_conf
    passes = 1

    # Candidate 2: toggle TTA behavior.
    try:
        alt_sal, alt_comp = compute_saliency(img, enable_tta=not SALIENCY_ENABLE_TTA)
        alt_comp = dict(alt_comp or {})
        alt_comp["confidence"] = round(float(alt_comp.get("confidence", _saliency_confidence_score(alt_sal))), 1)
        alt_comp["mode"] = "High-Confidence"
        candidates.append((alt_sal, alt_comp))
        passes += 1
        current_best = max(current_best, float(alt_comp["confidence"]))
    except Exception:
        pass

    # Candidate 3: mirror-consensus blend to reduce directional artifacts.
    try:
        flip_sal, _ = compute_saliency(img[:, ::-1, :], enable_tta=False)
        flip_sal = flip_sal[:, ::-1]
        agreement = _map_correlation(base_sal, flip_sal)
        blend_sal = _sharpen_sal(_norm(0.70 * base_sal + 0.30 * flip_sal))
        if scene_type == "social_busy":
            # Slightly stronger smoothing helps noisy clutter scenes.
            blend_sal = _norm(gaussian_filter(blend_sal, sigma=max(img.shape[:2]) / 150))
        blend_comp = dict(base_comp)
        blend_comp["engine"] = f"{base_comp.get('engine', 'Model')} + Mirror Consensus"
        blend_comp["agreement"] = round(float(agreement), 3)
        blend_comp["confidence"] = _saliency_confidence_score(blend_sal, agreement=agreement)
        blend_comp["mode"] = "High-Confidence"
        candidates.append((blend_sal, blend_comp))
        passes += 1
        current_best = max(current_best, float(blend_comp["confidence"]))
    except Exception:
        pass

    # Strict target mode: keep trying additional denoise/consensus variants until target or max passes.
    if strict_target and current_best < float(dynamic_target):
        extra_specs = [
            (0.62, 0.38, 180.0),  # stronger mirror blend + smoother
            (0.78, 0.22, 240.0),  # gentler mirror blend + lighter smoothing
            (0.55, 0.45, 160.0),  # aggressive consensus for noisy creatives
        ]
        for wb, wf, sigma_div in extra_specs:
            if passes >= int(max_passes):
                break
            try:
                flip_sal, _ = compute_saliency(img[:, ::-1, :], enable_tta=True)
                flip_sal = flip_sal[:, ::-1]
                agreement = _map_correlation(base_sal, flip_sal)
                extra_sal = _norm(wb * base_sal + wf * flip_sal)
                extra_sal = _norm(gaussian_filter(extra_sal, sigma=max(img.shape[:2]) / sigma_div))
                extra_sal = _sharpen_sal(extra_sal)
                extra_comp = dict(base_comp)
                extra_comp["engine"] = f"{base_comp.get('engine', 'Model')} + Strict Consensus"
                extra_comp["agreement"] = round(float(agreement), 3)
                extra_comp["confidence"] = _saliency_confidence_score(extra_sal, agreement=agreement)
                extra_comp["mode"] = "Target 85"
                candidates.append((extra_sal, extra_comp))
                passes += 1
                current_best = max(current_best, float(extra_comp["confidence"]))
                if current_best >= float(dynamic_target):
                    break
            except Exception:
                continue

    best_sal, best_comp = max(candidates, key=lambda item: float(item[1].get("confidence", 0.0)))
    best_comp = dict(best_comp)
    best_comp["mode"] = "Target 85" if strict_target else "High-Confidence"
    best_comp["target_confidence"] = float(dynamic_target)
    best_comp["passes"] = int(passes)
    best_comp["target_met"] = float(best_comp.get("confidence", 0.0)) >= float(dynamic_target)
    return best_sal, best_comp


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


def get_gaze_sequence(sal_map, n=4):
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


def estimate_fixation_seconds(gaze_points, total_seconds=3.0):
    """
    Allocate a 3-second viewing window across predicted fixation points.
    Earlier points get slightly more dwell time via rank decay.
    """
    if not gaze_points:
        return []

    probs = np.array([max(float(p[2]), 1.0) for p in gaze_points], dtype=np.float32)
    ranks = np.arange(len(gaze_points), dtype=np.float32)
    rank_decay = np.exp(-0.35 * ranks)
    weights = probs * rank_decay
    weights = weights / (weights.sum() + 1e-8)

    secs = float(total_seconds) * weights
    # Prevent unrealistically tiny dwell times when 4-5 points are present.
    min_sec = 0.22
    secs = np.maximum(secs, min_sec)
    secs = float(total_seconds) * (secs / (secs.sum() + 1e-8))
    return [float(s) for s in secs]


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


def compare_variant_metrics(metrics_a, metrics_b, name_a="A", name_b="B"):
    """Build compact comparison rows and overall winner."""
    keys = [
        ("Clarity", metrics_a["clarity"]["score"], metrics_b["clarity"]["score"]),
        ("Peak Attention", metrics_a["peak"], metrics_b["peak"]),
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
    txt = (txt
        .replace("\u2013", "-").replace("\u2014", "-").replace("\u2015", "-")
        .replace("\u2018", "'").replace("\u2019", "'")
        .replace("\u201c", '"').replace("\u201d", '"')
        .replace("\u00b7", ".").replace("\u2022", "-")
        .replace("\u00a0", " ").replace("\u2026", "...")
        .replace("\u2122", "(TM)").replace("\u00ae", "(R)")
    )
    # Drop any remaining characters outside latin-1 (Helvetica's encoding range)
    return txt.encode("latin-1", "replace").decode("latin-1")


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


def _colorbar_horizontal_figure():
    """Compact horizontal gradient strip for PDF heat-map legend."""
    fig, ax = plt.subplots(figsize=(6.8, 0.55))
    grad = np.linspace(1, 0, 256).reshape(1, 256)
    ax.imshow(grad, aspect="auto", cmap=ATTENTION_CMAP, extent=[0, 256, 0, 1])
    ax.set_xticks([0, 64, 128, 192, 256])
    ax.set_xticklabels(["100%", "75%", "50%", "25%", "0%"], color="white", fontsize=7)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#1a1a2e")
    fig.patch.set_facecolor("#080810")
    ax.set_facecolor("#080810")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.85, bottom=0.15)
    return fig


def _elastic_tree_logo_path():
    """Resolve Elastic Tree raster wordmark (bundled PNG)."""
    return _find_elastic_tree_source_png()


def _build_combined_brand_logo_file() -> str | None:
    """
    Single PNG: Elastic Tree + divider + AiGaze, transparent background.
    Rebuilt when either source logo changes.
    """
    ag = _aigaze_logo_path()
    et = _elastic_tree_logo_path()
    if ag is None or et is None:
        return None
    cache_dir = os.path.join(_APP_ROOT_DIR, ".cache")
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except OSError:
        return None
    # Bump basename when composite layout rules change so stale caches regenerate.
    outp = os.path.join(cache_dir, "combined_brand_logo_v3.png")
    try:
        dep_mtime = max(os.path.getmtime(ag), os.path.getmtime(et))
        if os.path.isfile(outp) and os.path.getmtime(outp) >= dep_mtime:
            return outp

        pil_ag = Image.open(ag).convert("RGBA").copy()
        pil_et = Image.open(et).convert("RGBA").copy()
        h = max(pil_ag.size[1], pil_et.size[1])
        if h < 1:
            return None

        def _scale_to_h(im: Image.Image, target_h: int) -> Image.Image:
            w_i, h_i = im.size
            if h_i == target_h:
                return im
            nw = max(1, int(round(w_i * (target_h / float(h_i)))))
            return im.resize((nw, target_h), Image.Resampling.LANCZOS)

        ag_s = _scale_to_h(pil_ag, h)
        et_h = max(14, int(round(h * 0.76)))
        et_s = _scale_to_h(pil_et, et_h)
        h_canvas = max(ag_s.size[1], et_s.size[1])
        gap = max(8, int(round(h_canvas * 0.07)))
        div_w = max(1, min(3, int(round(h_canvas / 64))))
        # Elastic Tree leftmost, then divider, then AiGaze
        w_tot = et_s.size[0] + gap + div_w + gap + ag_s.size[0]
        canvas = Image.new("RGBA", (w_tot, h_canvas), (0, 0, 0, 0))
        x0 = 0
        canvas.paste(et_s, (x0, (h_canvas - et_s.size[1]) // 2), et_s)
        x0 += et_s.size[0] + gap
        div_strip = Image.new("RGBA", (div_w, h_canvas), (255, 255, 255, 58))
        canvas.paste(div_strip, (x0, 0), div_strip)
        x0 += div_w + gap
        canvas.paste(ag_s, (x0, (h_canvas - ag_s.size[1]) // 2), ag_s)
        canvas.save(outp, format="PNG")
        return outp
    except Exception:
        try:
            if os.path.isfile(outp):
                os.unlink(outp)
        except OSError:
            pass
        return None


def _combined_brand_logo_path() -> str | None:
    return _build_combined_brand_logo_file()


def _pdf_image_mm_size(path):
    """Return (width_mm, height_mm) to preserve aspect ratio within a max box."""
    with Image.open(path) as pil:
        iw, ih = pil.size
    if iw <= 0 or ih <= 0:
        return (1.0, 1.0)
    ar = ih / float(iw)
    return (1.0, ar)


def _pdf_place_image_fit(pdf, path, x_left, slot_w_mm, y_top, max_h_mm):
    """
    Draw image preserving aspect ratio inside [x_left, x_left + slot_w_mm] x max_h_mm.
    Returns (used_width_mm, used_height_mm).
    """
    ar = _pdf_image_mm_size(path)[1]
    w_mm = float(slot_w_mm)
    h_mm = w_mm * ar
    if h_mm > float(max_h_mm):
        h_mm = float(max_h_mm)
        w_mm = h_mm / max(ar, 1e-6)
    cx = x_left + (float(slot_w_mm) - w_mm) / 2.0
    pdf.image(path, x=cx, y=y_top, w=w_mm)
    return (w_mm, h_mm)


def _pdf_brand_header_banner(pdf):
    # Colour rule stripes
    pdf.set_fill_color(245, 166, 35)
    pdf.rect(0, 0, 210, 2.8, "F")
    pdf.set_fill_color(248, 250, 252)
    pdf.rect(0, 2.8, 210, 19.3, "F")

    # Header identity line: AI Gaze(TM)  |  Predictive Eye Tracking  |  Powered by  [ET logo]
    # Band occupies y=6.1 .. 22.1 (height 16mm); text row height = 5mm
    BAND_TOP = 6.1
    BAND_H = 16.0
    etp = _elastic_tree_logo_path()
    logo_h = 3.8
    lw = _pdf_elastic_tree_logo_width_mm(etp, logo_h)
    pdf.set_font("Helvetica", "", 7.5)
    txt = _pdf_safe("AI Gaze(TM)  |  Predictive Eye Tracking  |  Powered by")
    w_txt = pdf.get_string_width(txt)
    sp = pdf.get_string_width(" ")          # one space gap before logo
    total = w_txt + sp + lw
    x0 = (210.0 - total) / 2.0
    row_h = 5.0
    y0 = BAND_TOP + (BAND_H - row_h) / 2.0  # vertically centred in band
    pdf.set_xy(x0, y0)
    pdf.set_text_color(61, 53, 135)
    pdf.cell(w_txt + sp, row_h, txt + " ", ln=0)
    if lw > 0 and etp:
        xi = pdf.get_x()
        yi = pdf.get_y()
        logo_y = yi + (row_h - logo_h) / 2.0   # vertically centre logo with text row
        pdf.image(etp, x=xi, y=logo_y, h=logo_h)
    # reset cursor below the header band so page content starts cleanly
    pdf.set_xy(12, BAND_TOP + BAND_H)


def _pdf_elastic_tree_logo_width_mm(et_path: str | None, logo_h_mm: float) -> float:
    if not et_path or not os.path.isfile(et_path):
        return 0.0
    with Image.open(et_path) as pil:
        iw, ih = pil.size
    if ih <= 0:
        return float(logo_h_mm)
    return logo_h_mm * (iw / float(ih))


def _pdf_draw_powered_by_with_logo(
    pdf,
    x_left: float,
    y_top: float,
    et_path: str | None,
    suffix: str,
    *,
    font_size: int = 9,
    line_h: float = 5.5,
    logo_h_mm: float = 4.5,
    gap_after_logo_mm: float = 1.0,
) -> None:
    """Left-aligned: 'Powered by' + Elastic Tree raster + suffix (e.g. timestamp)."""
    pdf.set_xy(x_left, y_top)
    pdf.set_font("Helvetica", "", font_size)
    pdf.set_text_color(96, 106, 122)
    prefix = "Powered by "
    w_pre = pdf.get_string_width(_pdf_safe(prefix))
    pdf.cell(w_pre + 0.3, line_h, _pdf_safe(prefix), ln=0)
    if et_path and os.path.isfile(et_path):
        xi = pdf.get_x()
        yi = pdf.get_y()
        lw = _pdf_elastic_tree_logo_width_mm(et_path, logo_h_mm)
        oy = yi + max(0.0, (line_h - logo_h_mm) * 0.22)
        pdf.image(et_path, x=xi, y=oy, h=logo_h_mm)
        pdf.set_xy(xi + lw + gap_after_logo_mm, yi)
    pdf.cell(0, line_h, _pdf_safe(suffix), ln=True)


def _pdf_draw_powered_by_centered_with_logo(
    pdf,
    y_top: float,
    et_path: str | None,
    suffix: str,
    *,
    font_size: int = 7,
    line_h: float = 8.0,
    logo_h_mm: float = 3.0,
    gap_after_logo_mm: float = 1.0,
) -> None:
    """Horizontally centered: 'Powered by' + logo + suffix (page line, etc.)."""
    pdf.set_font("Helvetica", "", font_size)
    prefix = "Powered by "
    w_pre = pdf.get_string_width(_pdf_safe(prefix))
    w_suf = pdf.get_string_width(_pdf_safe(suffix))
    lw = (
        _pdf_elastic_tree_logo_width_mm(et_path, logo_h_mm)
        if (et_path and os.path.isfile(et_path))
        else 0.0
    )
    gap = gap_after_logo_mm if lw > 0 else 0.0
    total = w_pre + lw + gap + w_suf
    x0 = max(12.0, (210.0 - total) / 2.0)
    pdf.set_xy(x0, y_top)
    pdf.set_text_color(118, 128, 145)
    pdf.cell(w_pre + 0.3, line_h, _pdf_safe(prefix), ln=0)
    if lw > 0:
        xi = pdf.get_x()
        yi = pdf.get_y()
        oy = yi + max(0.0, (line_h - logo_h_mm) * 0.18)
        pdf.image(et_path, x=xi, y=oy, h=logo_h_mm)
        pdf.set_xy(xi + lw + gap, yi)
    pdf.cell(0, line_h, _pdf_safe(suffix), ln=0)


class ElasticTreePDF(FPDF):
    """FPDF subclass with branded footer."""

    def footer(self):
        self.set_y(-10)
        self.set_font("Helvetica", "", 7)
        self.set_text_color(118, 128, 145)
        self.cell(
            0, 6,
            _pdf_safe("Page " + str(self.page_no()) + "/{nb}"),
            align="C",
        )


def _pdf_summarize_findings(meta, gaze_points):
    """Short narrative bullets derived from quantitative results."""
    if not isinstance(meta, dict):
        return ["Run the analyzer to populate scores; upload a creative to regenerate this section."]
    bullets = []
    peak = float(meta.get("peak_pct") or 0)
    clr = meta.get("clarity") or {}
    clr_s = float(clr.get("score") or 0)
    bal = meta.get("balance") or {}
    dist = float(bal.get("distraction") or 0)

    scene = meta.get("components") or {}
    scene_lab = str(scene.get("scene_type", "creative")).replace("_", " ").title()

    tier_lab = str(meta.get("top_tier", "") or "").strip()
    peak_bits = f"Peak attention tops out at about {peak:.0f}%"
    bullets.append(peak_bits + (f" ({tier_lab})" if tier_lab else "") + f" for this {scene_lab} layout.")
    if clr_s >= 65:
        bullets.append("Clarity is strong: gaze is concentrated in a few coherent focal areas rather than splintered.")
    elif clr_s >= 45:
        bullets.append("Clarity is moderate: consider sharpening contrast or simplifying competing focal regions.")
    else:
        bullets.append("Clarity is on the lower side; attention may feel busy or fragmented on first glance.")

    if gaze_points:
        gx, gy, gp = gaze_points[0]
        tier1 = "high" if gp >= 70 else "medium" if gp >= 40 else "low"
        bullets.append(f"The predicted first fixation is near ({gx}, {gy}) with roughly {gp:.0f}% relative salience ({tier1}).")

    if dist >= 42:
        bullets.append(
            "Distraction is elevated - multiple secondary regions compete; trim visual noise near edges "
            "if hierarchy matters."
        )
    elif dist <= 22:
        bullets.append("Low distraction versus strong regions: hierarchy is comparatively clean for guided viewing.")

    return bullets[:6]


# ══════════════════════════════════════════════════════════════
# PDF EXPORT
# ══════════════════════════════════════════════════════════════

def _pdf_note_box(pdf, key: str, *, score_val: float | None = None) -> None:
    """Draw a light note box (How to read / Good / Watch) below a section heading."""
    n = _ANALYSIS_NOTES.get(key, {})
    lines = []
    how = n.get("how", "")
    if how:
        lines.append(("HOW TO READ", how, (100, 110, 160)))
    # Dynamic clarity rating
    if key == "clarity" and score_val is not None:
        for lo, hi, rtag, rdesc in n.get("scale", []):
            if lo <= score_val <= hi:
                lines.append((f"SCORE  {score_val:.0f}/100  —  {rtag}", rdesc, (60, 140, 100) if lo >= 70 else (180, 120, 40) if lo >= 45 else (180, 60, 60)))
                break
    good = n.get("good", "")
    if good:
        lines.append(("GOOD", good, (50, 160, 90)))
    nw = n.get("needs_work", "")
    if nw:
        lines.append(("WATCH", nw, (190, 130, 40)))
    if not lines:
        return
    x0, w = 12.0, 186.0
    lh = 5.0
    pad = 4.0
    # Measure total height needed
    pdf.set_font("Helvetica", "", 8.5)
    total_h = pad
    for _, txt, _ in lines:
        # Estimate lines needed for multi_cell
        cw = pdf.get_string_width(_pdf_safe(txt))
        nlines = max(1, int(cw / (w - pad * 2 - 18)) + 1)
        total_h += lh * nlines + 3
    total_h += pad
    y0 = pdf.get_y()
    # Content
    pdf.set_xy(x0 + pad, y0 + pad)
    for label, txt, rgb in lines:
        pdf.set_font("Helvetica", "B", 7.5)
        pdf.set_text_color(*rgb)
        pdf.set_x(x0 + pad)
        pdf.cell(0, lh, _pdf_safe(label), ln=True)
        pdf.set_font("Helvetica", "", 8.5)
        pdf.set_text_color(55, 65, 85)
        pdf.set_x(x0 + pad)
        pdf.multi_cell(w - pad * 2, lh, _pdf_safe(txt), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
    pdf.set_y(y0 + total_h + 4)


def export_pdf(
    original,
    heatmap_img,
    hotspot_img,
    gaze_img,
    aoi_img=None,
    aoi_results=None,
    gaze_points=None,
    report_meta=None,
):
    if not FPDF_AVAILABLE:
        return None

    gaze_points = gaze_points or []
    aoi_results = aoi_results or []
    meta = report_meta if isinstance(report_meta, dict) else {}

    pdf = ElasticTreePDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=22)
    pdf.alias_nb_pages()

    # Header band bottom = 22.1mm; all content must start below this.
    CONTENT_TOP = 25.0   # mm from page top — first safe y for content after header

    tmp = []

    def _save(arr):
        f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        Image.fromarray(arr.astype(np.uint8)).save(f.name)
        tmp.append(f.name)
        return f.name

    def clean_page():
        pdf.add_page()
        _pdf_brand_header_banner(pdf)
        pdf.set_xy(12, CONTENT_TOP)

    def heading(title, subtitle=""):
        # Always place heading below the header band; never above CONTENT_TOP
        y0 = max(CONTENT_TOP, pdf.get_y())
        pdf.set_xy(12, y0)
        # Thin rule above heading to visually separate from any prior content
        pdf.set_draw_color(200, 208, 224)
        pdf.set_line_width(0.25)
        pdf.line(12, y0, 198, y0)
        pdf.ln(3)
        pdf.set_x(12)
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(24, 30, 42)
        pdf.cell(0, 7, _pdf_safe(title), ln=True)
        if subtitle:
            pdf.set_x(12)
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(96, 106, 122)
            pdf.cell(0, 5, _pdf_safe(subtitle), ln=True)
        pdf.ln(4)

    def body(txt, line_height=5.8):
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(55, 65, 80)
        pdf.set_x(12)
        pdf.multi_cell(186, line_height, _pdf_safe(txt), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    def bordered_image_row(path, y_start, slot_w_mm, max_h_mm):
        # Ensure image never starts inside header band
        y_start = max(CONTENT_TOP + 2, y_start)
        w_u, h_u = _pdf_place_image_fit(pdf, path, 12, slot_w_mm, y_start + 3, max_h_mm - 6)
        pdf.set_draw_color(210, 218, 232)
        pdf.set_line_width(0.25)
        x_c = 12 + (slot_w_mm - w_u) / 2 - 2
        pdf.rect(max(12, x_c), y_start, min(190.0, w_u + 4), h_u + 6)
        return y_start + h_u + 14

    def dual_images_row(path_l, path_r, y_start, max_h_mm=88):
        y_start = max(CONTENT_TOP + 2, y_start)
        gap = 5.0
        half_w = (186.0 - gap) / 2.0
        sizes = []

        def place(path, x_slot):
            w_u, h_u = _pdf_place_image_fit(pdf, path, x_slot, half_w, y_start + 3, max_h_mm - 6)
            sizes.append((w_u, h_u, x_slot))
            return h_u

        row_h = max(place(path_l, 12.0), place(path_r, 12.0 + half_w + gap))

        pdf.set_draw_color(210, 218, 232)
        pdf.set_line_width(0.25)
        for w_u, h_u, x_slot in sizes:
            x_c = x_slot + (half_w - w_u) / 2 - 2
            pdf.rect(max(12, x_c), y_start, min(half_w + 3.0, w_u + 4), h_u + 6)

        return y_start + row_h + 14

    orig_p = _save(original)
    heat_p = _save(heatmap_img)
    hot_p = _save(hotspot_img)
    gaze_p = _save(gaze_img)
    aoi_p = _save(aoi_img) if aoi_img is not None else None

    top_o = _save(meta["top_overlay"]) if meta.get("top_overlay") is not None else None
    bal_g = _save(meta["balance_grid"]) if meta.get("balance_grid") is not None else None

    cbar_path = None
    try:
        fc = _colorbar_horizontal_figure()
        cbar_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        fc.savefig(cbar_tmp.name, dpi=160)
        plt.close(fc)
        cbar_path = cbar_tmp.name
        tmp.append(cbar_path)
    except Exception:
        cbar_path = None

    aigaze_pdf_logo = _aigaze_logo_path()
    et_pdf_logo = _elastic_tree_logo_path()
    combined_pdf_logo = _combined_brand_logo_path()

    # ── Cover ───────────────────────────────────────────────────
    pdf.add_page()
    _pdf_brand_header_banner(pdf)

    # ── Cover content (below header band at 22.1 mm) ──────────────
    yo = CONTENT_TOP
    if aigaze_pdf_logo:
        _wu, hu = _pdf_place_image_fit(pdf, aigaze_pdf_logo, 12, 186, yo, max_h_mm=18)
        yo = yo + hu + 6

    # Gold rule separator between logo and title
    pdf.set_draw_color(245, 166, 35)
    pdf.set_line_width(0.5)
    pdf.line(12, yo, 198, yo)
    yo += 5

    pdf.set_xy(12, yo)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(61, 53, 135)
    pdf.cell(0, 8, _pdf_safe("AI Gaze — Visual Attention Report"), ln=True)

    pdf.set_x(12)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(96, 106, 122)
    when = datetime.now(timezone.utc).strftime("%d %b %Y  %H:%M UTC")
    pdf.cell(0, 6, _pdf_safe(when), ln=True)
    pdf.ln(4)

    eng = meta.get("engine_label", "DeepGaze IIE")
    Wm = meta.get("W") or original.shape[1]
    Hm = meta.get("H") or original.shape[0]
    body(
        f"This report summarises predicted gaze for the first ~3 seconds using {eng}. "
        f"Canvas size: {int(Wm)} x {int(Hm)} px.\n\n"
        "Intended for packaging, retail, poster, UI, and campaign creative review — "
        "not a substitute for live eye-tracking studies."
    )
    bordered_image_row(orig_p, pdf.get_y(), 186, max_h_mm=138)

    # ── Metrics & narrative ───────────────────────────────────
    clean_page()
    heading("Executive Metrics", "At-a-glance model outputs")
    clr = meta.get("clarity") or {}
    bal = meta.get("balance") or {}
    comp = meta.get("components") or {}
    pk = meta.get("peak_pct", "")
    sc = clr.get("score")
    clarity_disp = (f"{float(sc):.1f} / 100" if isinstance(sc, (int, float)) else "-")

    pdf.set_x(12)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(240, 243, 250)
    pdf.set_text_color(45, 55, 72)
    pdf.cell(42, 7, "Metric", border=1, fill=True)
    pdf.cell(52, 7, "Value", border=1, fill=True)
    pdf.cell(86, 7, "Notes", border=1, fill=True, ln=True)
    pdf.set_font("Helvetica", "", 9)
    rows_pdf = [
        ("Peak attention", f"{pk}%", str(meta.get("top_tier", ""))),
        ("Clarity score", clarity_disp, "Higher = fewer competing focal islands"),
        ("Clarity contrast", f"{clr.get('contrast', 0):.1f}%", "Spread between peak and median salience"),
        ("Focused area", f"{clr.get('focus_ratio', 0):.1f}%", "Share of canvas in top decile of salience"),
        ("Center share", f"{bal.get('center_share', 0):.1f}%", "Attention inside central 50% box"),
        ("Distraction index", f"{bal.get('distraction', 0):.1f}%", "Attention outside top 3 mass regions"),
        ("Scene signal", str(comp.get("scene_type", "-")).replace("_", " "), "Model scene prior / routing"),
    ]
    for a, b, c in rows_pdf:
        pdf.set_x(12)
        pdf.cell(42, 6.5, _pdf_safe(str(a)), border=1)
        pdf.cell(52, 6.5, _pdf_safe(str(b)), border=1)
        pdf.cell(86, 6.5, _pdf_safe(str(c)), border=1, ln=True)

    pdf.ln(6)
    pdf.set_x(12)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(61, 53, 135)
    pdf.cell(0, 6.5, _pdf_safe("Key Insights"), ln=True)
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(55, 65, 80)
    for line in _pdf_summarize_findings(meta, gaze_points):
        pdf.set_x(12)
        pdf.multi_cell(186, 5.8, _pdf_safe(f"-  {line}"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

    # ── Heat map ────────────────────────────────────────────────
    clean_page()
    heading("Heat Map", "Probability overlay of visual fixation")
    _pdf_note_box(pdf, "heatmap")
    y_heat = pdf.get_y()
    y_heat = bordered_image_row(heat_p, y_heat, 186, max_h_mm=122)
    if cbar_path:
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(120, 128, 145)
        pdf.set_x(12)
        pdf.cell(0, 4, _pdf_safe("Attention scale (model-relative)"), ln=True)
        _pdf_place_image_fit(pdf, cbar_path, 12, 186, y_heat, max_h_mm=14)
        y_heat += 16

    # ── Hot spot ────────────────────────────────────────────────
    clean_page()
    heading("Hot Spot", "Three-tier attention zones")
    _pdf_note_box(pdf, "hotspot")
    bordered_image_row(hot_p, pdf.get_y(), 186, max_h_mm=108)

    # ── Gaze sequence ───────────────────────────────────────────
    clean_page()
    heading("Gaze Sequence", "Predicted first viewing order")
    _pdf_note_box(pdf, "gaze")
    fix_secs = estimate_fixation_seconds(gaze_points, total_seconds=3.0)
    seq_lines = []
    for i, (gx, gy, prob) in enumerate(gaze_points):
        tier = "HIGH" if prob >= 70 else "MEDIUM" if prob >= 40 else "LOW"
        sec = fix_secs[i] if i < len(fix_secs) else 0.0
        seq_lines.append(f"Point {i+1}: ({gx}, {gy})  |  {prob:.0f}% ({tier})  |  {sec:.2f}s dwell")
    body("Top fixation points in a 3-second window (pixels, origin top-left):\n" + "\n".join(seq_lines))
    bordered_image_row(gaze_p, pdf.get_y(), 186, max_h_mm=100)

    # ── Clarity deep-dive ───────────────────────────────────────
    clean_page()
    heading("Clarity Deep-Dive", "How concentrated attention is on the creative")
    _pdf_note_box(pdf, "clarity", score_val=float(clr.get("score") or 0))
    body(
        f"Score: {clr.get('score', 0):.1f}/100  |  "
        f"Contrast: {clr.get('contrast', 0):.1f}%  |  "
        f"Focused Area: {clr.get('focus_ratio', 0):.1f}%  |  "
        f"Peak: {clr.get('peak', 0):.1f}%"
    )
    dual_images_row(orig_p, heat_p, pdf.get_y(), max_h_mm=76)

    # ── Top regions ─────────────────────────────────────────────
    if top_o:
        clean_page()
        heading("Top Attention Regions", "Automated hotspots ranked by blended peak + mass")
        _pdf_note_box(pdf, "top_elements")
        te = meta.get("top_elements") or []
        if te:
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_x(12)
            pdf.set_fill_color(240, 243, 250)
            pdf.set_text_color(45, 55, 72)
            pdf.cell(18, 6.5, "Rank", border=1, fill=True)
            pdf.cell(24, 6.5, "Peak %", border=1, fill=True)
            pdf.cell(24, 6.5, "Share %", border=1, fill=True)
            pdf.cell(26, 6.5, "Score", border=1, fill=True)
            pdf.cell(94, 6.5, "Notes", border=1, fill=True, ln=True)
            pdf.set_font("Helvetica", "", 9)
            for e in te[:5]:
                pdf.set_x(12)
                pdf.cell(18, 6, _pdf_safe(str(e.get("rank", ""))), border=1)
                pdf.cell(24, 6, _pdf_safe(str(e.get("peak", ""))), border=1)
                pdf.cell(24, 6, _pdf_safe(str(e.get("share", ""))), border=1)
                pdf.cell(26, 6, _pdf_safe(str(e.get("score", ""))), border=1)
                pdf.cell(94, 6, _pdf_safe("Highest detected mass in saliency blobs"), border=1, ln=True)
            pdf.ln(4)
        body("Original (left) and ranked overlay (right). Bounding boxes approximate dominant attention islands.")
        dual_images_row(orig_p, top_o, pdf.get_y())

    # ── Composition balance ────────────────────────────────────
    if bal_g:
        clean_page()
        heading("Attention Balance & Composition", "Rule-of-thirds and centre frame")
        _pdf_note_box(pdf, "balance")
        body(
            f"Left / Right: {bal.get('left_share', 0):.1f}% / {bal.get('right_share', 0):.1f}%  |  "
            f"Top / Bottom: {bal.get('top_share', 0):.1f}% / {bal.get('bottom_share', 0):.1f}%  |  "
            f"Distraction Index: {bal.get('distraction', 0):.1f}%"
        )
        dual_images_row(orig_p, bal_g, pdf.get_y())

    # ── AOI (optional) ───────────────────────────────────────────
    if aoi_p and aoi_results:
        clean_page()
        heading("Area of Attention", "Manually selected regions and estimated visibility")
        _pdf_note_box(pdf, "aoi")
        body("User-defined areas with estimated visibility percentages:")
        pdf.ln(1)
        pdf.set_x(12)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(240, 243, 250)
        pdf.set_text_color(45, 55, 72)
        pdf.cell(30, 6.5, "Region", border=1, fill=True)
        pdf.cell(26, 6.5, "Seen %", border=1, fill=True)
        pdf.cell(24, 6.5, "Peak %", border=1, fill=True)
        pdf.cell(30, 6.5, "Tier", border=1, fill=True)
        pdf.cell(95, 6.5, "Coordinates (x1,y1,x2,y2)", border=1, fill=True, ln=True)
        pdf.set_font("Helvetica", "", 9)
        for r in aoi_results:
            tier = "HIGH" if r["prob"] >= 70 else "MEDIUM" if r["prob"] >= 40 else "LOW"
            coords = ",".join(map(str, r["box"]))
            pdf.set_x(12)
            pdf.cell(30, 6, f"Region {r['label']}", border=1)
            pdf.cell(26, 6, f"{r['prob']:.1f}%", border=1)
            pdf.cell(24, 6, f"{r['peak']:.1f}%", border=1)
            pdf.cell(30, 6, tier, border=1)
            pdf.cell(95, 6, coords, border=1, ln=True)
        bordered_image_row(aoi_p, pdf.get_y(), 186, max_h_mm=88)

    out = pdf.output(dest="S")
    for f in tmp:
        try:
            os.unlink(f)
        except OSError:
            pass
    # Streamlit download_button requires immutable bytes(), not bytearray (fpdf2 returns bytearray sometimes).
    if isinstance(out, str):
        return bytes(out, "latin-1")
    return bytes(out)


# ══════════════════════════════════════════════════════════════
# ANALYSIS NOTES  (shared by dashboard + PDF)
# ══════════════════════════════════════════════════════════════

_ANALYSIS_NOTES = {
    "heatmap": {
        "how": (
            "Colour moves from deep red (highest predicted attention) through yellow to blue (lowest). "
            "The brightest zone is where most viewers look first within ~3 seconds."
        ),
        "good": "One or two dominant warm islands centred on your hero element — product, face, or headline.",
        "needs_work": (
            "Heat is spread evenly with no clear focal point, or the hottest zone falls on background "
            "decoration rather than your key message."
        ),
    },
    "hotspot": {
        "how": (
            "Red = top ~30% of attention (HIGH). Green = middle band 40–70% (MEDIUM). "
            "Blue = bottom 40% (LOW). Think of it as a ranked priority map for your layout."
        ),
        "good": "Your logo, hero product, or primary CTA sits inside a red zone.",
        "needs_work": (
            "The red zone is empty or covers only decorative elements. "
            "Key brand assets sitting in blue should be repositioned or made more visually dominant."
        ),
    },
    "gaze": {
        "how": (
            "Numbered dots show the predicted order of first fixations within a ~3-second viewing window. "
            "Point 1 is where the eye lands first; later points show where attention travels next."
        ),
        "good": "The sequence flows logically — brand → product → CTA — matching your intended reading journey.",
        "needs_work": (
            "The eye jumps erratically, skips the headline entirely, or fixates on empty space "
            "before reaching the key message."
        ),
    },
    "clarity": {
        "how": (
            "Clarity Score (0–100) measures how concentrated attention is. "
            "Clarity Contrast is the spread between peak and median salience — higher means a sharper focal point. "
            "Focused Area is the share of canvas in the top attention tier."
        ),
        "scale": [
            (70, 100, "Strong", "Attention is concentrated; clear visual hierarchy."),
            (45, 69,  "Moderate", "Some focus, but competing elements dilute the main story."),
            (0,  44,  "Weak", "Attention is scattered; viewers may feel overwhelmed."),
        ],
        "needs_work": (
            "Focused Area below ~15% may mean your hero element is too small relative to the canvas."
        ),
    },
    "top_elements": {
        "how": (
            "The model ranks the strongest attention islands by a blend of peak salience and total mass. "
            "Rank 1 is the most eye-catching region on the creative."
        ),
        "good": "Rank 1 maps directly to your hero element — face, logo, pack shot, or price.",
        "needs_work": (
            "Rank 1 is a background texture, colour block, or border. "
            "Shift visual weight toward the element you want viewers to notice first."
        ),
    },
    "balance": {
        "how": (
            "The thirds grid and centre frame show where attention weight falls spatially. "
            "Left/Right and Top/Bottom splits reveal if the creative is balanced or skewed. "
            "Distraction Index measures attention leaking to many minor peripheral areas."
        ),
        "good": "Weight aligns with your intended hierarchy. Centre share > 40% for a single-hero layout.",
        "needs_work": (
            "Heavy skew to one corner (>65%) without a deliberate off-centre composition. "
            "Distraction Index > 40% means too many edge elements compete with the main message."
        ),
    },
    "aoi": {
        "how": (
            "Each user-drawn box gets a Seen % (probability a viewer's gaze passes through it) "
            "and a Peak % (highest salience inside it). Tier is HIGH / MEDIUM / LOW."
        ),
        "good": "Priority zones — logo, CTA, price — score HIGH or at least MEDIUM.",
        "needs_work": (
            "A critical zone scores LOW. Consider increasing its contrast, size, or proximity "
            "to the natural gaze entry point shown in the Gaze Sequence."
        ),
    },
}


