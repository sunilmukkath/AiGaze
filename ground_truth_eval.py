import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

import app


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


@dataclass
class EvalRow:
    image_id: str
    nss: float
    auc_judd: float
    cc: float
    confidence: float
    scene_type: str
    engine: str


def _norm_map(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def _load_gray(path: Path, size_hw=None) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise ValueError(f"Failed to load image: {path}")
    if size_hw is not None:
        h, w = size_hw
        if arr.shape[:2] != (h, w):
            arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
    return arr.astype(np.float32)


def _fixation_binary(fix: np.ndarray) -> np.ndarray:
    # Works with {0,255} masks and soft maps.
    return (fix > np.percentile(fix, 85)).astype(np.uint8)


def nss(pred: np.ndarray, fix_bin: np.ndarray) -> float:
    pred = pred.astype(np.float32)
    mu, sd = float(np.mean(pred)), float(np.std(pred))
    if sd < 1e-8:
        return 0.0
    z = (pred - mu) / sd
    pos = z[fix_bin > 0]
    if pos.size == 0:
        return 0.0
    return float(np.mean(pos))


def auc_judd(pred: np.ndarray, fix_bin: np.ndarray) -> float:
    pred = _norm_map(pred)
    fix = fix_bin.astype(bool)
    n_fix = int(np.sum(fix))
    n_pix = pred.size
    if n_fix == 0 or n_fix == n_pix:
        return 0.0

    # Tie-break noise per standard AUC-Judd practice.
    rng = np.random.default_rng(7)
    pred = pred + rng.uniform(0, 1e-7, size=pred.shape).astype(np.float32)

    sal_fix = pred[fix]
    thresholds = np.sort(sal_fix)[::-1]
    tp = np.zeros(thresholds.size + 2, dtype=np.float32)
    fp = np.zeros(thresholds.size + 2, dtype=np.float32)
    tp[0], fp[0] = 0.0, 0.0
    tp[-1], fp[-1] = 1.0, 1.0

    non_fix = ~fix
    n_non = int(np.sum(non_fix))
    flat = pred.reshape(-1)
    fix_flat = fix.reshape(-1)
    non_flat = non_fix.reshape(-1)

    for i, t in enumerate(thresholds, start=1):
        above = flat >= t
        tp[i] = float(np.sum(above & fix_flat)) / max(n_fix, 1)
        fp[i] = float(np.sum(above & non_flat)) / max(n_non, 1)

    order = np.argsort(fp)
    return float(np.trapz(tp[order], fp[order]))


def cc(pred: np.ndarray, gt_density: np.ndarray) -> float:
    p = pred.astype(np.float32).reshape(-1)
    g = gt_density.astype(np.float32).reshape(-1)
    p -= float(np.mean(p))
    g -= float(np.mean(g))
    den = float(np.linalg.norm(p) * np.linalg.norm(g))
    if den < 1e-8:
        return 0.0
    return float(np.dot(p, g) / den)


def _derive_density_from_fixation(fix_gray: np.ndarray, sigma: float = 19.0) -> np.ndarray:
    fix = _fixation_binary(fix_gray).astype(np.float32)
    dens = gaussian_filter(fix, sigma=max(1.0, sigma))
    return _norm_map(dens)


def _resolve_pairs(images_dir: Path, fixation_dir: Path):
    pairs = []
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        stem = img_path.stem
        fix_match = None
        for ext in IMG_EXTS:
            candidate = fixation_dir / f"{stem}{ext}"
            if candidate.exists():
                fix_match = candidate
                break
        if fix_match is not None:
            pairs.append((img_path, fix_match))
    return pairs


def evaluate_dataset(
    images_dir: Path,
    fixation_dir: Path,
    density_dir: Path | None = None,
    max_images: int = 0,
):
    pairs = _resolve_pairs(images_dir, fixation_dir)
    if max_images > 0:
        pairs = pairs[:max_images]
    if not pairs:
        raise ValueError("No image/fixation pairs found. Check folder names and file stems.")

    rows = []
    for img_path, fix_path in pairs:
        img_rgb = cv2.cvtColor(cv2.imread(str(img_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        sal_map, comp = app.compute_saliency_high_confidence(
            img_rgb, target_confidence=85.0, enabled=True, strict_target=True, max_passes=5
        )
        h, w = sal_map.shape
        fix_gray = _load_gray(fix_path, size_hw=(h, w))
        fix_bin = _fixation_binary(fix_gray)

        if density_dir is not None:
            d_path = None
            for ext in IMG_EXTS:
                cand = density_dir / f"{img_path.stem}{ext}"
                if cand.exists():
                    d_path = cand
                    break
            if d_path is not None:
                gt_density = _norm_map(_load_gray(d_path, size_hw=(h, w)))
            else:
                gt_density = _derive_density_from_fixation(fix_gray)
        else:
            gt_density = _derive_density_from_fixation(fix_gray)

        row = EvalRow(
            image_id=img_path.stem,
            nss=nss(sal_map, fix_bin),
            auc_judd=auc_judd(sal_map, fix_bin),
            cc=cc(sal_map, gt_density),
            confidence=float(comp.get("confidence", 0.0)),
            scene_type=str(comp.get("scene_type", "editorial")),
            engine=str(comp.get("engine", "unknown")),
        )
        rows.append(row)

    agg = {
        "count": len(rows),
        "NSS_mean": float(np.mean([r.nss for r in rows])),
        "AUC_Judd_mean": float(np.mean([r.auc_judd for r in rows])),
        "CC_mean": float(np.mean([r.cc for r in rows])),
        "ModelConfidence_mean": float(np.mean([r.confidence for r in rows])),
    }
    return rows, agg


def _print_summary(rows, agg):
    print("image_id,nss,auc_judd,cc,model_confidence,scene_type,engine")
    for r in rows:
        print(
            f"{r.image_id},{r.nss:.4f},{r.auc_judd:.4f},{r.cc:.4f},"
            f"{r.confidence:.1f},{r.scene_type},{r.engine}"
        )
    print("---")
    print(json.dumps(agg, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate AiGaze predictions against ground-truth fixation maps "
            "(MIT/CAT2000 style) using NSS, AUC-Judd, and CC."
        )
    )
    parser.add_argument("--images_dir", required=True, help="Folder of source images")
    parser.add_argument("--fixation_dir", required=True, help="Folder of binary fixation maps")
    parser.add_argument(
        "--density_dir",
        default="",
        help="Optional folder of fixation density maps for CC (if omitted, derived from fixations)",
    )
    parser.add_argument("--max_images", type=int, default=0, help="Limit number of pairs (0 = all)")
    args = parser.parse_args()

    rows, agg = evaluate_dataset(
        images_dir=Path(args.images_dir),
        fixation_dir=Path(args.fixation_dir),
        density_dir=Path(args.density_dir) if args.density_dir else None,
        max_images=int(args.max_images),
    )
    _print_summary(rows, agg)


if __name__ == "__main__":
    main()
