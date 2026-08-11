"""High-level analysis pipeline for the HTTP API."""

from __future__ import annotations

import base64
from typing import Any

import cv2
import numpy as np

from .core import (
    arr_to_png_bytes,
    compute_attention_balance,
    compute_clarity_score,
    compute_saliency,
    compute_saliency_high_confidence,
    detect_top_elements,
    draw_attention_balance_overlay,
    draw_gaze_sequence,
    draw_top_elements_overlay,
    estimate_fixation_seconds,
    export_pdf,
    generate_heatmap,
    generate_hotspot,
    get_gaze_sequence,
)


def _b64_png(arr: np.ndarray) -> str:
    return base64.b64encode(arr_to_png_bytes(arr)).decode("ascii")


def decode_upload(data: bytes) -> np.ndarray:
    """Decode image bytes to RGB uint8 array."""
    buf = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image — use JPG, PNG, WebP, or BMP")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def run_analysis(img_rgb: np.ndarray, *, high_confidence: bool = True) -> dict[str, Any]:
    """Run full saliency suite and return JSON-serializable result."""
    if high_confidence:
        sal, meta = compute_saliency_high_confidence(
            img_rgb, target_confidence=85.0, enabled=True, strict_target=True
        )
    else:
        sal, meta = compute_saliency(img_rgb)

    heatmap = generate_heatmap(img_rgb, sal)
    hotspot = generate_hotspot(img_rgb, sal)
    gaze_points = get_gaze_sequence(sal, n=4)
    gaze_img = draw_gaze_sequence(img_rgb, gaze_points)
    fixations = estimate_fixation_seconds(gaze_points, total_seconds=3.0)
    clarity = compute_clarity_score(sal)
    elements = detect_top_elements(sal, max_items=5)
    elements_img = draw_top_elements_overlay(img_rgb, elements)
    balance = compute_attention_balance(sal)
    balance_img = draw_attention_balance_overlay(img_rgb)

    report_meta = {
        **(meta or {}),
        "clarity": clarity,
        "balance": balance,
        "elements": elements,
        "fixations": fixations,
    }

    return {
        "meta": {
            "engine": meta.get("engine"),
            "confidence": meta.get("confidence"),
            "scene_type": meta.get("scene_type"),
            "face_found": bool(meta.get("face_found")),
            "fallback_reason": meta.get("fallback_reason"),
            "clarity": clarity if isinstance(clarity, dict) else {"score": clarity},
            "balance": balance,
            "elements": [
                {
                    "rank": int(e.get("rank", i + 1)),
                    "score": float(e.get("score", 0)),
                    "bbox": e.get("bbox"),
                }
                for i, e in enumerate(elements or [])
            ],
            "gaze": [
                {"x": int(p[0]), "y": int(p[1]), "seconds": float(fixations[i]) if i < len(fixations) else None}
                for i, p in enumerate(gaze_points or [])
            ],
        },
        "images": {
            "original": _b64_png(img_rgb),
            "heatmap": _b64_png(heatmap),
            "hotspot": _b64_png(hotspot),
            "gaze": _b64_png(gaze_img),
            "elements": _b64_png(elements_img),
            "balance": _b64_png(balance_img),
        },
        "_pdf_bundle": {
            "original": img_rgb,
            "heatmap": heatmap,
            "hotspot": hotspot,
            "gaze": gaze_img,
            "gaze_points": gaze_points,
            "report_meta": report_meta,
        },
    }


def build_pdf_from_bundle(bundle: dict[str, Any]) -> bytes | None:
    pdf_bytes = export_pdf(
        bundle["original"],
        bundle["heatmap"],
        bundle["hotspot"],
        bundle["gaze"],
        aoi_img=None,
        aoi_results=None,
        gaze_points=bundle.get("gaze_points"),
        report_meta=bundle.get("report_meta"),
    )
    if pdf_bytes is None:
        return None
    if isinstance(pdf_bytes, (bytes, bytearray)):
        return bytes(pdf_bytes)
    # Some fpdf paths return bytearray via .output()
    return bytes(pdf_bytes)
