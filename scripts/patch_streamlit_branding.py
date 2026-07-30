#!/usr/bin/env python3
"""Replace Streamlit boot splash branding with AI Gaze (run at Docker image build)."""
from __future__ import annotations

import shutil
from pathlib import Path

import streamlit

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(streamlit.__file__).resolve().parent / "static"
INDEX = STATIC / "index.html"
ICON_SRC = ROOT / "aigaze_icon.png"
if not ICON_SRC.is_file():
    ICON_SRC = ROOT / "apps" / "web" / "public" / "favicon.png"

BOOT_CSS = """
<style id="aigaze-boot">
  html, body, #root {
    background: #0a1f4a !important;
  }
  /* Hide Streamlit boot / connection logos before app CSS loads */
  img[alt*="Streamlit" i],
  img[src*="streamlit"],
  [data-testid="stLogo"],
  [data-testid="stAppDeployButton"],
  .stAppDeployButton {
    display: none !important;
  }
</style>
"""


def main() -> None:
    if not INDEX.is_file():
        raise SystemExit(f"Streamlit index not found: {INDEX}")

    html = INDEX.read_text(encoding="utf-8")
    if 'id="aigaze-boot"' not in html:
        if "</head>" not in html:
            raise SystemExit("Unexpected Streamlit index.html (no </head>)")
        html = html.replace("</head>", BOOT_CSS + "\n</head>", 1)

    html = html.replace("<title>Streamlit</title>", "<title>AI Gaze™ | Elastic Tree</title>")
    INDEX.write_text(html, encoding="utf-8")

    if ICON_SRC.is_file():
        for name in ("favicon.png", "favicon.ico"):
            dest = STATIC / name
            try:
                shutil.copyfile(ICON_SRC, dest)
            except OSError:
                pass

    print(f"Patched Streamlit branding in {INDEX}")


if __name__ == "__main__":
    main()
