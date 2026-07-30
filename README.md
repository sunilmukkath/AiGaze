# AI Gaze™ — Elastic Tree Product

Self-service **predictive eye tracking** product by **Elastic Tree**.  
Structure mirrors **Ethos Pulse** (`EThos+/ethos-pulse`): ET design system + product brand + domain tool.

## Architecture

```
ai-gaze/
├── package.json              # npm workspaces → apps/web
├── docs/
│   └── DESIGN_SYSTEM.md      # ET tokens (shared with Ethos / elastictree.com)
├── apps/
│   ├── web/                  # Next.js product site (marketing + pricing)
│   │   └── src/
│   │       ├── app/          # /, /pricing, /methodology
│   │       ├── components/et # BrandLogo, ETHeader, ETFooter
│   │       └── lib/          # product copy + studio URL
│   └── studio/               # Streamlit docs (tool currently at repo root)
├── app.py                    # Streamlit Studio (Cloud entrypoint)
├── requirements.txt
└── aigaze_logo.png
```

| Layer | Role |
|-------|------|
| **apps/web** | Product marketing (like Ethos Pulse web) — AI Gaze logo in header, Elastic Tree in footer |
| **app.py (studio)** | Domain tool — attention analysis, PDF export |
| **ET Website `/ai-gaze`** | Corporate site product page linking here / to studio |

## Quick start

### Marketing site (Next.js)

```bash
npm install
npm run dev
# http://localhost:3000
```

### Studio (Streamlit)

```bash
pip install -r requirements.txt
npm run studio
# or: streamlit run app.py
```

Set the studio URL for web CTAs:

```bash
# apps/web/.env.local
NEXT_PUBLIC_AI_GAZE_STUDIO_URL=https://aigaze-production.up.railway.app
```

### Deploy studio on Railway

Config: `railway.toml`, `Procfile`, `nixpacks.toml` (Python 3.11).

```bash
railway link   # or: railway init --name aigaze
railway up
railway domain
```

Start command binds Streamlit to `$PORT`. Point Elastic Tree “Launch Studio” and `NEXT_PUBLIC_AI_GAZE_STUDIO_URL` at the Railway domain.

## Branding (same pattern as Ethos+)

- **Header:** AI Gaze™ product logo
- **Footer:** AI Gaze product + Elastic Tree company mark / legal
- **Tokens:** navy `#0a1f4a`, amber `#e8a820`, teal `#2dd4bf` — see `docs/DESIGN_SYSTEM.md`

## Pricing

Starter ₹2,999 · Growth ₹7,999 · Enterprise custom (from ₹19,999) — configured in `apps/web/src/lib/product.ts`.

## Related

- Ethos Pulse sibling: `Desktop/EThos+/ethos-pulse`
- Corporate site product route: `ET Website/app/ai-gaze`
