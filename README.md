> **Runtime:** Railway serves FastAPI (`api_app.py`) — Streamlit has been removed.

# AI Gaze™ — Elastic Tree Product

Self-service **predictive eye tracking** product by **Elastic Tree**.  
Structure mirrors **Ethos Pulse** (`EThos+/ethos-pulse`): ET design system + product brand + domain tool.

## Architecture

```
ai-gaze/
├── api_app.py                # FastAPI studio + API (Railway entrypoint)
├── engine/                   # DeepGaze saliency + PDF report
├── templates/studio.html     # Studio UI
├── auth_users.py             # SQLite auth / billing / SSO bridge
├── apps/web/                 # Next.js marketing site
└── archive/                  # Legacy Streamlit monolith (retired)
```

| Layer | Role |
|-------|------|
| **api_app.py** | Railway studio — auth, analyze, PDF, PayU fulfill |
| **apps/web** | Product marketing — CTAs to Railway studio URL |
| **ET Website `/ai-gaze`** | Corporate product page linking to studio |

## Quick start

### Marketing site (Next.js)

```bash
npm install
npm run dev
# http://localhost:3000
```

### Studio (FastAPI on Railway)

```bash
pip install -r requirements.txt
uvicorn api_app:app --host 0.0.0.0 --port 8080
# http://127.0.0.1:8080/studio
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

Start command binds FastAPI studio to `$PORT`. Point Elastic Tree “Launch Studio” and `NEXT_PUBLIC_AI_GAZE_STUDIO_URL` at the Railway domain.

## Branding (same pattern as Ethos+)

- **Header:** AI Gaze™ product logo
- **Footer:** AI Gaze product + Elastic Tree company mark / legal
- **Tokens:** navy `#0a1f4a`, amber `#e8a820`, teal `#2dd4bf` — see `docs/DESIGN_SYSTEM.md`

## Pricing

Starter ₹2,999 · Growth ₹7,999 · Enterprise custom (from ₹19,999) — configured in `apps/web/src/lib/product.ts`.

## Related

- Ethos Pulse sibling: `Desktop/EThos+/ethos-pulse`
- Corporate site product route: `ET Website/app/ai-gaze`
