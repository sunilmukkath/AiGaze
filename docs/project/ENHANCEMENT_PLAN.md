# AI Gaze — Enhancement plan

Status: draft · Owner: Elastic Tree · Updated: 2026-08-11

## Goals

1. Make the studio **reliable and fast** (analyse never hangs).
2. Give teams a **project library** like TScribe / DataWiz (projects → folders → creatives → analysis runs).
3. Match sibling ET studios for **auth, billing, and UI chrome**.
4. Restore **DeepGaze-quality** inference behind a safe, opt-in path once stability is proven.

## Current baseline (post–Streamlit migration)

| Area | Today |
|------|--------|
| Runtime | FastAPI + HTML studio on Railway |
| Auth | Email + ET SSO bridge |
| Analysis | Fast heuristic saliency (DeepGaze opt-in via `AIGAZE_USE_DEEPGAZE=1`) |
| Storage | SQLite users only — **no project library** |
| Marketing | `apps/web` Next.js (CTAs → Railway studio) |
| Missing vs old Streamlit | A/B compare polish, guided tour, richer landing inside studio |

---

## Phase P0 — Stabilize (1–2 weeks)

**Outcome:** Analyse always finishes; SSO always lands in studio.

| # | Item | Notes |
|---|------|--------|
| P0.1 | Keep fast default engine | Heuristic path default; document DeepGaze env flag |
| P0.2 | Hard server timeout | Cap analyse at ~45–60s; return clear error |
| P0.3 | Progress UX | “Preparing…” / “Running saliency…” / “Building PDF…” |
| P0.4 | Prefetch / cache models | Only when DeepGaze enabled; never block request thread |
| P0.5 | Error surfacing | Show API `detail` in studio; log `analyse start/done` |
| P0.6 | Health + smoke | `/health`, signed-in analyse of 1 fixture in CI |

**Exit:** 10 consecutive uploads complete &lt; 10s on Railway (fast engine).

---

## Phase P1 — Project library like TScribe / DataWiz (2–3 weeks)

**Outcome:** Users organise work in **Projects → Folders → Creatives → Runs**.

See [PROJECT_LIBRARY.md](./PROJECT_LIBRARY.md).

| # | Item | Notes |
|---|------|--------|
| P1.1 | Data model | `projects`, `folders`, `creatives`, `analysis_runs` in SQLite (then Postgres if needed) |
| P1.2 | API | CRUD + upload creative + attach run JSON/overlays |
| P1.3 | Studio library rail | Left rail: projects / folders / assets (DataWiz LibraryHub pattern) |
| P1.4 | Studio workspace | Select creative → analyse → save run into project |
| P1.5 | Compare within project | A/B two creatives from same project |
| P1.6 | PDF from saved run | Re-download without re-running |

**Exit:** Create project → upload 3 packs → run heatmaps → reopen runs next session.

---

## Phase P2 — Product depth (3–4 weeks)

| # | Item | Notes |
|---|------|--------|
| P2.1 | Re-enable DeepGaze safely | Background worker / queue; first request never blocks |
| P2.2 | AOI drawing | Click-drag boxes → % seen (drawable canvas or fabric.js) |
| P2.3 | Clarity + balance + top elements | Already in engine — expose clearly in UI tabs |
| P2.4 | Brand kits | Per-project logo/colours for PDF |
| P2.5 | Share links | Read-only share of a run (token URL) |
| P2.6 | Next.js studio shell | Move UI from `templates/studio.html` → `apps/web/studio` calling API |

**Exit:** DeepGaze available without hangs; AOI + share working.

---

## Phase P3 — Platform alignment (ongoing)

| # | Item | Notes |
|---|------|--------|
| P3.1 | Same SSO copy / buttons as QualView & DataWiz | |
| P3.2 | PayU packs tied to project quotas | Analyses consumed per project or per seat |
| P3.3 | Team seats | Invite @elastictree.com + client emails to a project |
| P3.4 | Audit log | Who ran what, when |
| P3.5 | Eval harness | MIT/CAT2000 or internal GT set → NSS/AUC dashboard |

---

## Suggested build order (next 30 days)

```
Week 1  P0 stabilize + timeout + progress UX
Week 2  P1.1–P1.3 SQLite projects + library rail API/UI
Week 3  P1.4–P1.6 save runs + compare + PDF from run
Week 4  P2.6 start Next studio shell OR P2.1 DeepGaze worker
```

## Non-goals (for now)

- Reintroducing Streamlit
- Face Pull as a user-facing analysis tab
- Interactive AOI before project library exists

## Success metrics

| Metric | Target |
|--------|--------|
| Analyse success rate | ≥ 99% |
| p50 analyse latency (fast engine) | &lt; 3s |
| p95 analyse latency (fast engine) | &lt; 8s |
| Users with ≥1 project after login | ≥ 70% (after P1 ships) |
| PDF download from saved run | Works without re-inference |
