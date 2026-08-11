# AI Gaze — Project library spec

Mirrors **DataWiz** (`LibraryProject` → folders → datasets → runs) and **QualView/TScribe** project workspaces: one place for a client engagement, with assets and analysis history inside.

## Information architecture

```
Project                  e.g. "Brand X Summer Pack"
├── Folder (optional)    e.g. "Round 1", "Shelf", "OOH"
│   └── Creative         uploaded image (+ thumb)
│       └── AnalysisRun  heatmap / hotspot / gaze / metrics / pdf token
└── Creative (unfiled)
    └── AnalysisRun …
```

## Entities

### Project
| Field | Type | Notes |
|-------|------|--------|
| id | string | `prj_…` |
| name | string | Required |
| clientName | string? | Optional label |
| ownerEmail | string | Creator |
| createdAt / updatedAt | ISO | |

### Folder
| Field | Type | Notes |
|-------|------|--------|
| id | string | `fld_…` |
| projectId | string | |
| name | string | |
| parentId | string? | Nested folders later; v1 flat OK |
| createdAt | ISO | |

### Creative
| Field | Type | Notes |
|-------|------|--------|
| id | string | `crv_…` |
| projectId | string | |
| folderId | string? | |
| name | string | Display name |
| fileName | string | Original upload name |
| mimeType | string | |
| width / height | int | |
| storagePath | string | Disk / object key |
| thumbPath | string? | |
| createdAt / updatedAt | ISO | |

### AnalysisRun
| Field | Type | Notes |
|-------|------|--------|
| id | string | `run_…` |
| projectId | string | |
| creativeId | string | |
| engine | string | e.g. `Fallback Saliency`, `DeepGaze IIE+` |
| confidence | number? | |
| clarity | object? | score, focus_ratio, … |
| meta | JSON | gaze points, elements, balance, scene |
| overlayPaths | JSON | heatmap/hotspot/gaze/… file paths or keys |
| createdAt | ISO | |
| label | string? | Optional user label |

## API (FastAPI)

```
GET    /api/projects
POST   /api/projects                    { name, clientName? }
GET    /api/projects/{id}
PATCH  /api/projects/{id}
DELETE /api/projects/{id}

POST   /api/projects/{id}/folders       { name, parentId? }
DELETE /api/folders/{id}

POST   /api/projects/{id}/creatives     multipart file + folderId?
GET    /api/creatives/{id}
DELETE /api/creatives/{id}

POST   /api/creatives/{id}/analyze      → creates AnalysisRun
GET    /api/creatives/{id}/runs
GET    /api/runs/{id}
GET    /api/runs/{id}/report.pdf
GET    /api/runs/{id}/overlays/{kind}   heatmap|hotspot|gaze|…
```

Auth: same session cookie as today. Quota: `consume_analysis` on each new run.

## UI (studio)

### Library rail (left)
- List projects (create / rename / delete)
- Expand → folders + unfiled creatives
- Upload drops into selected project/folder
- Empty state: “Create a project to start” (DataWiz copy pattern)

### Workspace (main)
- Selected creative preview
- **Analyse** → progress → tabs (Heat Map, Hot Spot, Gaze, …)
- **Save run** automatic on success
- Run history for that creative
- **Compare** pick two creatives in project

### Header
- Project name breadcrumb
- Sign out / Upgrade (existing)

## Storage

**v1:** SQLite under `AIGAZE_DATA_DIR` + files in `AIGAZE_DATA_DIR/creatives/…`  
**v2:** Postgres + S3/R2 if multi-instance Railway

## Frontend scaffold

Types live in `apps/web/src/lib/project/types.ts` (DataWiz-shaped).  
Until Next studio ships, FastAPI HTML can call the same API; then `apps/web` adopts the store.

## Migration from today’s studio

1. On first login after P1: auto-create project **“My creatives”**.
2. Ephemeral analyses (no project) remain possible via “Quick analyse” but prompt to save into a project.
3. No change to billing fulfill / SSO bridge contracts.
