/**
 * AI Gaze project library types — DataWiz / TScribe-shaped.
 * Spec: docs/project/PROJECT_LIBRARY.md
 */

export type WorkspaceMode = "library" | "analyse" | "compare" | "report";

export interface LibraryFolder {
  id: string;
  name: string;
  parentId: string | null;
  createdAt: string;
}

export interface LibraryProject {
  id: string;
  name: string;
  clientName?: string | null;
  createdAt: string;
  updatedAt: string;
  folders: LibraryFolder[];
}

export interface CreativeRef {
  id: string;
  projectId: string;
  folderId: string | null;
  name: string;
  fileName: string;
  mimeType: string;
  width: number;
  height: number;
  createdAt: string;
  updatedAt: string;
  /** Latest run id if any */
  latestRunId?: string | null;
}

export interface ClarityMetrics {
  score: number;
  focus_ratio?: number;
  contrast?: number;
  peak?: number;
}

export interface GazePoint {
  x: number;
  y: number;
  seconds?: number | null;
}

export interface TopElement {
  rank: number;
  score: number;
  bbox?: number[] | null;
}

export interface AnalysisRunMeta {
  engine?: string;
  confidence?: number;
  scene_type?: string;
  face_found?: boolean;
  fallback_reason?: string;
  clarity?: ClarityMetrics;
  balance?: Record<string, unknown>;
  elements?: TopElement[];
  gaze?: GazePoint[];
}

export type OverlayKind =
  | "original"
  | "heatmap"
  | "hotspot"
  | "gaze"
  | "elements"
  | "balance";

export interface AnalysisRun {
  id: string;
  projectId: string;
  creativeId: string;
  createdAt: string;
  label?: string | null;
  engine: string;
  confidence?: number | null;
  meta: AnalysisRunMeta;
  /** Relative API paths or storage keys for overlays */
  overlays: Partial<Record<OverlayKind, string>>;
}

export interface ProjectLibraryState {
  projects: LibraryProject[];
  creatives: CreativeRef[];
  analysisRuns: AnalysisRun[];
}

export interface LibrarySelection {
  projectId: string | null;
  folderId: string | null;
  creativeId: string | null;
  runId: string | null;
}

export function newId(prefix: string): string {
  return `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`;
}
