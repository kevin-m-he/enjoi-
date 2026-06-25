// Typed client for every endpoint in docs/API_CONTRACT.md.
import type {
  Arrangement,
  Health,
  InstrumentalGrid,
  Job,
  Placement,
  ProjectState,
  ReferenceProfile,
  RenderOptions,
  SearchResult,
  SongManifest,
  UniquenessReport,
  VocalAnalysis,
} from './types';

// Backend origin. Defaults to the local desktop backend; the public web build
// overrides it at build time with VITE_API_BASE (e.g. the hosted generator URL):
//   VITE_API_BASE=https://api.enjoi.dev  npm run build
// The WS URL is derived so http→ws / https→wss can never drift apart.
export const API_BASE = (
  import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8723'
).replace(/\/+$/, '');
export const WS_URL = `${API_BASE.replace(/^http/, 'ws')}/ws`;

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

// Per-client owner id, persisted in localStorage. Sent on every API request so
// the backend only ever lists THIS client's projects/reference tracks (privacy:
// a new user must never see another user's uploads). Not a security boundary —
// just per-user scoping of an otherwise shared store.
const OWNER_KEY = 'enjoi_owner_id';
function ownerId(): string {
  try {
    let id = localStorage.getItem(OWNER_KEY);
    if (!id) {
      id =
        globalThis.crypto?.randomUUID?.() ??
        `o_${Math.random().toString(36).slice(2)}${Date.now().toString(36)}`;
      localStorage.setItem(OWNER_KEY, id);
    }
    return id;
  } catch {
    return '';
  }
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  headers.set('X-Enjoi-Owner', ownerId());
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, { ...init, headers });
  } catch {
    throw new ApiError(0, 'Backend not reachable');
  }
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const body: unknown = await res.json();
      if (body && typeof body === 'object' && 'detail' in body) {
        const detail = (body as { detail: unknown }).detail;
        msg = typeof detail === 'string' ? detail : JSON.stringify(detail);
      }
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, msg);
  }
  return (await res.json()) as T;
}

function jsonPost(body: unknown, method: 'POST' | 'PUT' = 'POST'): RequestInit {
  return {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  };
}

/** URL for static project media served by the backend (audio, thumbnails, JSON artifacts). */
export function mediaUrl(pid: string, relpath: string): string {
  const safe = relpath
    .split('/')
    .map((p) => encodeURIComponent(p))
    .join('/');
  return `${API_BASE}/media/${encodeURIComponent(pid)}/${safe}`;
}

export const api = {
  // ---- system ----
  health: () => req<Health>('/api/health'),

  // ---- search ----
  search: (q: string, limit = 12) =>
    req<{ results: SearchResult[] }>(`/api/search?q=${encodeURIComponent(q)}&limit=${limit}`),

  // ---- projects CRUD ----
  listProjects: () => req<{ projects: ProjectState[] }>('/api/projects'),
  createProject: (name?: string) =>
    req<ProjectState>('/api/projects', jsonPost(name ? { name } : {})),
  getProject: (pid: string) => req<ProjectState>(`/api/projects/${encodeURIComponent(pid)}`),
  deleteProject: (pid: string) =>
    req<{ ok: boolean }>(`/api/projects/${encodeURIComponent(pid)}`, { method: 'DELETE' }),

  // ---- pipeline steps (all return {job_id}) ----
  setReference: (pid: string, url: string) =>
    req<{ job_id: string }>(`/api/projects/${encodeURIComponent(pid)}/reference`, jsonPost({ url })),
  uploadReference: (pid: string, file: File) => {
    const fd = new FormData();
    fd.append('file', file, file.name);
    return req<{ job_id: string }>(
      `/api/projects/${encodeURIComponent(pid)}/reference/upload`,
      { method: 'POST', body: fd }
    );
  },
  generate: (pid: string, similarity: number) =>
    req<{ job_id: string }>(
      `/api/projects/${encodeURIComponent(pid)}/generate`,
      jsonPost({ similarity })
    ),
  uploadVocal: (pid: string, file: File) => {
    const fd = new FormData();
    fd.append('file', file, file.name);
    return req<{ job_id: string }>(`/api/projects/${encodeURIComponent(pid)}/vocal`, {
      method: 'POST',
      body: fd,
    });
  },
  rearrange: (pid: string, weights?: Record<string, number>) =>
    req<{ job_id: string }>(
      `/api/projects/${encodeURIComponent(pid)}/rearrange`,
      jsonPost(weights ? { weights } : {})
    ),
  render: (pid: string, opts: RenderOptions) =>
    req<{ job_id: string }>(`/api/projects/${encodeURIComponent(pid)}/render`, jsonPost(opts)),

  // ---- arrangement ----
  getArrangement: (pid: string) =>
    req<Arrangement>(`/api/projects/${encodeURIComponent(pid)}/arrangement`),
  putArrangement: (pid: string, placements: Placement[]) =>
    req<Arrangement>(
      `/api/projects/${encodeURIComponent(pid)}/arrangement`,
      jsonPost({ placements }, 'PUT')
    ),

  // ---- jobs / similarity ----
  getJob: (jobId: string) => req<Job>(`/api/jobs/${encodeURIComponent(jobId)}`),
  similarityPreview: (pid: string, value: number) =>
    req<{ summary: string }>(
      `/api/similarity/preview?pid=${encodeURIComponent(pid)}&value=${Math.round(value)}`
    ),

  // ---- JSON artifacts (served by the /media static route per storage layout) ----
  getReferenceProfile: (pid: string) =>
    req<ReferenceProfile>(`/media/${encodeURIComponent(pid)}/reference_profile.json`),
  getGrid: (pid: string) =>
    req<InstrumentalGrid>(`/media/${encodeURIComponent(pid)}/instrumental_grid.json`),
  getVocalAnalysis: (pid: string) =>
    req<VocalAnalysis>(`/media/${encodeURIComponent(pid)}/vocal_analysis.json`),
  getUniquenessReport: (pid: string) =>
    req<UniquenessReport>(`/media/${encodeURIComponent(pid)}/uniqueness_report.json`),
  getManifest: (pid: string) =>
    req<SongManifest>(`/media/${encodeURIComponent(pid)}/exports/song_manifest.json`),
};
