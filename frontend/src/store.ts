import { create } from 'zustand';
import { api } from './lib/api';
import { errMsg } from './lib/format';
import type {
  Arrangement,
  Health,
  InstrumentalGrid,
  Job,
  JobType,
  Placement,
  ProjectState,
  ReferenceProfile,
  RenderOptions,
  SearchResult,
  SongManifest,
  UniquenessReport,
  VocalAnalysis,
} from './lib/types';

export const STEPS = ['Upload', 'Analysis', 'Similarity', 'Vocal', 'Arrange', 'Mix & Export'] as const;

export type ToastKind = 'info' | 'error' | 'success';
export interface Toast {
  id: number;
  kind: ToastKind;
  text: string;
}

/** Highest step unlocked for a project, per spec §3 prerequisites. */
export function maxStepFor(p: ProjectState | null): number {
  if (!p) return 0;
  if (p.arrangement_ready) return 5;
  if (p.vocal) return 4;
  if (p.instrumental) return 3;
  if (p.reference?.analyzed) return 2;
  if (p.reference) return 1;
  return 0;
}

interface AppState {
  // connection
  backendUp: boolean | null; // null = still checking
  health: Health | null;
  wsConnected: boolean;

  // navigation
  step: number;

  // similarity slider value, persisted in the store so it survives screen
  // navigation / re-mounts (a local useState reverted the user's choice).
  similarity: number;

  // data
  project: ProjectState | null;
  projects: ProjectState[];
  jobs: Record<string, Job>;
  activeJobs: Partial<Record<JobType, string>>;
  profile: ReferenceProfile | null;
  grid: InstrumentalGrid | null;
  vocalAnalysis: VocalAnalysis | null;
  arrangement: Arrangement | null;
  uniqueness: UniquenessReport | null;
  manifest: SongManifest | null;
  toasts: Toast[];

  // actions
  checkHealth: () => Promise<void>;
  setWsConnected: (v: boolean) => void;
  setStep: (n: number) => void;
  setSimilarity: (n: number) => void;
  toast: (text: string, kind?: ToastKind) => void;
  dismissToast: (id: number) => void;

  loadProjects: () => Promise<void>;
  refreshProject: () => Promise<void>;
  openProject: (p: ProjectState) => Promise<void>;
  deleteProject: (pid: string) => Promise<void>;
  loadArtifacts: () => Promise<void>;

  startReference: (result: SearchResult) => Promise<void>;
  startReferenceUpload: (file: File) => Promise<void>;
  retryReference: () => Promise<void>;
  startGenerate: (similarity: number) => Promise<void>;
  uploadVocal: (file: File) => Promise<void>;
  saveArrangement: (placements: Placement[]) => Promise<boolean>;
  startRearrange: (weights?: Record<string, number>) => Promise<void>;
  startRender: (opts: RenderOptions) => Promise<void>;

  upsertJob: (job: Job) => void;
}

let toastSeq = 0;

export const useStore = create<AppState>()((set, get) => {
  /** Fetch helper that swallows 404s (artifact not written yet). */
  async function safe<T>(fn: () => Promise<T>): Promise<T | null> {
    try {
      return await fn();
    } catch {
      return null;
    }
  }

  /** Refresh artifacts after a job of a given type finished. */
  async function onJobDone(job: Job): Promise<void> {
    const pid = job.project_id;
    switch (job.type) {
      case 'reference': {
        await get().refreshProject();
        const profile = await safe(() => api.getReferenceProfile(pid));
        set({ profile });
        break;
      }
      case 'generate': {
        await get().refreshProject();
        const [grid, uniqueness] = await Promise.all([
          safe(() => api.getGrid(pid)),
          safe(() => api.getUniquenessReport(pid)),
        ]);
        set({ grid, uniqueness });
        break;
      }
      case 'vocal': {
        await get().refreshProject();
        const vocalAnalysis = await safe(() => api.getVocalAnalysis(pid));
        set({ vocalAnalysis });
        if (get().project?.arrangement_ready) {
          set({ arrangement: await safe(() => api.getArrangement(pid)) });
        }
        break;
      }
      case 'rearrange': {
        const [arrangement, vocalAnalysis] = await Promise.all([
          safe(() => api.getArrangement(pid)),
          safe(() => api.getVocalAnalysis(pid)),
        ]);
        set({ arrangement, vocalAnalysis });
        await get().refreshProject();
        break;
      }
      case 'render': {
        await get().refreshProject();
        const manifest = await safe(() => api.getManifest(pid));
        set({ manifest });
        break;
      }
    }
  }

  return {
    backendUp: null,
    health: null,
    wsConnected: false,
    step: 0,
    similarity: 70,
    project: null,
    projects: [],
    jobs: {},
    activeJobs: {},
    profile: null,
    grid: null,
    vocalAnalysis: null,
    arrangement: null,
    uniqueness: null,
    manifest: null,
    toasts: [],

    toast(text, kind = 'info') {
      const id = ++toastSeq;
      set((s) => ({ toasts: [...s.toasts, { id, kind, text }] }));
      window.setTimeout(() => get().dismissToast(id), kind === 'error' ? 9000 : 5000);
    },

    dismissToast(id) {
      set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) }));
    },

    async checkHealth() {
      try {
        const health = await api.health();
        const wasDown = get().backendUp === false;
        set({ health, backendUp: true });
        if (wasDown) get().loadProjects();
      } catch {
        set({ backendUp: false });
      }
    },

    setWsConnected(v) {
      set({ wsConnected: v });
    },

    setStep(n) {
      set({ step: n });
    },

    setSimilarity(n) {
      set({ similarity: Math.min(100, Math.max(0, Math.round(n))) });
    },

    async loadProjects() {
      try {
        const r = await api.listProjects();
        set({ projects: r.projects });
      } catch {
        /* list refresh is best-effort */
      }
    },

    async refreshProject() {
      const p = get().project;
      if (!p) return;
      try {
        const np = await api.getProject(p.id);
        set((s) => ({
          project: np,
          projects: s.projects.map((x) => (x.id === np.id ? np : x)),
        }));
      } catch {
        /* keep stale state, surface nothing — health poll handles backend-down */
      }
    },

    async openProject(p) {
      set({
        project: p,
        similarity: p.similarity ?? 70,
        profile: null,
        grid: null,
        vocalAnalysis: null,
        arrangement: null,
        uniqueness: null,
        manifest: null,
      });
      await get().loadArtifacts();
      set({ step: maxStepFor(get().project) });
    },

    async deleteProject(pid) {
      try {
        await api.deleteProject(pid);
        set((s) => {
          // Drop any jobs belonging to the deleted project. Otherwise a stale
          // reference job lingers in `activeJobs`/`jobs` and SearchScreen keeps
          // the results locked ("analyzing your reference…"), so you can't pick a
          // new reference after deleting one.
          const jobs = Object.fromEntries(
            Object.entries(s.jobs).filter(([, j]) => j.project_id !== pid),
          );
          const wasActive = s.project?.id === pid;
          return {
            projects: s.projects.filter((x) => x.id !== pid),
            jobs,
            ...(wasActive
              ? {
                  project: null,
                  step: 0,
                  activeJobs: {},
                  profile: null,
                  grid: null,
                  vocalAnalysis: null,
                  arrangement: null,
                  uniqueness: null,
                  manifest: null,
                }
              : {}),
          };
        });
        get().toast('Project deleted.', 'success');
      } catch (e) {
        get().toast(errMsg(e), 'error');
      }
    },

    async loadArtifacts() {
      const p = get().project;
      if (!p) return;
      const pid = p.id;
      const [profile, grid, uniqueness, vocalAnalysis, arrangement, manifest] = await Promise.all([
        p.reference?.analyzed ? safe(() => api.getReferenceProfile(pid)) : Promise.resolve(null),
        p.instrumental ? safe(() => api.getGrid(pid)) : Promise.resolve(null),
        p.instrumental ? safe(() => api.getUniquenessReport(pid)) : Promise.resolve(null),
        p.vocal ? safe(() => api.getVocalAnalysis(pid)) : Promise.resolve(null),
        p.arrangement_ready ? safe(() => api.getArrangement(pid)) : Promise.resolve(null),
        p.exports?.length ? safe(() => api.getManifest(pid)) : Promise.resolve(null),
      ]);
      set({ profile, grid, uniqueness, vocalAnalysis, arrangement, manifest });
    },

    async startReference(result) {
      try {
        let p = get().project;
        // Start a FRESH project whenever there's no open project OR the open one
        // already has a reference — so picking another song always begins a new
        // song instead of being locked to the first reference.
        if (!p || p.reference) {
          p = await api.createProject(result.title);
          set({
            project: p,
            similarity: 70,
            profile: null,
            grid: null,
            vocalAnalysis: null,
            arrangement: null,
            uniqueness: null,
            manifest: null,
          });
        }
        const { job_id } = await api.setReference(p.id, result.url);
        set((s) => ({
          activeJobs: { ...s.activeJobs, reference: job_id },
          profile: null,
          step: 1,
        }));
        await get().refreshProject();
        void get().loadProjects();
      } catch (e) {
        get().toast(errMsg(e), 'error');
      }
    },

    async startReferenceUpload(file) {
      try {
        let p = get().project;
        // Same fresh-project rule as startReference: uploading begins a new song
        // unless the current project has no reference yet.
        if (!p || p.reference) {
          p = await api.createProject(file.name.replace(/\.[^./\\]+$/, '') || 'Uploaded track');
          set({
            project: p,
            similarity: 70,
            profile: null,
            grid: null,
            vocalAnalysis: null,
            arrangement: null,
            uniqueness: null,
            manifest: null,
          });
        }
        const { job_id } = await api.uploadReference(p.id, file);
        set((s) => ({
          activeJobs: { ...s.activeJobs, reference: job_id },
          profile: null,
          step: 1,
        }));
        await get().refreshProject();
        void get().loadProjects();
      } catch (e) {
        get().toast(errMsg(e), 'error');
      }
    },

    async retryReference() {
      const p = get().project;
      if (!p?.reference) return;
      try {
        const { job_id } = await api.setReference(p.id, p.reference.url);
        set((s) => ({ activeJobs: { ...s.activeJobs, reference: job_id } }));
      } catch (e) {
        get().toast(errMsg(e), 'error');
      }
    },

    async startGenerate(similarity) {
      const p = get().project;
      if (!p) return;
      try {
        const { job_id } = await api.generate(p.id, similarity);
        set((s) => ({
          activeJobs: { ...s.activeJobs, generate: job_id },
          uniqueness: null,
        }));
      } catch (e) {
        get().toast(errMsg(e), 'error');
      }
    },

    async uploadVocal(file) {
      const p = get().project;
      if (!p) return;
      try {
        const { job_id } = await api.uploadVocal(p.id, file);
        set((s) => ({ activeJobs: { ...s.activeJobs, vocal: job_id } }));
      } catch (e) {
        get().toast(errMsg(e), 'error');
      }
    },

    async saveArrangement(placements) {
      const p = get().project;
      if (!p) return false;
      try {
        const arrangement = await api.putArrangement(p.id, placements);
        set({ arrangement });
        return true;
      } catch (e) {
        get().toast(errMsg(e), 'error');
        // re-sync with the server-validated truth
        const arrangement = await safe(() => api.getArrangement(p.id));
        if (arrangement) set({ arrangement });
        return false;
      }
    },

    async startRearrange(weights) {
      const p = get().project;
      if (!p) return;
      try {
        const { job_id } = await api.rearrange(p.id, weights);
        set((s) => ({ activeJobs: { ...s.activeJobs, rearrange: job_id } }));
      } catch (e) {
        get().toast(errMsg(e), 'error');
      }
    },

    async startRender(opts) {
      const p = get().project;
      if (!p) return;
      try {
        const { job_id } = await api.render(p.id, opts);
        set((s) => ({
          activeJobs: { ...s.activeJobs, render: job_id },
          manifest: null,
        }));
      } catch (e) {
        get().toast(errMsg(e), 'error');
      }
    },

    upsertJob(job) {
      const prev = get().jobs[job.id];
      set((s) => ({ jobs: { ...s.jobs, [job.id]: job } }));

      const p = get().project;
      if (!p || job.project_id !== p.id) return;

      // Track jobs we did not start ourselves (e.g. pushed over WS after reconnect).
      if (
        (job.status === 'queued' || job.status === 'running') &&
        get().activeJobs[job.type] !== job.id
      ) {
        set((s) => ({ activeJobs: { ...s.activeJobs, [job.type]: job.id } }));
      }

      if (job.status === 'done' && prev?.status !== 'done') {
        void onJobDone(job);
      }
      if (job.status === 'error' && prev?.status !== 'error') {
        get().toast(job.error || `${job.type} job failed.`, 'error');
      }
    },
  };
});
