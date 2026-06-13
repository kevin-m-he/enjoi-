import { useEffect, useState } from 'react';
import { api } from '../lib/api';
import { abbrev, errMsg, fmtTime } from '../lib/format';
import type { ProjectState, SearchResult } from '../lib/types';
import { maxStepFor, STEPS, useStore } from '../store';
import Card from './Card';
import EmptyState from './EmptyState';
import JobProgressBar from './JobProgressBar';
import { LockIcon } from './WaveArt';

function stageLabel(p: ProjectState): string {
  const max = maxStepFor(p);
  if (p.exports?.length) return 'Exported';
  return `At: ${STEPS[max]}`;
}

export default function SearchScreen() {
  const projects = useStore((s) => s.projects);
  const loadProjects = useStore((s) => s.loadProjects);
  const openProject = useStore((s) => s.openProject);
  const deleteProject = useStore((s) => s.deleteProject);
  const startReference = useStore((s) => s.startReference);
  const backendUp = useStore((s) => s.backendUp);
  const project = useStore((s) => s.project);
  const jobs = useStore((s) => s.jobs);
  const activeJobs = useStore((s) => s.activeJobs);
  const setStep = useStore((s) => s.setStep);

  const [q, setQ] = useState('');
  const [results, setResults] = useState<SearchResult[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selecting, setSelecting] = useState<string | null>(null);

  // --- reference job state (UX fix #1 & #2) ---
  const refJob = activeJobs.reference ? jobs[activeJobs.reference] : undefined;
  const refRunning = refJob?.status === 'queued' || refJob?.status === 'running';
  const refDone = refJob?.status === 'done' || !!project?.reference?.analyzed;
  // The chosen video id, so we can visibly mark the current project's reference.
  const chosenVideoId = project?.reference?.video_id ?? null;

  useEffect(() => {
    if (backendUp) void loadProjects();
  }, [backendUp, loadProjects]);

  // 400 ms debounce per spec §4.1
  useEffect(() => {
    const query = q.trim();
    if (query.length < 2) {
      setResults(null);
      setError(null);
      setLoading(false);
      return;
    }
    setLoading(true);
    const t = window.setTimeout(async () => {
      try {
        const r = await api.search(query, 12);
        setResults(r.results);
        setError(null);
      } catch (e) {
        setError(errMsg(e));
      } finally {
        setLoading(false);
      }
    }, 400);
    return () => window.clearTimeout(t);
  }, [q]);

  const onSelect = async (r: SearchResult) => {
    // Only block while a selection is mid-flight or a reference job is running
    // (prevents double-submit). Picking a track when a project already has a
    // reference starts a FRESH project (handled in the store) — so the user can
    // always search and pick a new song.
    if (selecting || refRunning) return;
    setSelecting(r.video_id);
    try {
      await startReference(r);
    } finally {
      setSelecting(null);
    }
  };

  const locked = (videoId: string): boolean =>
    (selecting !== null && selecting !== videoId) || (refRunning && videoId !== chosenVideoId);

  return (
    <div className="space-y-8">
      <div className="pt-6 text-center">
        <h2 className="font-display text-4xl font-black uppercase tracking-tight text-ink">
          Find your <span className="text-pink">reference vibe</span>
        </h2>
        <p className="mx-auto mt-2 max-w-xl text-sm font-medium text-prussian-900">
          Search YouTube for a track whose feel you want. We analyze its style only — the reference
          audio is <span className="font-extrabold text-ink">never</span> part of your final song.
        </p>
        <p className="mx-auto mt-2 max-w-xl text-sm font-bold text-ink">
          Whether you want to be the next <span className="text-pink">Justin Bieber</span> or stay
          underground like <span className="text-pink">King Steaks</span>.
        </p>
      </div>

      {/* big search bar */}
      <div className="mx-auto max-w-2xl">
        <div className="relative">
          <span className="pointer-events-none absolute left-5 top-1/2 -translate-y-1/2 text-prussian-700">
            <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2.5">
              <circle cx="11" cy="11" r="7" />
              <path d="m20 20-3.5-3.5" strokeLinecap="round" />
            </svg>
          </span>
          <input
            autoFocus
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Search YouTube for a song, artist or vibe…"
            className="w-full rounded-brutal border-4 border-ink bg-foam-50 py-4 pl-14 pr-5 text-base font-semibold text-ink placeholder-prussian-900/50 shadow-brutal outline-none transition focus:bg-foam"
          />
        </div>
        {loading && (
          <p className="mt-2 text-center text-xs font-bold uppercase text-prussian-700">Searching…</p>
        )}
        {error && <p className="mt-2 text-center text-xs font-bold text-pink">{error}</p>}
      </div>

      {/* UX FIX #1 — prominent reference loading bar + "move on" gate */}
      {refJob && (
        <div className="mx-auto max-w-3xl space-y-3">
          <JobProgressBar job={refJob} prominent hint="Downloading audio, extracting BPM, key, structure, energy and instrumentation…" />
          {refDone ? (
            <button
              onClick={() => setStep(1)}
              className="w-full rounded-brutal border-4 border-ink bg-pink px-6 py-4 font-display text-lg font-black uppercase tracking-tight text-white shadow-brutal transition active:translate-x-[6px] active:translate-y-[6px] active:shadow-none"
            >
              Continue ▶ Analysis
            </button>
          ) : (
            <p className="text-center text-sm font-bold uppercase tracking-tight text-prussian-700">
              Hold on — analyzing your reference. You can move on the moment it’s done.
            </p>
          )}
        </div>
      )}

      {/* results grid */}
      {results && results.length === 0 && !loading && (
        <EmptyState title="No results" hint="Try a different search — artist plus song title usually works best." />
      )}
      {results && results.length > 0 && (
        <div className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-4">
          {results.map((r) => {
            const isChosen = chosenVideoId === r.video_id;
            const isLocked = locked(r.video_id);
            return (
              <button
                key={r.video_id}
                onClick={() => void onSelect(r)}
                disabled={isLocked}
                className={`group relative overflow-hidden rounded-brutal border-4 border-ink bg-foam-50 text-left shadow-brutal-sm transition ${
                  isChosen
                    ? 'shadow-brutal ring-0'
                    : isLocked
                      ? 'cursor-not-allowed opacity-50'
                      : 'hover:-translate-x-[2px] hover:-translate-y-[2px] hover:shadow-brutal'
                }`}
              >
                <div className="relative aspect-video w-full overflow-hidden border-b-4 border-ink bg-prussian-900">
                  <img
                    src={r.thumbnail_url}
                    alt=""
                    loading="lazy"
                    className="h-full w-full object-cover transition group-hover:scale-105"
                  />
                  <span className="absolute bottom-2 right-2 rounded-sm border-2 border-ink bg-foam px-1.5 py-0.5 text-[11px] font-extrabold tabular-nums text-ink">
                    {fmtTime(r.duration_sec)}
                  </span>
                  {selecting === r.video_id && (
                    <span className="absolute inset-0 grid place-items-center bg-prussian-900/80 text-xs font-extrabold uppercase text-cyan">
                      Starting analysis…
                    </span>
                  )}
                  {isChosen && (
                    <span className="absolute left-2 top-2 rounded-sm border-2 border-ink bg-pink px-2 py-0.5 text-[11px] font-black uppercase text-white">
                      ✓ Reference
                    </span>
                  )}
                  {isLocked && !isChosen && (
                    <span className="absolute inset-0 grid place-items-center bg-prussian-900/60">
                      <LockIcon className="h-9 w-9" />
                    </span>
                  )}
                </div>
                <div className="p-3">
                  <p className="line-clamp-2 text-sm font-bold leading-snug text-ink">{r.title}</p>
                  <p className="mt-1 truncate text-xs font-medium text-prussian-900">
                    {r.channel} · {abbrev(r.view_count)} views
                  </p>
                </div>
              </button>
            );
          })}
        </div>
      )}

      {chosenVideoId && results && results.length > 0 && (
        <p className="text-center text-xs font-bold uppercase tracking-tight text-prussian-700">
          Picking another track starts a fresh song — your current one stays saved below.
        </p>
      )}

      {/* existing projects */}
      <Card
        title="…or pick up an existing project"
        subtitle="Projects live locally under %APPDATA%\enjoi\projects"
      >
        {projects.length === 0 ? (
          <p className="text-sm font-medium text-prussian-900">
            No projects yet — pick a reference above to start your first one.
          </p>
        ) : (
          <ul className="divide-y-2 divide-ink/15">
            {projects.map((p) => (
              <li key={p.id} className="flex items-center gap-3 py-2.5">
                {p.reference?.thumbnail_url ? (
                  <img
                    src={p.reference.thumbnail_url}
                    alt=""
                    className="h-10 w-16 shrink-0 rounded-sm border-2 border-ink object-cover"
                  />
                ) : (
                  <div className="grid h-10 w-16 shrink-0 place-items-center rounded-sm border-2 border-ink bg-washi-200 text-prussian-700">
                    ♪
                  </div>
                )}
                <button
                  onClick={() => void openProject(p)}
                  className="min-w-0 flex-1 text-left transition hover:opacity-80"
                >
                  <p className="truncate text-sm font-bold text-ink">{p.name}</p>
                  <p className="truncate text-xs font-medium text-prussian-700/80">
                    {p.reference ? p.reference.title : 'No reference yet'} ·{' '}
                    {new Date(p.created_at).toLocaleDateString()}
                  </p>
                </button>
                <span className="shrink-0 rounded-sm border-2 border-ink bg-pink px-2 py-0.5 text-[11px] font-extrabold uppercase text-white">
                  {stageLabel(p)}
                </span>
                <button
                  onClick={() => {
                    if (window.confirm(`Delete project "${p.name}"? This cannot be undone.`)) {
                      void deleteProject(p.id);
                    }
                  }}
                  title="Delete project"
                  className="shrink-0 rounded-sm border-2 border-transparent px-2 py-1 text-prussian-700/60 transition hover:border-ink hover:bg-pink hover:text-white"
                >
                  <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2.5">
                    <path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </button>
              </li>
            ))}
          </ul>
        )}
      </Card>

      <p className="pb-4 text-center text-xs font-medium text-prussian-700/70">
        The reference is used for analysis only — its audio never appears in your output, so the
        finished song is 100% yours.
      </p>
    </div>
  );
}
