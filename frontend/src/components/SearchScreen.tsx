import { useEffect, useState } from 'react';
import { api } from '../lib/api';
import { abbrev, errMsg, fmtTime } from '../lib/format';
import type { ProjectState, SearchResult } from '../lib/types';
import { maxStepFor, STEPS, useStore } from '../store';
import Card from './Card';
import EmptyState from './EmptyState';

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

  const [q, setQ] = useState('');
  const [results, setResults] = useState<SearchResult[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selecting, setSelecting] = useState<string | null>(null);

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
    if (selecting) return;
    setSelecting(r.video_id);
    try {
      await startReference(r);
    } finally {
      setSelecting(null);
    }
  };

  return (
    <div className="space-y-8">
      <div className="pt-6 text-center">
        <h2 className="text-3xl font-extrabold tracking-tight">
          Find your <span className="text-grad">reference vibe</span>
        </h2>
        <p className="mx-auto mt-2 max-w-xl text-sm text-zinc-400">
          Search YouTube for a track whose feel you want. We analyze its style only — the
          reference audio is <span className="font-semibold text-zinc-300">never</span> part of
          your final song.
        </p>
      </div>

      {/* big search bar */}
      <div className="mx-auto max-w-2xl">
        <div className="relative">
          <span className="pointer-events-none absolute left-5 top-1/2 -translate-y-1/2 text-zinc-500">
            <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="11" cy="11" r="7" />
              <path d="m20 20-3.5-3.5" strokeLinecap="round" />
            </svg>
          </span>
          <input
            autoFocus
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Search YouTube for a song, artist or vibe…"
            className="w-full rounded-2xl border border-white/10 bg-white/5 py-4 pl-14 pr-5 text-base text-zinc-100 placeholder-zinc-500 shadow-glow outline-none transition focus:border-pink-500/50 focus:bg-white/10"
          />
        </div>
        {loading && <p className="mt-2 text-center text-xs text-zinc-500">Searching…</p>}
        {error && <p className="mt-2 text-center text-xs text-rose-400">{error}</p>}
      </div>

      {/* results grid */}
      {results && results.length === 0 && !loading && (
        <EmptyState title="No results" hint="Try a different search — artist plus song title usually works best." />
      )}
      {results && results.length > 0 && (
        <div className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-4">
          {results.map((r) => (
            <button
              key={r.video_id}
              onClick={() => void onSelect(r)}
              disabled={selecting !== null}
              className={`group overflow-hidden rounded-2xl border border-white/10 bg-white/5 text-left transition hover:-translate-y-0.5 hover:border-pink-500/40 hover:shadow-glow ${
                selecting === r.video_id ? 'opacity-60' : ''
              }`}
            >
              <div className="relative aspect-video w-full overflow-hidden bg-black/40">
                <img
                  src={r.thumbnail_url}
                  alt=""
                  loading="lazy"
                  className="h-full w-full object-cover transition group-hover:scale-105"
                />
                <span className="absolute bottom-2 right-2 rounded-md bg-black/80 px-1.5 py-0.5 text-[11px] font-medium tabular-nums text-zinc-200">
                  {fmtTime(r.duration_sec)}
                </span>
                {selecting === r.video_id && (
                  <span className="absolute inset-0 grid place-items-center bg-black/60 text-xs font-medium text-pink-300">
                    Starting analysis…
                  </span>
                )}
              </div>
              <div className="p-3">
                <p className="line-clamp-2 text-sm font-medium leading-snug text-zinc-100">
                  {r.title}
                </p>
                <p className="mt-1 truncate text-xs text-zinc-500">
                  {r.channel} · {abbrev(r.view_count)} views
                </p>
              </div>
            </button>
          ))}
        </div>
      )}

      {/* existing projects */}
      <Card
        title="…or pick up an existing project"
        subtitle="Projects live locally under %APPDATA%\enjoi\projects"
      >
        {projects.length === 0 ? (
          <p className="text-sm text-zinc-500">
            No projects yet — pick a reference above to start your first one.
          </p>
        ) : (
          <ul className="divide-y divide-white/5">
            {projects.map((p) => (
              <li key={p.id} className="flex items-center gap-3 py-2.5">
                {p.reference?.thumbnail_url ? (
                  <img
                    src={p.reference.thumbnail_url}
                    alt=""
                    className="h-10 w-16 shrink-0 rounded-lg object-cover"
                  />
                ) : (
                  <div className="grid h-10 w-16 shrink-0 place-items-center rounded-lg bg-white/5 text-zinc-600">
                    ♪
                  </div>
                )}
                <button
                  onClick={() => void openProject(p)}
                  className="min-w-0 flex-1 text-left transition hover:opacity-80"
                >
                  <p className="truncate text-sm font-medium text-zinc-200">{p.name}</p>
                  <p className="truncate text-xs text-zinc-500">
                    {p.reference ? p.reference.title : 'No reference yet'} ·{' '}
                    {new Date(p.created_at).toLocaleDateString()}
                  </p>
                </button>
                <span className="shrink-0 rounded-full border border-white/10 bg-white/5 px-2 py-0.5 text-[11px] text-zinc-400">
                  {stageLabel(p)}
                </span>
                <button
                  onClick={() => {
                    if (window.confirm(`Delete project "${p.name}"? This cannot be undone.`)) {
                      void deleteProject(p.id);
                    }
                  }}
                  title="Delete project"
                  className="shrink-0 rounded-lg px-2 py-1 text-zinc-600 transition hover:bg-rose-500/10 hover:text-rose-400"
                >
                  <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </button>
              </li>
            ))}
          </ul>
        )}
      </Card>

      <p className="pb-4 text-center text-xs text-zinc-600">
        The reference is used for analysis only — its audio never appears in your output, so the
        finished song is 100% yours.
      </p>
    </div>
  );
}
