import { useEffect, useRef, useState } from 'react';
import type { ProjectState } from '../lib/types';
import { maxStepFor, STEPS, useStore } from '../store';
import Card from './Card';
import JobProgressBar from './JobProgressBar';

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
  const startReferenceUpload = useStore((s) => s.startReferenceUpload);
  const backendUp = useStore((s) => s.backendUp);
  const project = useStore((s) => s.project);
  const jobs = useStore((s) => s.jobs);
  const activeJobs = useStore((s) => s.activeJobs);
  const setStep = useStore((s) => s.setStep);

  const [selecting, setSelecting] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  // --- reference job state ---
  const refJob = activeJobs.reference ? jobs[activeJobs.reference] : undefined;
  const refRunning = refJob?.status === 'queued' || refJob?.status === 'running';
  const refDone = refJob?.status === 'done' || !!project?.reference?.analyzed;

  useEffect(() => {
    if (backendUp) void loadProjects();
  }, [backendUp, loadProjects]);

  const onUpload = async (file: File) => {
    if (selecting || refRunning) return;
    setSelecting('upload');
    try {
      await startReferenceUpload(file);
    } finally {
      setSelecting(null);
    }
  };

  return (
    <div className="space-y-8">
      <div className="pt-6 text-center">
        <h2 className="font-display text-4xl font-black uppercase tracking-tight text-ink">
          Find your <span className="text-pink">reference vibe</span>
        </h2>
        <p className="mx-auto mt-3 max-w-2xl text-base font-medium text-prussian-900">
          Upload a popular song of your choice — we analyze its style only and generate you your
          very own song based on the one-take vocals you upload later.{' '}
          <span className="font-extrabold text-pink">All yours.</span>
        </p>
      </div>

      {/* Upload your own audio — the reference. Works for any user. */}
      <div className="mx-auto max-w-2xl">
        <input
          ref={fileRef}
          type="file"
          accept="audio/*,.wav,.mp3,.m4a,.aac,.ogg,.flac,.opus"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            e.target.value = '';
            if (f) void onUpload(f);
          }}
        />
        <button
          onClick={() => fileRef.current?.click()}
          disabled={!!selecting || refRunning}
          className="flex w-full items-center justify-center gap-2 rounded-brutal border-4 border-ink bg-foam-50 px-5 py-5 font-display text-lg font-black uppercase tracking-tight text-ink shadow-brutal outline-none transition hover:-translate-x-[2px] hover:-translate-y-[2px] hover:bg-foam disabled:cursor-not-allowed disabled:opacity-50"
        >
          {selecting === 'upload' ? 'Uploading…' : '⬆ Upload your own audio'}
        </button>
        <p className="mt-2 text-center text-xs font-medium text-prussian-700/70">
          MP3, WAV, M4A… 15 sec–10 min. Analyzed for style only — never used in your song.
        </p>
      </div>

      {/* reference loading bar + "move on" gate */}
      {refJob && (
        <div className="mx-auto max-w-3xl space-y-3">
          <JobProgressBar
            job={refJob}
            prominent
            hint="Analyzing your track — extracting BPM, key, structure, energy and instrumentation…"
          />
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

      {/* existing projects */}
      <Card title="Uploaded projects" subtitle="Pick up a song you started earlier">
        {projects.length === 0 ? (
          <p className="text-sm font-medium text-prussian-900">
            No projects yet — upload a reference track above to start your first one.
          </p>
        ) : (
          <ul className="divide-y-2 divide-ink/15">
            {projects.map((p) => (
              <li key={p.id} className="flex items-center gap-3 py-2.5">
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
