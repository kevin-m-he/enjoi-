import { useEffect, useMemo, useState } from 'react';
import { api, mediaUrl } from '../lib/api';
import { useStore } from '../store';
import AudioPlayer from './AudioPlayer';
import Card from './Card';
import EmptyState from './EmptyState';
import JobProgressBar from './JobProgressBar';

const LIMIT_NOTE_KEY = 'enjoi.limitnote.dismissed';

/** Spec §4.3 mapping table — what each slider tier locks. */
const TIER_ROWS: { name: string; tiers: [string, string, string, string, string] }[] = [
  {
    name: 'BPM',
    tiers: ['random in genre range', '±15% of reference', 'within ±7%', 'within ±3%', 'exact match'],
  },
  {
    name: 'Key / mode',
    tiers: ['random', 'related key', 'same mode', 'same key family', 'exact key'],
  },
  {
    name: 'Section structure',
    tiers: ['free', 'loose (has a chorus)', 'same section count', 'same order', 'same order + same bar lengths'],
  },
  {
    name: 'Energy curve',
    tiers: ['free', 'trend only', 'per-section match', 'per-bar (loose)', 'per-bar (tight)'],
  },
  {
    name: 'Instrumentation palette',
    tiers: ['free', '1 shared stem role', '2 shared roles', '3 shared roles', 'full palette match'],
  },
  {
    name: 'Groove fingerprint',
    tiers: ['free', 'genre-typical', 'similar swing', 'similar pattern class', 'matched pattern class'],
  },
];

const CHECK_LABELS: Record<string, string> = {
  melody_ngram_overlap: 'Melody overlap',
  chord_run_length: 'Longest shared chord run',
  chroma_correlation: 'Chroma correlation',
  audio_fingerprint: 'Audio fingerprint matches',
};

function checkValue(name: string, value: number): string {
  if (name === 'melody_ngram_overlap' || name === 'chroma_correlation') {
    return `${Math.round(value * 100)}%`;
  }
  if (name === 'chord_run_length') return `${value} chords`;
  return String(value);
}

export default function SimilarityScreen() {
  const project = useStore((s) => s.project);
  const grid = useStore((s) => s.grid);
  const uniqueness = useStore((s) => s.uniqueness);
  const jobs = useStore((s) => s.jobs);
  const activeJobs = useStore((s) => s.activeJobs);
  const startGenerate = useStore((s) => s.startGenerate);
  const setStep = useStore((s) => s.setStep);

  const [value, setValue] = useState<number>(project?.similarity ?? 70);
  const [summary, setSummary] = useState<string>('');
  const [showLimitNote, setShowLimitNote] = useState<boolean>(
    () => localStorage.getItem(LIMIT_NOTE_KEY) !== '1'
  );

  const job = activeJobs.generate ? jobs[activeJobs.generate] : undefined;
  const generating = job?.status === 'queued' || job?.status === 'running';
  const tier = Math.min(4, Math.max(0, Math.round(value / 25)));

  // live summary from the backend, debounced 250 ms
  useEffect(() => {
    if (!project) return;
    const t = window.setTimeout(() => {
      api
        .similarityPreview(project.id, value)
        .then((r) => setSummary(r.summary))
        .catch(() =>
          setSummary(`${value}% style match target — melody & chords stay 100% original.`)
        );
    }, 250);
    return () => window.clearTimeout(t);
  }, [project, value]);

  const instrumentalUrl = useMemo(() => {
    if (!project?.instrumental) return null;
    // bust the cache after regeneration (same path, new audio)
    const v = encodeURIComponent(activeJobs.generate ?? 'initial');
    return `${mediaUrl(project.id, 'instrumental.wav')}?v=${v}`;
  }, [project, activeJobs.generate]);

  if (!project) {
    return <EmptyState title="No project open" hint="Start from the Search step." />;
  }

  const engineIsMusicGen = (project.instrumental?.engine ?? grid?.engine ?? '')
    .toLowerCase()
    .includes('musicgen');

  return (
    <div className="space-y-6 pt-4">
      <div className="text-center">
        <h2 className="text-3xl font-extrabold tracking-tight">
          How close to the <span className="text-grad">vibe</span>?
        </h2>
        <p className="mx-auto mt-2 max-w-xl text-sm text-zinc-400">
          The slider controls style only — tempo, key, structure, groove, energy, instrument
          palette. The melody and chords of your song are always written from scratch.
        </p>
      </div>

      {/* hero slider */}
      <Card className="px-8 py-8">
        <div className="mb-4 text-center">
          <span className="text-6xl font-black tabular-nums text-grad">{value}%</span>
        </div>
        <input
          type="range"
          className="slider-hero w-full"
          min={0}
          max={100}
          step={1}
          value={value}
          disabled={generating}
          onChange={(e) => setValue(Number(e.target.value))}
        />
        <div className="mt-2 flex justify-between text-[11px] text-zinc-500">
          <span>0% — only the song length is kept</span>
          <span>100% — same vibe, never the same song</span>
        </div>
        <p className="mt-4 min-h-[1.5rem] text-center text-sm font-medium text-amber-300/90">
          {summary}
        </p>
      </Card>

      {/* what locks at this value */}
      <Card title={`What locks at ${value}%`} subtitle="Style, never substance">
        <table className="w-full text-sm">
          <tbody>
            <tr className="border-b border-white/5">
              <td className="py-2 pr-4 text-zinc-400">Song length</td>
              <td className="py-2 font-medium text-zinc-200">
                always kept (±5%) <span className="text-zinc-500">— at every value</span>
              </td>
            </tr>
            {TIER_ROWS.map((row) => (
              <tr key={row.name} className="border-b border-white/5">
                <td className="py-2 pr-4 text-zinc-400">{row.name}</td>
                <td className="py-2 font-medium text-zinc-200">{row.tiers[tier]}</td>
              </tr>
            ))}
            <tr className="border-b border-white/5">
              <td className="py-2 pr-4 text-zinc-400">Melody</td>
              <td className="py-2 font-semibold text-emerald-300">always 100% original</td>
            </tr>
            <tr className="border-b border-white/5">
              <td className="py-2 pr-4 text-zinc-400">Chord progression</td>
              <td className="py-2 font-semibold text-emerald-300">always 100% original</td>
            </tr>
            <tr>
              <td className="py-2 pr-4 text-zinc-400">Reference audio in output</td>
              <td className="py-2 font-semibold text-emerald-300">never</td>
            </tr>
          </tbody>
        </table>
        <p className="mt-3 text-xs text-zinc-500">
          Style, never substance: the slider only ever controls non-copyrightable style
          descriptors. The melody, chord progression, lyrics and the recording itself are never
          copied or conditioned on at any value — including 100%. 100% means “same vibe”, never
          “same song”.
        </p>
      </Card>

      {showLimitNote && (
        <div className="flex items-start justify-between gap-4 rounded-2xl border border-amber-500/30 bg-amber-500/10 p-4 text-sm text-amber-200/90">
          <p>
            <span className="font-semibold">One honest note:</span> no software can{' '}
            <em>guarantee</em> legal non-infringement. The automated Uniqueness Guard enforces
            measurable divergence on melody, harmony and audio — which removes the practical
            copying risk — but it is not legal advice.
          </p>
          <button
            onClick={() => {
              localStorage.setItem(LIMIT_NOTE_KEY, '1');
              setShowLimitNote(false);
            }}
            className="shrink-0 rounded-lg border border-amber-400/40 px-3 py-1 text-xs font-medium transition hover:bg-amber-500/20"
          >
            Got it
          </button>
        </div>
      )}

      <div className="flex justify-center">
        <button
          onClick={() => void startGenerate(value)}
          disabled={generating}
          className="rounded-2xl bg-gradient-to-r from-pink-500 to-amber-500 px-10 py-4 text-base font-bold text-white shadow-glow transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {generating
            ? 'Generating…'
            : project.instrumental
              ? 'Regenerate Instrumental'
              : 'Generate Instrumental'}
        </button>
      </div>

      <JobProgressBar
        job={job}
        onRetry={() => void startGenerate(value)}
        hint="Includes the originality audit — sections that come out too close to the reference are regenerated automatically, so this bar can take a few extra passes."
      />

      {project.instrumental && instrumentalUrl && !generating && (
        <Card
          title="Your instrumental"
          actions={
            <span
              className={`rounded-full border px-3 py-1 text-xs font-medium ${
                engineIsMusicGen
                  ? 'border-violet-500/40 bg-violet-500/10 text-violet-300'
                  : 'border-white/10 bg-white/5 text-zinc-400'
              }`}
              title={
                engineIsMusicGen
                  ? `Generated with ${project.instrumental.engine}`
                  : 'Generated with the built-in procedural engine (MusicGen not available)'
              }
            >
              {engineIsMusicGen ? 'MusicGen' : 'Procedural engine'}
            </span>
          }
        >
          <AudioPlayer src={instrumentalUrl} title="instrumental.wav" />

          {uniqueness && (
            <div className="mt-4 rounded-xl border border-emerald-500/20 bg-emerald-500/5 p-4">
              <p
                className={`text-sm font-semibold ${
                  uniqueness.passed ? 'text-emerald-300' : 'text-rose-300'
                }`}
              >
                {uniqueness.summary}
              </p>
              <div className="mt-3 grid gap-2 sm:grid-cols-2">
                {Object.entries(uniqueness.checks).map(([name, c]) => (
                  <div
                    key={name}
                    className="flex items-center justify-between rounded-lg bg-black/20 px-3 py-2 text-xs"
                  >
                    <span className="text-zinc-400">{CHECK_LABELS[name] ?? name}</span>
                    <span className={c.passed ? 'text-emerald-300' : 'text-rose-300'}>
                      {checkValue(name, c.value)} {c.passed ? '✓' : '✗'}
                    </span>
                  </div>
                ))}
              </div>
              <p className="mt-2 text-[11px] text-zinc-500">
                {uniqueness.attempts} generation attempt{uniqueness.attempts === 1 ? '' : 's'} ·
                effective similarity {uniqueness.effective_similarity}%
              </p>
            </div>
          )}

          <div className="mt-4 flex justify-end">
            <button
              onClick={() => setStep(3)}
              className="rounded-xl bg-gradient-to-r from-pink-500 to-amber-500 px-6 py-3 text-sm font-semibold text-white shadow-glow transition hover:opacity-90"
            >
              Continue → Vocal
            </button>
          </div>
        </Card>
      )}
    </div>
  );
}
