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

  // Persisted in the store so navigating away and back keeps the user's choice.
  const value = useStore((s) => s.similarity);
  const setValue = useStore((s) => s.setSimilarity);
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
    return <EmptyState title="No project open" hint="Start from the Upload step." />;
  }

  const engineRaw = (project.instrumental?.engine ?? grid?.engine ?? '').toLowerCase();
  const engineLabel = engineRaw.includes('band')
    ? 'Live band'
    : engineRaw.includes('musicgen')
      ? 'MusicGen'
      : 'Procedural';
  const engineIsReal = engineRaw.includes('band') || engineRaw.includes('musicgen');

  return (
    <div className="space-y-6 pt-4">
      <div className="text-center">
        <h2 className="font-display text-4xl font-black uppercase tracking-tight text-ink">
          How close to the <span className="text-pink">vibe</span>?
        </h2>
        <p className="mx-auto mt-2 max-w-xl text-sm font-medium text-prussian-700">
          The slider controls style only — tempo, key, structure, groove, energy, instrument
          palette. The melody and chords of your song are always written from scratch.
        </p>
      </div>

      {/* hero slider */}
      <Card className="px-8 py-8">
        <div className="mb-5 text-center">
          <span className="font-display text-7xl font-black tabular-nums text-stroke-ink text-pink">
            {value}%
          </span>
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
        <div className="mt-2 flex justify-between text-[11px] font-bold uppercase text-prussian-700/70">
          <span>0% — only the song length is kept</span>
          <span>100% — same vibe, never the same song</span>
        </div>
        <p className="mt-4 min-h-[1.5rem] text-center text-sm font-bold text-pink">{summary}</p>
      </Card>

      {/* what locks at this value */}
      <Card title={`What locks at ${value}%`} subtitle="Style, never substance">
        <table className="w-full text-sm">
          <tbody>
            <tr className="border-b-2 border-ink/10">
              <td className="py-2 pr-4 font-bold text-prussian-700">Song length</td>
              <td className="py-2 font-bold text-ink">
                always kept (±5%) <span className="text-prussian-700/60">— at every value</span>
              </td>
            </tr>
            {TIER_ROWS.map((row) => (
              <tr key={row.name} className="border-b-2 border-ink/10">
                <td className="py-2 pr-4 font-bold text-prussian-700">{row.name}</td>
                <td className="py-2 font-bold text-ink">{row.tiers[tier]}</td>
              </tr>
            ))}
            <tr className="border-b-2 border-ink/10">
              <td className="py-2 pr-4 font-bold text-prussian-700">Melody</td>
              <td className="py-2 font-extrabold text-pink">always 100% original</td>
            </tr>
            <tr className="border-b-2 border-ink/10">
              <td className="py-2 pr-4 font-bold text-prussian-700">Chord progression</td>
              <td className="py-2 font-extrabold text-pink">always 100% original</td>
            </tr>
            <tr>
              <td className="py-2 pr-4 font-bold text-prussian-700">Reference audio in output</td>
              <td className="py-2 font-extrabold text-pink">never</td>
            </tr>
          </tbody>
        </table>
        <p className="mt-3 text-xs font-medium text-prussian-700/70">
          Style, never substance: the slider only ever controls non-copyrightable style
          descriptors. The melody, chord progression, lyrics and the recording itself are never
          copied or conditioned on at any value — including 100%. 100% means “same vibe”, never
          “same song”.
        </p>
      </Card>

      {showLimitNote && (
        <div className="flex items-start justify-between gap-4 rounded-brutal border-4 border-ink bg-foam-50 p-4 text-sm font-semibold text-ink shadow-brutal-sm">
          <p>
            <span className="font-extrabold uppercase">One honest note:</span> no software can{' '}
            <em>guarantee</em> legal non-infringement. The automated Uniqueness Guard enforces
            measurable divergence on melody, harmony and audio — which removes the practical
            copying risk — but it is not legal advice.
          </p>
          <button
            onClick={() => {
              localStorage.setItem(LIMIT_NOTE_KEY, '1');
              setShowLimitNote(false);
            }}
            className="shrink-0 rounded-sm border-3 border-ink bg-foam px-3 py-1 text-xs font-extrabold uppercase shadow-brutal-sm transition active:translate-x-[3px] active:translate-y-[3px] active:shadow-none"
          >
            Got it
          </button>
        </div>
      )}

      <div className="flex justify-center">
        <button
          onClick={() => void startGenerate(value)}
          disabled={generating}
          className="rounded-brutal border-4 border-ink bg-pink px-10 py-4 font-display text-base font-black uppercase tracking-tight text-white shadow-brutal-lg transition active:translate-x-[6px] active:translate-y-[6px] active:shadow-none disabled:cursor-not-allowed disabled:opacity-50"
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
        prominent
        hint="Playing the band and mixing it down — drums, bass, guitar/keys and melody, leveled into one cohesive instrumental."
      />

      {project.instrumental && instrumentalUrl && !generating && (
        <Card
          title="Your instrumental"
          actions={
            <span
              className={`rounded-sm border-2 border-ink px-3 py-1 text-xs font-extrabold uppercase ${
                engineIsReal ? 'bg-pink text-white' : 'bg-washi-200 text-prussian-700'
              }`}
              title={`Generated with the ${project.instrumental.engine} engine`}
            >
              {engineLabel}
            </span>
          }
        >
          <AudioPlayer src={instrumentalUrl} title="instrumental.wav" />

          {uniqueness && (
            <div className="mt-4 rounded-brutal border-3 border-ink bg-foam p-4">
              <p
                className={`text-sm font-extrabold ${
                  uniqueness.passed ? 'text-prussian-700' : 'text-pink'
                }`}
              >
                {uniqueness.summary}
              </p>
              <div className="mt-3 grid gap-2 sm:grid-cols-2">
                {Object.entries(uniqueness.checks).map(([name, c]) => (
                  <div
                    key={name}
                    className="flex items-center justify-between rounded-sm border-2 border-ink bg-washi-200 px-3 py-2 text-xs font-bold"
                  >
                    <span className="text-prussian-700">{CHECK_LABELS[name] ?? name}</span>
                    <span className={c.passed ? 'text-prussian-700' : 'text-pink'}>
                      {checkValue(name, c.value)} {c.passed ? '✓' : '✗'}
                    </span>
                  </div>
                ))}
              </div>
              <p className="mt-2 text-[11px] font-semibold text-prussian-700/70">
                {uniqueness.attempts} generation attempt{uniqueness.attempts === 1 ? '' : 's'} ·
                effective similarity {uniqueness.effective_similarity}%
              </p>
            </div>
          )}

          <div className="mt-4 flex justify-end">
            <button
              onClick={() => setStep(3)}
              className="rounded-brutal border-4 border-ink bg-pink px-6 py-3 font-display text-sm font-black uppercase tracking-tight text-white shadow-brutal transition active:translate-x-[6px] active:translate-y-[6px] active:shadow-none"
            >
              Continue ▶ Vocal
            </button>
          </div>
        </Card>
      )}
    </div>
  );
}
