import { fmtTime, keyName, sectionColor } from '../lib/format';
import type { StructureSection } from '../lib/types';
import { useStore } from '../store';
import Card from './Card';
import EmptyState from './EmptyState';
import JobProgressBar from './JobProgressBar';

function StructureStrip({
  structure,
  duration,
}: {
  structure: StructureSection[];
  duration: number;
}) {
  const labels = [...new Set(structure.map((s) => s.label))];
  return (
    <div>
      <div className="flex h-9 w-full overflow-hidden rounded-lg">
        {structure.map((s, i) => (
          <div
            key={i}
            title={`${s.label} · ${fmtTime(s.start)}–${fmtTime(s.end)} · ${s.bars} bars`}
            className="h-full border-r border-black/30 opacity-90 last:border-r-0"
            style={{
              width: `${Math.max(((s.end - s.start) / Math.max(duration, 1)) * 100, 0.5)}%`,
              backgroundColor: sectionColor(s.label),
            }}
          />
        ))}
      </div>
      <div className="mt-2 flex flex-wrap gap-3 text-xs text-zinc-400">
        {labels.map((l) => (
          <span key={l} className="inline-flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: sectionColor(l) }} />
            {l}
          </span>
        ))}
      </div>
    </div>
  );
}

function EnergySparkline({ values }: { values: number[] }) {
  if (values.length === 0) {
    return <p className="text-xs text-zinc-500">No energy data.</p>;
  }
  const W = 600;
  const H = 70;
  const max = Math.max(...values, 1e-9);
  const pts = values
    .map(
      (v, i) =>
        `${(i / Math.max(values.length - 1, 1)) * W},${H - 4 - (Math.max(v, 0) / max) * (H - 10)}`
    )
    .join(' ');
  return (
    <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="h-20 w-full">
      <defs>
        <linearGradient id="energy-grad" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0" stopColor="#ec4899" />
          <stop offset="1" stopColor="#f59e0b" />
        </linearGradient>
      </defs>
      <polygon points={`0,${H} ${pts} ${W},${H}`} fill="url(#energy-grad)" opacity="0.15" />
      <polyline
        points={pts}
        fill="none"
        stroke="url(#energy-grad)"
        strokeWidth="2"
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  );
}

export default function AnalysisScreen() {
  const project = useStore((s) => s.project);
  const profile = useStore((s) => s.profile);
  const jobs = useStore((s) => s.jobs);
  const activeJobs = useStore((s) => s.activeJobs);
  const setStep = useStore((s) => s.setStep);
  const retryReference = useStore((s) => s.retryReference);

  const job = activeJobs.reference ? jobs[activeJobs.reference] : undefined;

  if (!project?.reference) {
    return (
      <EmptyState
        title="No reference selected"
        hint="Go back to Search and pick a video to analyze."
      />
    );
  }

  return (
    <div className="space-y-6 pt-4">
      <div className="flex items-center gap-4">
        <img
          src={project.reference.thumbnail_url}
          alt=""
          className="h-16 w-28 rounded-xl object-cover"
        />
        <div className="min-w-0">
          <h2 className="truncate text-xl font-bold text-zinc-100">{project.reference.title}</h2>
          <p className="truncate text-sm text-zinc-500">
            {project.reference.channel} · {fmtTime(project.reference.duration_sec)} — analyzed for
            style only, never copied into your song
          </p>
        </div>
      </div>

      {!profile && (
        <>
          <JobProgressBar
            job={job}
            onRetry={() => void retryReference()}
            hint="Downloading audio for analysis, extracting BPM, key, structure, energy and instrumentation…"
          />
          {!job && (
            <EmptyState
              title="Waiting for analysis to start…"
              hint="If nothing happens, retry from the Search step."
            />
          )}
        </>
      )}

      {profile && (
        <>
          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            {[
              { label: 'BPM', value: profile.bpm.toFixed(1) },
              {
                label: 'Key',
                value: keyName(profile.key),
                sub:
                  profile.key.confidence !== undefined
                    ? `${Math.round(profile.key.confidence * 100)}% confidence`
                    : undefined,
              },
              { label: 'Time signature', value: profile.time_signature },
              { label: 'Duration', value: fmtTime(profile.duration_sec) },
            ].map((t) => (
              <div key={t.label} className="rounded-2xl border border-white/10 bg-white/5 p-4">
                <p className="text-xs uppercase tracking-wide text-zinc-500">{t.label}</p>
                <p className="mt-1 text-2xl font-bold text-zinc-100">{t.value}</p>
                {t.sub && <p className="text-[11px] text-zinc-500">{t.sub}</p>}
              </div>
            ))}
          </div>

          <Card title="Song structure" subtitle="Section map extracted from the reference">
            <StructureStrip structure={profile.structure} duration={profile.duration_sec} />
          </Card>

          <div className="grid gap-4 md:grid-cols-2">
            <Card title="Energy curve" subtitle="Per-bar RMS">
              <EnergySparkline values={profile.energy_curve.per_bar_rms} />
            </Card>
            <Card title="Instrumentation" subtitle="Per-stem activity (Demucs profile)">
              <div className="space-y-3">
                {Object.entries(profile.instrumentation).map(([name, v]) => (
                  <div key={name}>
                    <div className="mb-1 flex justify-between text-xs text-zinc-400">
                      <span className="capitalize">{name}</span>
                      <span className="tabular-nums">{Math.round(v * 100)}%</span>
                    </div>
                    <div className="h-2 overflow-hidden rounded-full bg-white/10">
                      <div
                        className="h-full rounded-full bg-gradient-to-r from-pink-500 to-amber-500"
                        style={{ width: `${Math.min(Math.max(v, 0), 1) * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          </div>

          <Card title="Genre & mood">
            <div className="flex flex-wrap gap-2">
              {profile.genre_tags.map((g) => (
                <span
                  key={`g-${g}`}
                  className="rounded-full border border-pink-500/40 bg-pink-500/10 px-3 py-1 text-xs font-medium text-pink-300"
                >
                  {g}
                </span>
              ))}
              {profile.mood_tags.map((m) => (
                <span
                  key={`m-${m}`}
                  className="rounded-full border border-amber-500/40 bg-amber-500/10 px-3 py-1 text-xs font-medium text-amber-300"
                >
                  {m}
                </span>
              ))}
              <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-zinc-400">
                groove: {profile.groove.pattern_class} · swing{' '}
                {Math.round(profile.groove.swing * 100)}%
              </span>
            </div>
          </Card>

          <div className="flex justify-end pb-6">
            <button
              onClick={() => setStep(2)}
              className="rounded-xl bg-gradient-to-r from-pink-500 to-amber-500 px-6 py-3 text-sm font-semibold text-white shadow-glow transition hover:opacity-90"
            >
              Continue → Similarity
            </button>
          </div>
        </>
      )}
    </div>
  );
}
