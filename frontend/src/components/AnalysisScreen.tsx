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
      <div className="flex h-10 w-full overflow-hidden rounded-sm border-3 border-ink">
        {structure.map((s, i) => (
          <div
            key={i}
            title={`${s.label} · ${fmtTime(s.start)}–${fmtTime(s.end)} · ${s.bars} bars`}
            className="h-full border-r-2 border-ink last:border-r-0"
            style={{
              width: `${Math.max(((s.end - s.start) / Math.max(duration, 1)) * 100, 0.5)}%`,
              backgroundColor: sectionColor(s.label),
            }}
          />
        ))}
      </div>
      <div className="mt-2 flex flex-wrap gap-3 text-xs font-bold text-prussian-700">
        {labels.map((l) => (
          <span key={l} className="inline-flex items-center gap-1.5">
            <span
              className="h-3 w-3 border-2 border-ink"
              style={{ backgroundColor: sectionColor(l) }}
            />
            {l}
          </span>
        ))}
      </div>
    </div>
  );
}

function EnergySparkline({ values }: { values: number[] }) {
  if (values.length === 0) {
    return <p className="text-xs font-medium text-prussian-700/70">No energy data.</p>;
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
          <stop offset="0" stopColor="#00E5FF" />
          <stop offset="1" stopColor="#FF2D95" />
        </linearGradient>
      </defs>
      <polygon points={`0,${H} ${pts} ${W},${H}`} fill="url(#energy-grad)" opacity="0.2" />
      <polyline
        points={pts}
        fill="none"
        stroke="#0B0B0C"
        strokeWidth="3"
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
          className="h-16 w-28 rounded-sm border-3 border-ink object-cover shadow-brutal-sm"
        />
        <div className="min-w-0">
          <h2 className="truncate font-display text-2xl font-black uppercase tracking-tight text-ink">
            {project.reference.title}
          </h2>
          <p className="truncate text-sm font-medium text-prussian-700">
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
            prominent
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
              <div
                key={t.label}
                className="rounded-brutal border-4 border-ink bg-foam-50 p-4 shadow-brutal-sm"
              >
                <p className="text-xs font-extrabold uppercase tracking-wide text-prussian-700/70">
                  {t.label}
                </p>
                <p className="mt-1 font-display text-2xl font-black text-ink">{t.value}</p>
                {t.sub && <p className="text-[11px] font-semibold text-prussian-700/70">{t.sub}</p>}
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
                    <div className="mb-1 flex justify-between text-xs font-bold text-prussian-700">
                      <span className="capitalize">{name}</span>
                      <span className="tabular-nums">{Math.round(v * 100)}%</span>
                    </div>
                    <div className="h-3 overflow-hidden rounded-sm border-2 border-ink bg-washi-200">
                      <div
                        className="h-full bg-prussian"
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
                  className="rounded-sm border-2 border-ink bg-pink px-3 py-1 text-xs font-extrabold uppercase text-white"
                >
                  {g}
                </span>
              ))}
              {profile.mood_tags.map((m) => (
                <span
                  key={`m-${m}`}
                  className="rounded-sm border-2 border-ink bg-cyan px-3 py-1 text-xs font-extrabold uppercase text-ink"
                >
                  {m}
                </span>
              ))}
              <span className="rounded-sm border-2 border-ink bg-washi-200 px-3 py-1 text-xs font-bold text-prussian-700">
                groove: {profile.groove.pattern_class} · swing{' '}
                {Math.round(profile.groove.swing * 100)}%
              </span>
            </div>
          </Card>

          <div className="flex justify-end pb-6">
            <button
              onClick={() => setStep(2)}
              className="rounded-brutal border-4 border-ink bg-pink px-6 py-3 font-display text-sm font-black uppercase tracking-tight text-white shadow-brutal transition active:translate-x-[6px] active:translate-y-[6px] active:shadow-none"
            >
              Continue ▶ Similarity
            </button>
          </div>
        </>
      )}
    </div>
  );
}
