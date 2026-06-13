import type { Job } from '../lib/types';
import { clamp } from '../lib/format';
import { WaveSpinner } from './WaveArt';

const TYPE_LABEL: Record<string, string> = {
  reference: 'Analyzing reference',
  generate: 'Generating instrumental',
  vocal: 'Processing vocal take',
  rearrange: 'Re-detecting sections',
  render: 'Building your song',
};

export default function JobProgressBar({
  job,
  onRetry,
  hint,
  prominent,
}: {
  job?: Job;
  onRetry?: () => void;
  hint?: string;
  /** Larger, unmistakable wave-themed bar (used to gate "move on"). */
  prominent?: boolean;
}) {
  if (!job) return null;

  if (job.status === 'error') {
    return (
      <div className="rounded-brutal border-4 border-ink bg-pink p-4 text-white shadow-brutal">
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="font-display text-base font-extrabold uppercase tracking-tight">
              {TYPE_LABEL[job.type] ?? job.type} failed
            </div>
            <p className="mt-1 text-sm font-medium">{job.error || 'Unknown error.'}</p>
            <p className="mt-2 text-xs font-medium opacity-90">
              Check that the backend is still running and any required tools are installed (see the
              capability badges in the header), then retry this step.
            </p>
          </div>
          {onRetry && (
            <button
              onClick={onRetry}
              className="shrink-0 rounded-brutal border-3 border-ink bg-foam px-3 py-1.5 text-xs font-extrabold uppercase text-ink shadow-brutal-sm transition active:translate-x-[3px] active:translate-y-[3px] active:shadow-none"
            >
              Retry
            </button>
          )}
        </div>
      </div>
    );
  }

  const pct = Math.round(clamp(job.progress, 0, 1) * 100);
  const running = job.status === 'queued' || job.status === 'running';
  const done = job.status === 'done';

  const barHeight = prominent ? 'h-7' : 'h-4';

  return (
    <div className="rounded-brutal border-4 border-ink bg-cyan p-4 shadow-brutal">
      <div className="flex items-center justify-between gap-3">
        <span className="flex items-center gap-2 font-display text-sm font-extrabold uppercase tracking-tight text-ink">
          {running && <WaveSpinner className="h-5 w-5" />}
          {TYPE_LABEL[job.type] ?? job.type}
          {job.status === 'queued' ? ' · queued' : ''}
        </span>
        <span className="tabular-nums rounded-brutal border-2 border-ink bg-foam px-2 py-0.5 text-sm font-extrabold text-ink">
          {done ? 'DONE ✓' : `${pct}%`}
        </span>
      </div>

      {/* wave-themed track */}
      <div
        className={`mt-3 overflow-hidden rounded-sm border-3 border-ink bg-washi-200 ${barHeight}`}
      >
        <div
          className={`h-full border-r-3 border-ink bg-prussian ${
            running ? 'pixel-wave-fill animate-pixel-wave-march' : ''
          }`}
          style={{ width: `${done ? 100 : Math.max(pct, 3)}%` }}
        />
      </div>

      {job.message && (
        <p className="mt-2.5 text-sm font-semibold text-prussian-700">{job.message}</p>
      )}
      {hint && running && <p className="mt-1 text-[11px] font-medium text-prussian-700/70">{hint}</p>}
    </div>
  );
}
