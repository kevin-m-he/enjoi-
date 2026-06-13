import type { Job } from '../lib/types';
import { clamp } from '../lib/format';

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
}: {
  job?: Job;
  onRetry?: () => void;
  hint?: string;
}) {
  if (!job) return null;

  if (job.status === 'error') {
    return (
      <div className="rounded-2xl border border-rose-500/40 bg-rose-500/10 p-4">
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="text-sm font-semibold text-rose-300">
              {TYPE_LABEL[job.type] ?? job.type} failed
            </div>
            <p className="mt-1 text-sm text-rose-200/80">{job.error || 'Unknown error.'}</p>
            <p className="mt-2 text-xs text-zinc-400">
              Check that the backend is still running and any required tools are installed (see
              the capability badges in the header), then retry this step.
            </p>
          </div>
          {onRetry && (
            <button
              onClick={onRetry}
              className="shrink-0 rounded-lg border border-rose-400/40 px-3 py-1.5 text-xs font-medium text-rose-200 transition hover:bg-rose-500/20"
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

  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
      <div className="flex items-center justify-between text-sm">
        <span className="font-medium text-zinc-200">
          {TYPE_LABEL[job.type] ?? job.type}
          {job.status === 'queued' ? ' (queued)' : ''}
        </span>
        <span className="tabular-nums text-zinc-400">
          {job.status === 'done' ? 'Done ✓' : `${pct}%`}
        </span>
      </div>
      <div className="mt-2 h-2.5 overflow-hidden rounded-full bg-white/10">
        <div
          className={`h-full rounded-full bg-gradient-to-r from-pink-500 to-amber-500 transition-all duration-500 ${
            running ? 'animate-pulse' : ''
          }`}
          style={{ width: `${job.status === 'done' ? 100 : Math.max(pct, 2)}%` }}
        />
      </div>
      {job.message && <p className="mt-2 text-xs text-zinc-400">{job.message}</p>}
      {hint && running && <p className="mt-1 text-[11px] text-zinc-500">{hint}</p>}
    </div>
  );
}
