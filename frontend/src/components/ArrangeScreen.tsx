import { useMemo, useRef, useState } from 'react';
import type { PointerEvent as ReactPointerEvent } from 'react';
import { mediaUrl } from '../lib/api';
import { clamp, fmtTime, roleLabel, sectionColor } from '../lib/format';
import type { Placement } from '../lib/types';
import { DEFAULT_WEIGHTS, SCORE_WEIGHT_KEYS } from '../lib/types';
import { useStore } from '../store';
import AudioPlayer from './AudioPlayer';
import Card from './Card';
import EmptyState from './EmptyState';
import JobProgressBar from './JobProgressBar';
import Slider from './Slider';

/** Seconds a placement occupies on the timeline. */
function placementLen(p: Placement): number {
  return Math.max((p.source_end - p.source_start) * (p.stretch || 1), 0.25);
}

function pickTickStep(duration: number): number {
  const candidates = [5, 10, 15, 30, 60, 120];
  for (const c of candidates) {
    if (duration / c <= 10) return c;
  }
  return 240;
}

const WEIGHT_LABELS: Record<string, string> = {
  energy: 'Energy peak',
  pitch_range: 'Pitch range',
  pitch_height: 'Pitch height',
  vibrato: 'Vibrato intensity',
  repetition: 'Repetition',
  brightness: 'Spectral brightness',
  hookiness: 'Lyric hookiness',
};

export default function ArrangeScreen() {
  const project = useStore((s) => s.project);
  const grid = useStore((s) => s.grid);
  const arrangement = useStore((s) => s.arrangement);
  const vocalAnalysis = useStore((s) => s.vocalAnalysis);
  const jobs = useStore((s) => s.jobs);
  const activeJobs = useStore((s) => s.activeJobs);
  const saveArrangement = useStore((s) => s.saveArrangement);
  const startRearrange = useStore((s) => s.startRearrange);
  const setStep = useStore((s) => s.setStep);

  const laneRef = useRef<HTMLDivElement | null>(null);
  const dragRef = useRef<{
    id: number;
    pointerId: number;
    startX: number;
    origStart: number;
    moved: boolean;
  } | null>(null);
  const [dragPos, setDragPos] = useState<{ id: number; start: number } | null>(null);
  const [saving, setSaving] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [weights, setWeights] = useState<Record<string, number>>(() => ({
    ...DEFAULT_WEIGHTS,
    ...(vocalAnalysis?.weights ?? {}),
  }));

  const rearrangeJob = activeJobs.rearrange ? jobs[activeJobs.rearrange] : undefined;
  const rearranging = rearrangeJob?.status === 'queued' || rearrangeJob?.status === 'running';

  const duration = useMemo(() => {
    if (grid?.duration_sec) return grid.duration_sec;
    const slotEnd = arrangement?.slots.reduce((m, s) => Math.max(m, s.end), 0) ?? 0;
    return Math.max(slotEnd, 1);
  }, [grid, arrangement]);

  const tickStep = pickTickStep(duration);
  const ticks = useMemo(() => {
    const out: number[] = [];
    for (let t = 0; t <= duration; t += tickStep) out.push(t);
    return out;
  }, [duration, tickStep]);

  if (!project) {
    return <EmptyState title="No project open" hint="Start from the Search step." />;
  }

  const placements = arrangement?.placements ?? [];

  const save = async (next: Placement[]) => {
    setSaving(true);
    try {
      await saveArrangement(next);
    } finally {
      setSaving(false);
    }
  };

  const onPillPointerDown = (e: ReactPointerEvent, p: Placement) => {
    if (saving || rearranging) return;
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
    dragRef.current = {
      id: p.id,
      pointerId: e.pointerId,
      startX: e.clientX,
      origStart: p.target_start,
      moved: false,
    };
  };

  const onPillPointerMove = (e: ReactPointerEvent, p: Placement) => {
    const d = dragRef.current;
    if (!d || d.id !== p.id || e.pointerId !== d.pointerId) return;
    const lane = laneRef.current;
    if (!lane) return;
    const dx = e.clientX - d.startX;
    if (Math.abs(dx) > 4) d.moved = true;
    if (!d.moved) return;
    const dsec = (dx / Math.max(lane.clientWidth, 1)) * duration;
    const len = placementLen(p);
    setDragPos({
      id: d.id,
      start: clamp(d.origStart + dsec, 0, Math.max(0, duration - len)),
    });
  };

  const onPillPointerUp = (_e: ReactPointerEvent, p: Placement) => {
    const d = dragRef.current;
    if (!d || d.id !== p.id) return;
    dragRef.current = null;
    const pos = dragPos;
    setDragPos(null);

    if (!d.moved) {
      // simple click → toggle enabled (PUT enabled:false / true)
      const next = placements.map((x) =>
        x.id === p.id ? { ...x, enabled: x.enabled === false } : x
      );
      void save(next);
      return;
    }
    if (pos) {
      const next = placements.map((x) =>
        x.id === p.id ? { ...x, target_start: Math.round(pos.start * 1000) / 1000 } : x
      );
      void save(next);
    }
  };

  return (
    <div className="space-y-6 pt-4">
      <div className="flex items-end justify-between gap-4">
        <div>
          <h2 className="text-3xl font-extrabold tracking-tight">
            Your <span className="text-grad">arrangement</span>
          </h2>
          <p className="mt-1 text-sm text-zinc-400">
            Drag a vocal pill left or right to move it. Click a pill to mute / unmute it. The
            server keeps everything on the beat grid and rejects overlaps.
          </p>
        </div>
        <button
          onClick={() => void startRearrange(weights)}
          disabled={rearranging || saving}
          className="shrink-0 rounded-xl border border-white/15 bg-white/5 px-4 py-2.5 text-sm font-medium text-zinc-200 transition hover:border-pink-500/40 disabled:opacity-50"
          title="Re-run chorus/bridge detection and placement with the weights below"
        >
          {rearranging ? 'Re-detecting…' : '↻ Re-detect'}
        </button>
      </div>

      <JobProgressBar job={rearrangeJob} onRetry={() => void startRearrange(weights)} />

      {!arrangement && !rearranging && (
        <EmptyState
          title="No arrangement yet"
          hint="Upload a vocal take first — the arrangement is built automatically from it."
        />
      )}

      {arrangement && (
        <Card>
          {/* vocal lane */}
          <div ref={laneRef} className="relative mb-1 h-16 w-full select-none">
            <span className="absolute -top-1 left-0 text-[10px] uppercase tracking-wider text-zinc-600">
              vocals
            </span>
            {placements.map((p) => {
              const start = dragPos?.id === p.id ? dragPos.start : p.target_start;
              const len = placementLen(p);
              const disabled = p.enabled === false;
              return (
                <div
                  key={p.id}
                  onPointerDown={(e) => onPillPointerDown(e, p)}
                  onPointerMove={(e) => onPillPointerMove(e, p)}
                  onPointerUp={(e) => onPillPointerUp(e, p)}
                  title={`${roleLabel(p.role)} · ${len.toFixed(1)}s · slot ${p.slot_label} — drag to move, click to ${disabled ? 'enable' : 'mute'}`}
                  className={`absolute top-4 flex h-10 cursor-grab touch-none items-center gap-1.5 overflow-hidden rounded-full border px-3 text-xs font-semibold shadow-lg transition-opacity active:cursor-grabbing ${
                    disabled ? 'opacity-35' : 'opacity-100'
                  } ${
                    dragPos?.id === p.id ? 'z-20 ring-2 ring-white/40' : 'z-10'
                  }`}
                  style={{
                    left: `${(start / duration) * 100}%`,
                    width: `${Math.max((len / duration) * 100, 3)}%`,
                    backgroundColor: `${sectionColor(p.role)}33`,
                    borderColor: `${sectionColor(p.role)}aa`,
                    color: sectionColor(p.role),
                  }}
                >
                  <span className="truncate">
                    {roleLabel(p.role)}
                    {disabled ? ' (muted)' : ''}
                  </span>
                  <span className="shrink-0 font-normal tabular-nums opacity-70">
                    {len.toFixed(1)}s
                  </span>
                </div>
              );
            })}
          </div>

          {/* instrumental section lane */}
          <div className="relative h-12 w-full overflow-hidden rounded-lg bg-black/30">
            {(grid?.sections ?? arrangement.slots.map((s) => ({ label: s.label, start: s.start, end: s.end, bars: 0 }))).map(
              (s, i) => (
                <div
                  key={i}
                  title={`${s.label} · ${fmtTime(s.start)}–${fmtTime(s.end)}`}
                  className="absolute top-0 flex h-full items-center justify-center overflow-hidden border-r border-black/40 text-[10px] font-medium text-black/70"
                  style={{
                    left: `${(s.start / duration) * 100}%`,
                    width: `${((s.end - s.start) / duration) * 100}%`,
                    backgroundColor: `${sectionColor(s.label)}cc`,
                  }}
                >
                  <span className="truncate px-1">{s.label}</span>
                </div>
              )
            )}
          </div>

          {/* time axis */}
          <div className="relative mt-1 h-5 w-full text-[10px] tabular-nums text-zinc-500">
            {ticks.map((t) => (
              <span
                key={t}
                className="absolute -translate-x-1/2"
                style={{ left: `${(t / duration) * 100}%` }}
              >
                {fmtTime(t)}
              </span>
            ))}
          </div>

          {saving && <p className="mt-2 text-xs text-zinc-500">Saving placement…</p>}
        </Card>
      )}

      {/* advanced re-detect weights */}
      <Card
        title="Advanced — Impact Score weights"
        subtitle="Used by Re-detect to re-pick the chorus, verses and bridge"
        actions={
          <button
            onClick={() => setShowAdvanced((v) => !v)}
            className="rounded-lg border border-white/10 px-3 py-1 text-xs text-zinc-400 transition hover:border-white/25"
          >
            {showAdvanced ? 'Hide' : 'Show'}
          </button>
        }
      >
        {showAdvanced ? (
          <div className="grid gap-x-8 gap-y-4 sm:grid-cols-2">
            {SCORE_WEIGHT_KEYS.map((k) => (
              <Slider
                key={k}
                label={WEIGHT_LABELS[k]}
                value={weights[k] ?? DEFAULT_WEIGHTS[k]}
                min={0}
                max={1}
                step={0.05}
                format={(v) => v.toFixed(2)}
                onChange={(v) => setWeights((w) => ({ ...w, [k]: v }))}
                disabled={rearranging}
              />
            ))}
          </div>
        ) : (
          <p className="text-xs text-zinc-500">
            Seven tunable weights (energy, pitch range, pitch height, vibrato, repetition,
            brightness, hookiness). Open to tweak, then press Re-detect.
          </p>
        )}
      </Card>

      <Card title="Instrumental" subtitle="What the vocals sit on">
        <AudioPlayer
          src={`${mediaUrl(project.id, 'instrumental.wav')}?v=${encodeURIComponent(activeJobs.generate ?? 'initial')}`}
          title="instrumental.wav"
        />
      </Card>

      <div className="flex justify-end pb-6">
        <button
          onClick={() => setStep(5)}
          disabled={!arrangement}
          className="rounded-xl bg-gradient-to-r from-pink-500 to-amber-500 px-6 py-3 text-sm font-semibold text-white shadow-glow transition hover:opacity-90 disabled:opacity-50"
        >
          Continue → Mix & Export
        </button>
      </div>
    </div>
  );
}
