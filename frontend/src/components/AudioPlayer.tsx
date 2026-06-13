import { useEffect, useRef, useState } from 'react';
import { fmtTime } from '../lib/format';

const PlayIcon = (
  <svg viewBox="0 0 24 24" className="h-4 w-4 translate-x-[1px]" fill="currentColor">
    <path d="M8 5.14v13.72c0 .8.87 1.3 1.56.88l11-6.86a1.04 1.04 0 0 0 0-1.76l-11-6.86A1.04 1.04 0 0 0 8 5.14z" />
  </svg>
);

const PauseIcon = (
  <svg viewBox="0 0 24 24" className="h-4 w-4" fill="currentColor">
    <rect x="6" y="5" width="4" height="14" rx="1" />
    <rect x="14" y="5" width="4" height="14" rx="1" />
  </svg>
);

/** Styled <audio> wrapper with play/pause, seek and time readout. */
export default function AudioPlayer({ src, title }: { src: string; title?: string }) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    setPlaying(false);
    setTime(0);
    setDuration(0);
  }, [src]);

  const toggle = () => {
    const a = audioRef.current;
    if (!a) return;
    if (a.paused) void a.play().catch(() => undefined);
    else a.pause();
  };

  return (
    <div className="flex items-center gap-3 rounded-brutal border-3 border-ink bg-foam-50 px-4 py-3 shadow-brutal-sm">
      <audio
        ref={audioRef}
        src={src}
        preload="metadata"
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
        onTimeUpdate={(e) => setTime(e.currentTarget.currentTime)}
        onLoadedMetadata={(e) => setDuration(e.currentTarget.duration)}
      />
      <button
        onClick={toggle}
        aria-label={playing ? 'Pause' : 'Play'}
        className="grid h-11 w-11 shrink-0 place-items-center rounded-brutal border-3 border-ink bg-pink text-white shadow-brutal-sm transition active:translate-x-[3px] active:translate-y-[3px] active:shadow-none"
      >
        {playing ? PauseIcon : PlayIcon}
      </button>
      <div className="min-w-0 flex-1">
        {title && (
          <div className="mb-1 truncate text-xs font-bold uppercase tracking-tight text-prussian-900">
            {title}
          </div>
        )}
        <div className="flex items-center gap-2 text-xs font-semibold tabular-nums text-prussian-900">
          <span className="w-9 text-right">{fmtTime(time)}</span>
          <input
            type="range"
            className="flex-1"
            min={0}
            max={duration || 0}
            step={0.1}
            value={Math.min(time, duration || 0)}
            onChange={(e) => {
              const a = audioRef.current;
              if (!a) return;
              a.currentTime = Number(e.target.value);
              setTime(a.currentTime);
            }}
          />
          <span className="w-9">{fmtTime(duration)}</span>
        </div>
      </div>
    </div>
  );
}
