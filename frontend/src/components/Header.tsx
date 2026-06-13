import CapabilityBadge from './CapabilityBadge';
import { FoamMark, PixelHeart } from './WaveArt';
import { useStore } from '../store';

export default function Header() {
  const health = useStore((s) => s.health);
  const project = useStore((s) => s.project);
  const wsConnected = useStore((s) => s.wsConnected);
  const caps = health?.capabilities;

  return (
    <header className="sticky top-0 z-30 border-b-4 border-ink bg-washi">
      <div className="relative mx-auto flex h-16 w-full max-w-6xl items-center justify-between px-6">
        {/* Support the dev — dead-center in the bar, shown on wide screens */}
        <a
          href="https://venmo.com/u/Kevin-He-516"
          target="_blank"
          rel="noopener noreferrer"
          title="Support the developer on Venmo"
          className="absolute left-1/2 top-1/2 hidden -translate-x-1/2 -translate-y-1/2 items-center gap-1.5 whitespace-nowrap text-xs font-bold text-prussian-700 transition hover:text-ink lg:flex"
        >
          <PixelHeart className="h-3 w-[14px] shrink-0 text-cyan" />
          <span>Support the Developer: Venmo @Kevin-He-516</span>
          <PixelHeart className="h-3 w-[14px] shrink-0 text-pink" />
        </a>
        <div className="flex min-w-0 items-center gap-3">
          <FoamMark className="hidden h-7 w-11 shrink-0 sm:block" />
          <h1 className="font-display text-3xl font-black tracking-tight">
            <span className="text-wave">enjoi</span>{' '}
            <span className="text-ink">享受</span>
          </h1>
          {/* 朱印 seal-stamp touch */}
          <span
            title="enjoi"
            className="hidden h-7 w-7 shrink-0 place-items-center rounded-brutal border-2 border-pink bg-pink/10 text-[11px] font-black text-pink sm:grid"
          >
            朱
          </span>
          {project && (
            <span className="max-w-[16rem] truncate border-l-3 border-ink pl-3 text-sm font-bold text-prussian-700">
              {project.name}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <CapabilityBadge
            label="GPU"
            ok={!!caps?.gpu}
            tooltipOn="GPU detected — fast instrumental generation."
            tooltipOff="No GPU detected — generation falls back to CPU (slower, smaller model)."
          />
          <CapabilityBadge
            label="MusicGen"
            ok={!!caps?.musicgen}
            tooltipOn="MusicGen is available for instrumental generation."
            tooltipOff="MusicGen not installed — the built-in procedural engine will generate instrumentals instead."
          />
          <CapabilityBadge
            label="Whisper"
            ok={!!caps?.whisper}
            tooltipOn="Whisper is available for lyric transcription."
            tooltipOff="Whisper not installed — vocals are segmented by energy only and no lyric transcript is produced."
          />
          <span
            title={
              wsConnected
                ? 'Live job updates connected'
                : 'Live updates offline — falling back to polling'
            }
            className={`ml-1 inline-block h-3 w-3 border-2 border-ink ${
              wsConnected ? 'bg-cyan' : 'bg-washi-200'
            }`}
          />
        </div>
      </div>
    </header>
  );
}
