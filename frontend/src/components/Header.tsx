import CapabilityBadge from './CapabilityBadge';
import { useStore } from '../store';

export default function Header() {
  const health = useStore((s) => s.health);
  const project = useStore((s) => s.project);
  const wsConnected = useStore((s) => s.wsConnected);
  const caps = health?.capabilities;

  return (
    <header className="sticky top-0 z-30 border-b border-white/10 bg-[#0c0a14]/80 backdrop-blur">
      <div className="mx-auto flex h-16 w-full max-w-6xl items-center justify-between px-6">
        <div className="flex min-w-0 items-baseline gap-3">
          <h1 className="text-2xl font-black tracking-tight">
            <span className="text-grad">enjoi</span>{' '}
            <span className="text-zinc-200">享受</span>
          </h1>
          {project && (
            <span className="max-w-[18rem] truncate text-sm text-zinc-500">· {project.name}</span>
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
            className={`ml-1 inline-block h-2 w-2 rounded-full ${
              wsConnected ? 'bg-emerald-400' : 'bg-zinc-600'
            }`}
          />
        </div>
      </div>
    </header>
  );
}
