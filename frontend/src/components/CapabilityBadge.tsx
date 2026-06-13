export default function CapabilityBadge({
  label,
  ok,
  tooltipOn,
  tooltipOff,
}: {
  label: string;
  ok: boolean;
  tooltipOn: string;
  tooltipOff: string;
}) {
  return (
    <span
      title={ok ? tooltipOn : tooltipOff}
      className={`cursor-help select-none rounded-full border px-2.5 py-1 text-xs font-medium transition ${
        ok
          ? 'border-pink-500/40 bg-pink-500/10 text-pink-300'
          : 'border-white/10 bg-white/5 text-zinc-500'
      }`}
    >
      {label}
    </span>
  );
}
