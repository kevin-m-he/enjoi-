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
      className={`cursor-help select-none rounded-brutal border-3 border-ink px-2.5 py-1 text-xs font-extrabold uppercase tracking-tight shadow-brutal-sm transition ${
        ok ? 'bg-cyan text-ink' : 'bg-washi-200 text-prussian-700/60'
      }`}
    >
      {label}
    </span>
  );
}
