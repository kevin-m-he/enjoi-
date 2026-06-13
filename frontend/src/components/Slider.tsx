export default function Slider({
  label,
  value,
  min = 0,
  max = 100,
  step = 1,
  onChange,
  format,
  leftLabel,
  rightLabel,
  disabled,
  hero,
}: {
  label?: string;
  value: number;
  min?: number;
  max?: number;
  step?: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
  leftLabel?: string;
  rightLabel?: string;
  disabled?: boolean;
  hero?: boolean;
}) {
  return (
    <div>
      {(label || format) && (
        <div className="mb-1 flex items-center justify-between text-xs text-zinc-400">
          <span>{label}</span>
          <span className="tabular-nums font-medium text-zinc-200">
            {format ? format(value) : value}
          </span>
        </div>
      )}
      <input
        type="range"
        className={`w-full ${hero ? 'slider-hero' : ''}`}
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      {(leftLabel || rightLabel) && (
        <div className="mt-1 flex justify-between text-[11px] text-zinc-500">
          <span>{leftLabel}</span>
          <span>{rightLabel}</span>
        </div>
      )}
    </div>
  );
}
