export default function EmptyState({
  title,
  hint,
  icon = '♪',
}: {
  title: string;
  hint?: string;
  icon?: string;
}) {
  return (
    <div className="flex flex-col items-center justify-center rounded-brutal border-4 border-dashed border-ink bg-cyan px-6 py-12 text-center">
      <div className="mb-3 grid h-14 w-14 place-items-center rounded-brutal border-3 border-ink bg-pink text-2xl text-white shadow-brutal-sm">
        {icon}
      </div>
      <p className="font-display text-base font-extrabold uppercase tracking-tight text-ink">
        {title}
      </p>
      {hint && <p className="mt-1 max-w-sm text-xs font-medium text-prussian-700/80">{hint}</p>}
    </div>
  );
}
