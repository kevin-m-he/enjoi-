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
    <div className="flex flex-col items-center justify-center rounded-2xl border border-dashed border-white/10 bg-white/[0.02] px-6 py-12 text-center">
      <div className="mb-3 grid h-12 w-12 place-items-center rounded-full bg-white/5 text-2xl text-zinc-500">
        {icon}
      </div>
      <p className="text-sm font-medium text-zinc-300">{title}</p>
      {hint && <p className="mt-1 max-w-sm text-xs text-zinc-500">{hint}</p>}
    </div>
  );
}
