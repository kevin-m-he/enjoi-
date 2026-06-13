import { useStore } from '../store';
import type { ToastKind } from '../store';

const KIND_CLASSES: Record<ToastKind, string> = {
  info: 'border-sky-500/40 bg-sky-950/80 text-sky-100',
  error: 'border-rose-500/40 bg-rose-950/80 text-rose-100',
  success: 'border-emerald-500/40 bg-emerald-950/80 text-emerald-100',
};

export default function Toasts() {
  const toasts = useStore((s) => s.toasts);
  const dismiss = useStore((s) => s.dismissToast);
  if (toasts.length === 0) return null;
  return (
    <div className="fixed bottom-6 right-6 z-50 flex w-96 max-w-[90vw] flex-col gap-2">
      {toasts.map((t) => (
        <div
          key={t.id}
          className={`flex items-start justify-between gap-3 rounded-xl border px-4 py-3 text-sm shadow-xl backdrop-blur ${KIND_CLASSES[t.kind]}`}
        >
          <span className="break-words">{t.text}</span>
          <button
            onClick={() => dismiss(t.id)}
            aria-label="Dismiss"
            className="shrink-0 text-lg leading-none opacity-60 transition hover:opacity-100"
          >
            ×
          </button>
        </div>
      ))}
    </div>
  );
}
