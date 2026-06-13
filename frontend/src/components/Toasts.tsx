import { useStore } from '../store';
import type { ToastKind } from '../store';

const KIND_CLASSES: Record<ToastKind, string> = {
  info: 'bg-cyan text-ink',
  error: 'bg-pink text-white',
  success: 'bg-prussian text-foam',
};

export default function Toasts() {
  const toasts = useStore((s) => s.toasts);
  const dismiss = useStore((s) => s.dismissToast);
  if (toasts.length === 0) return null;
  return (
    <div className="fixed bottom-6 right-6 z-50 flex w-96 max-w-[90vw] flex-col gap-3">
      {toasts.map((t) => (
        <div
          key={t.id}
          className={`flex items-start justify-between gap-3 rounded-brutal border-4 border-ink px-4 py-3 text-sm font-bold shadow-brutal ${KIND_CLASSES[t.kind]}`}
        >
          <span className="break-words">{t.text}</span>
          <button
            onClick={() => dismiss(t.id)}
            aria-label="Dismiss"
            className="shrink-0 text-lg font-black leading-none opacity-70 transition hover:opacity-100"
          >
            ×
          </button>
        </div>
      ))}
    </div>
  );
}
