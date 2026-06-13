import { Fragment } from 'react';
import { STEPS, maxStepFor, useStore } from '../store';

/** 6-step stepper; steps stay locked until their prerequisites are met (spec §3). */
export default function Stepper() {
  const step = useStore((s) => s.step);
  const project = useStore((s) => s.project);
  const setStep = useStore((s) => s.setStep);
  const max = Math.max(maxStepFor(project), step);

  return (
    <nav className="mx-auto w-full max-w-6xl px-6 pb-2 pt-5">
      <ol className="flex items-center">
        {STEPS.map((label, i) => {
          const unlocked = i <= max;
          const active = i === step;
          return (
            <Fragment key={label}>
              {i > 0 && (
                <div
                  className={`mx-2 h-px flex-1 ${
                    i <= max
                      ? 'bg-gradient-to-r from-pink-500/60 to-amber-500/60'
                      : 'bg-white/10'
                  }`}
                />
              )}
              <li>
                <button
                  onClick={() => unlocked && setStep(i)}
                  disabled={!unlocked}
                  title={unlocked ? label : 'Locked — complete the previous steps first'}
                  className={`group flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-medium transition ${
                    active
                      ? 'border-pink-500/60 bg-pink-500/15 text-pink-200 shadow-glow'
                      : unlocked
                        ? 'border-white/10 bg-white/5 text-zinc-300 hover:border-white/25'
                        : 'cursor-not-allowed border-white/5 bg-transparent text-zinc-600'
                  }`}
                >
                  <span
                    className={`grid h-5 w-5 place-items-center rounded-full text-[10px] font-bold ${
                      active
                        ? 'bg-gradient-to-br from-pink-500 to-amber-500 text-white'
                        : unlocked
                          ? 'bg-white/10 text-zinc-300'
                          : 'bg-white/5 text-zinc-600'
                    }`}
                  >
                    {unlocked && i < max && !active ? '✓' : !unlocked ? '🔒' : i + 1}
                  </span>
                  <span className="whitespace-nowrap">{label}</span>
                </button>
              </li>
            </Fragment>
          );
        })}
      </ol>
    </nav>
  );
}
