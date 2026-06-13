import { Fragment } from 'react';
import { STEPS, maxStepFor, useStore } from '../store';
import { LockIcon } from './WaveArt';

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
          const done = unlocked && i < max && !active;
          return (
            <Fragment key={label}>
              {i > 0 && (
                <div
                  className={`mx-1.5 h-1 flex-1 border-y-2 border-ink ${
                    i <= max ? 'bg-prussian' : 'bg-washi-200'
                  }`}
                />
              )}
              <li>
                <button
                  onClick={() => unlocked && setStep(i)}
                  disabled={!unlocked}
                  title={unlocked ? label : 'Locked — complete the previous steps first'}
                  className={`group flex items-center gap-2 rounded-brutal border-3 border-ink px-3 py-1.5 text-xs font-extrabold uppercase tracking-tight transition active:translate-x-[2px] active:translate-y-[2px] ${
                    active
                      ? 'bg-pink text-white shadow-brutal-sm'
                      : unlocked
                        ? 'bg-foam text-ink shadow-brutal-sm hover:bg-cyan active:shadow-none'
                        : 'cursor-not-allowed border-prussian-700/30 bg-washi-200 text-prussian-700/40 shadow-none'
                  }`}
                >
                  <span
                    className={`grid h-5 w-5 place-items-center rounded-sm border-2 text-[10px] font-black ${
                      active
                        ? 'border-ink bg-foam text-ink'
                        : unlocked
                          ? 'border-ink bg-prussian text-foam'
                          : 'border-ink bg-washi-200 text-prussian-700/40'
                    }`}
                  >
                    {done ? '✓' : !unlocked ? <LockIcon className="h-3.5 w-3.5" /> : i + 1}
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
