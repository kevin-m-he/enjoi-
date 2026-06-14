import { useEffect, useState } from 'react';

/**
 * Rare Easter egg: a single crowned pixel-steak (King Steaks 👑) drifts up with
 * the background bubbles, wiggling as it rises. Appears ~once every 5 minutes.
 */
function SteakArt({ className = '' }: { className?: string }) {
  return (
    <svg viewBox="0 0 44 50" className={className} shapeRendering="geometricPrecision" aria-hidden="true">
      {/* crown */}
      <path
        d="M11 12 L11 6 L16 9 L22 3 L28 9 L33 6 L33 12 Z"
        fill="#FFC83D"
        stroke="#0B0B0C"
        strokeWidth="2"
        strokeLinejoin="round"
      />
      <rect x="11" y="11" width="22" height="4" rx="1" fill="#FFC83D" stroke="#0B0B0C" strokeWidth="2" />
      <circle cx="22" cy="13" r="1.3" fill="#FF2D95" />
      <circle cx="16" cy="13" r="1" fill="#00E5FF" />
      <circle cx="28" cy="13" r="1" fill="#00E5FF" />
      {/* steak body (T-bone) */}
      <path
        d="M9 27 C5 31 5 38 10 42 C14 46 20 47 24 45 C27 47 33 46 37 42 C42 37 41 30 36 27 C39 22 35 17 28 18 C25 15 19 15 16 18 C10 17 6 22 9 27 Z"
        fill="#A23A2B"
        stroke="#0B0B0C"
        strokeWidth="2.4"
        strokeLinejoin="round"
      />
      {/* fat rim highlight */}
      <path
        d="M11 25 C8 29 9 36 13 40"
        fill="none"
        stroke="#D98C72"
        strokeWidth="2"
        strokeLinecap="round"
        opacity="0.8"
      />
      {/* T-bone */}
      <path
        d="M23 22 L23 40 M23 31 C20 31 18 33 18 36 M23 31 C26 31 28 33 28 36"
        fill="none"
        stroke="#F0E6CC"
        strokeWidth="3"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {/* marbling specks */}
      <circle cx="15" cy="33" r="1.1" fill="#C9614E" />
      <circle cx="31" cy="31" r="1.1" fill="#C9614E" />
      <circle cx="30" cy="38" r="1" fill="#C9614E" />
    </svg>
  );
}

export default function FloatingSteak() {
  const [show, setShow] = useState(false);
  const [leftPct, setLeftPct] = useState(50);

  useEffect(() => {
    const appear = () => {
      setLeftPct(8 + Math.random() * 78); // random horizontal lane each time
      setShow(true);
      window.setTimeout(() => setShow(false), 15000); // matches the float duration
    };
    const first = window.setTimeout(appear, 14000); // a teaser shortly after load
    const every = window.setInterval(appear, 5 * 60 * 1000); // then ~every 5 min
    return () => {
      window.clearTimeout(first);
      window.clearInterval(every);
    };
  }, []);

  if (!show) return null;
  return (
    <div
      key={leftPct} // restart the animation each appearance
      className="steak-float pointer-events-none fixed bottom-[-140px] z-20 h-24 w-24 drop-shadow"
      style={{ left: `${leftPct}%` }}
      aria-hidden="true"
    >
      <SteakArt className="h-full w-full" />
    </div>
  );
}
