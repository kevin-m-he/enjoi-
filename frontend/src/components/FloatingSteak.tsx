import { useEffect, useState } from 'react';
import steakImg from '../assets/king-steak.png';

/**
 * Rare Easter egg: the King Steak 👑 drifts up the viewport with the background
 * bubbles, wiggling as it rises. Appears ~once every 5 minutes. Uses the provided
 * transparent PNG asset, just resized + animated.
 */
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
    <img
      key={leftPct} // restart the animation each appearance
      src={steakImg}
      alt=""
      aria-hidden="true"
      className="steak-float pointer-events-none fixed bottom-[-32px] h-5 w-5 select-none"
      style={{ left: `${leftPct}%`, zIndex: -1, imageRendering: 'pixelated' }}
    />
  );
}
