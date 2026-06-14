import { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import steakImg from '../assets/king-steak.png';

/**
 * Rare Easter egg: the King Steak 👑 drifts up the viewport with the background
 * bubbles, wiggling as it rises. Appears ~once every 5 minutes. Portaled to
 * <body> at z-index -1 so it sits in the bubble layer BEHIND the cards.
 */
export default function FloatingSteak() {
  const [show, setShow] = useState(false);
  const [leftPct, setLeftPct] = useState(50);

  useEffect(() => {
    const appear = () => {
      setLeftPct(8 + Math.random() * 78); // random horizontal lane each time
      setShow(true);
      window.setTimeout(() => setShow(false), 16000); // matches the float duration
    };
    const first = window.setTimeout(appear, 14000); // a teaser shortly after load
    const every = window.setInterval(appear, 5 * 60 * 1000); // then ~every 5 min
    return () => {
      window.clearTimeout(first);
      window.clearInterval(every);
    };
  }, []);

  if (!show) return null;
  return createPortal(
    <img
      key={leftPct} // restart the animation each appearance
      src={steakImg}
      alt=""
      aria-hidden="true"
      className="steak-float pointer-events-none fixed bottom-[-80px] h-16 w-16 select-none"
      style={{ left: `${leftPct}%`, zIndex: -1, imageRendering: 'pixelated' }}
    />,
    document.body,
  );
}
