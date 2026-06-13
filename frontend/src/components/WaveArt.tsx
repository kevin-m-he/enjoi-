// Inline SVG motifs for the Great-Wave-off-Kanagawa identity.
// Used as the hero, section dividers, loading indicators and card accents.

/**
 * The iconic clawing Great Wave, recolored to fuse the Kanagawa Prussian-blue
 * with the vibrant cyan highlights and hot-pink foam accents. Pure inline SVG.
 */
export function HeroWave({ className = '' }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 800 360"
      className={className}
      preserveAspectRatio="xMidYMid slice"
      role="img"
      aria-label="The Great Wave"
    >
      {/* sky / washi */}
      <rect x="0" y="0" width="800" height="360" fill="#F4ECD8" />

      {/* Mount Fuji silhouette */}
      <path d="M250 300 L360 150 L400 185 L420 150 L530 300 Z" fill="#1B3A5B" />
      <path d="M345 165 L360 150 L400 185 L420 150 L438 168 L408 168 L388 156 L368 172 Z" fill="#FBF7EC" />

      {/* deep wave body */}
      <path
        d="M0 360 L0 200 C 120 120, 220 300, 360 240 C 470 192, 520 90, 660 120 C 740 137, 760 230, 800 210 L800 360 Z"
        fill="#0B2C4D"
        stroke="#0B0B0C"
        strokeWidth="4"
      />
      {/* cyan-highlighted inner wave */}
      <path
        d="M0 360 L0 250 C 110 190, 230 330, 360 280 C 470 238, 540 160, 660 185 C 730 200, 760 260, 800 250 L800 360 Z"
        fill="#1B3A5B"
        stroke="#00E5FF"
        strokeWidth="3"
      />
      {/* the clawing crest */}
      <path
        d="M620 150 C 600 70, 660 40, 700 60 C 660 50, 640 95, 660 120 C 690 100, 720 118, 715 150 C 700 128, 678 132, 672 152 C 700 142, 712 168, 700 188 C 690 165, 668 165, 660 188 C 678 175, 690 200, 678 218 C 666 196, 646 196, 642 220 C 654 198, 636 176, 612 184 C 640 176, 636 150, 620 150 Z"
        fill="#FBF7EC"
        stroke="#0B0B0C"
        strokeWidth="3"
      />
      {/* pink foam claw accents */}
      <g fill="#FF2D95">
        <circle cx="694" cy="64" r="7" />
        <circle cx="712" cy="150" r="6" />
        <circle cx="676" cy="188" r="6" />
        <circle cx="640" cy="220" r="6" />
      </g>
      {/* scattered foam droplets */}
      <g fill="#FBF7EC" stroke="#0B0B0C" strokeWidth="2">
        <circle cx="540" cy="120" r="6" />
        <circle cx="500" cy="150" r="5" />
        <circle cx="470" cy="115" r="4" />
        <circle cx="430" cy="170" r="5" />
      </g>
    </svg>
  );
}

/** Small foam-claw wave mark for headers / wordmark backdrop. */
export function FoamMark({ className = '' }: { className?: string }) {
  return (
    <svg viewBox="0 0 64 40" className={className} role="img" aria-label="wave">
      <path
        d="M2 34 C 14 10, 26 30, 36 18 C 44 8, 54 12, 62 4"
        fill="none"
        stroke="#00E5FF"
        strokeWidth="4"
        strokeLinecap="round"
      />
      <path
        d="M2 30 C 12 38, 22 22, 34 30 C 46 38, 54 24, 62 30"
        fill="none"
        stroke="#FF2D95"
        strokeWidth="4"
        strokeLinecap="round"
      />
    </svg>
  );
}

/**
 * A spinning foam-and-wave loading mark used while jobs run.
 */
export function WaveSpinner({ className = '' }: { className?: string }) {
  return (
    <svg viewBox="0 0 40 40" className={`animate-spin ${className}`} role="img" aria-label="working">
      <circle cx="20" cy="20" r="15" fill="none" stroke="#0B0B0C" strokeWidth="4" opacity="0.15" />
      <path
        d="M20 5 A15 15 0 0 1 35 20"
        fill="none"
        stroke="#00E5FF"
        strokeWidth="4"
        strokeLinecap="round"
      />
      <path
        d="M20 35 A15 15 0 0 1 5 20"
        fill="none"
        stroke="#FF2D95"
        strokeWidth="4"
        strokeLinecap="round"
      />
    </svg>
  );
}
