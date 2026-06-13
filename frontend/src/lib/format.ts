// Small shared formatting + UI helpers.

/** Seconds → "m:ss". */
export function fmtTime(sec: number): string {
  if (!Number.isFinite(sec) || sec < 0) return '0:00';
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

/** 1234567 → "1.2M". */
export function abbrev(n: number): string {
  if (!Number.isFinite(n)) return '0';
  const fmt = (v: number, suffix: string) => `${v.toFixed(1).replace(/\.0$/, '')}${suffix}`;
  if (n >= 1e9) return fmt(n / 1e9, 'B');
  if (n >= 1e6) return fmt(n / 1e6, 'M');
  if (n >= 1e3) return fmt(n / 1e3, 'K');
  return String(Math.round(n));
}

export function clamp(v: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, v));
}

/** Structure-label palette shared by analysis strip + arrange timeline.
 *  Kanagawa + cyan/pink: Prussian blues for structure, cyan/pink for the hooks. */
export const SECTION_COLORS: Record<string, string> = {
  intro: '#274C73',
  verse: '#1B3A5B',
  prechorus: '#00E5FF',
  chorus: '#FF2D95',
  bridge: '#FF1FA2',
  outro: '#0B2C4D',
  inst: '#06B6CC',
};

export function sectionColor(label: string): string {
  return SECTION_COLORS[label.toLowerCase()] ?? '#274C73';
}

export function roleLabel(role: string): string {
  if (role === 'chorus') return 'Chorus ★';
  return role.charAt(0).toUpperCase() + role.slice(1);
}

export function roleClasses(role: string): string {
  switch (role) {
    case 'chorus':
      return 'bg-pink text-white border-ink';
    case 'bridge':
      return 'bg-pink-600 text-white border-ink';
    default:
      return 'bg-cyan text-ink border-ink';
  }
}

export function errMsg(e: unknown): string {
  if (e instanceof Error) return e.message;
  return String(e);
}

/** "A" + "minor" → "A minor". */
export function keyName(key: { tonic: string; mode: string } | null | undefined): string {
  if (!key) return '—';
  return `${key.tonic} ${key.mode}`;
}
