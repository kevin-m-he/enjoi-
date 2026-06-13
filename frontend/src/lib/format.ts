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

/** Structure-label palette shared by analysis strip + arrange timeline. */
export const SECTION_COLORS: Record<string, string> = {
  intro: '#818cf8',
  verse: '#38bdf8',
  prechorus: '#a78bfa',
  chorus: '#ec4899',
  bridge: '#f59e0b',
  outro: '#64748b',
  inst: '#34d399',
};

export function sectionColor(label: string): string {
  return SECTION_COLORS[label.toLowerCase()] ?? '#71717a';
}

export function roleLabel(role: string): string {
  if (role === 'chorus') return 'Chorus ★';
  return role.charAt(0).toUpperCase() + role.slice(1);
}

export function roleClasses(role: string): string {
  switch (role) {
    case 'chorus':
      return 'bg-pink-500/20 text-pink-300 border-pink-500/40';
    case 'bridge':
      return 'bg-amber-500/20 text-amber-300 border-amber-500/40';
    default:
      return 'bg-sky-500/20 text-sky-300 border-sky-500/40';
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
