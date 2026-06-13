// TypeScript mirrors of docs/API_CONTRACT.md schemas. Keep in sync with the contract.

// ---- /api/health ----
export interface Capabilities {
  ffmpeg: boolean;
  gpu: boolean;
  musicgen: boolean;
  whisper: boolean;
  demucs: boolean;
  pedalboard: boolean;
  rubberband: boolean;
  madmom: boolean;
  essentia: boolean;
  crepe: boolean;
}

export interface Health {
  status: string;
  version: string;
  capabilities: Capabilities;
}

// ---- /api/search ----
export interface SearchResult {
  video_id: string;
  title: string;
  channel: string;
  duration_sec: number;
  thumbnail_url: string;
  view_count: number;
  url: string;
}

// ---- Jobs ----
export type JobType = 'reference' | 'generate' | 'vocal' | 'rearrange' | 'render';
export type JobStatus = 'queued' | 'running' | 'done' | 'error';

export interface Job {
  id: string;
  type: JobType;
  project_id: string;
  status: JobStatus;
  progress: number; // 0..1
  message: string;
  result?: unknown;
  error?: string | null;
}

// ---- ProjectState (project.json) ----
export interface ProjectReference {
  url: string;
  video_id: string;
  title: string;
  channel: string;
  duration_sec: number;
  thumbnail_url: string;
  analyzed: boolean;
}

export interface ProjectInstrumental {
  file: string;
  grid: string;
  engine: string; // "musicgen-small" | "musicgen-large" | "procedural" | ...
  uniqueness_passed: boolean;
}

export interface ProjectVocal {
  file: string;
  analysis: string;
  lyrics_available: boolean;
}

export interface ProjectExport {
  file: string;
  format: string;
}

export interface ProjectState {
  id: string;
  name: string;
  created_at: string;
  reference: ProjectReference | null;
  similarity: number | null;
  instrumental: ProjectInstrumental | null;
  vocal: ProjectVocal | null;
  arrangement_ready: boolean | null;
  exports: ProjectExport[] | null;
}

// ---- reference_profile.json ----
export interface KeyInfo {
  tonic: string;
  mode: string;
  confidence?: number;
}

export interface StructureSection {
  label: string; // intro|verse|prechorus|chorus|bridge|outro|inst
  start: number;
  end: number;
  bars: number;
}

export interface ReferenceProfile {
  source: {
    title: string;
    channel: string;
    url: string;
    video_id: string;
    duration_sec: number;
  };
  duration_sec: number;
  bpm: number;
  beat_times: number[];
  downbeats: number[];
  time_signature: string;
  key: KeyInfo;
  structure: StructureSection[];
  energy_curve: { per_bar_rms: number[]; per_bar_flux: number[] };
  instrumentation: Record<string, number>; // drums/bass/melodic/vocals 0..1
  groove: { swing: number; pattern_class: string; onset_histogram: number[] };
  genre_tags: string[];
  mood_tags: string[];
  ref_audio?: string;
  fingerprints?: {
    melody_interval_ngrams: string[];
    chord_sequence: string[];
    chroma_downbeat: number[][];
    fp_hashes: number[];
  };
}

// ---- instrumental_grid.json ----
export interface GridSection {
  label: string;
  start: number;
  end: number;
  bars: number;
}

export interface InstrumentalGrid {
  bpm: number;
  time_signature: string;
  key: KeyInfo & { scale_midi?: number[] };
  beat_times: number[];
  downbeats: number[];
  sections: GridSection[];
  duration_sec: number;
  engine: string;
}

// ---- vocal_analysis.json ----
export interface Word {
  w: string;
  start: number;
  end: number;
}

export interface PhraseFeatures {
  rms: number;
  f0_mean_hz: number;
  f0_range_semitones: number;
  pitch_height: number;
  vibrato: number;
  brightness: number;
}

export interface Phrase {
  id: number;
  start: number;
  end: number;
  text: string;
  features: PhraseFeatures;
}

export type SectionRole = 'chorus' | 'verse' | 'bridge';

export interface VocalSection {
  id: number;
  phrase_ids: number[];
  start: number;
  end: number;
  text: string;
  impact_score: number;
  scores: Record<string, number>; // energy/pitch_range/pitch_height/vibrato/repetition/brightness/hookiness
  role: SectionRole;
}

export interface VocalAnalysis {
  file: string;
  duration_sec: number;
  lyrics: string;
  words: Word[];
  phrases: Phrase[];
  sections: VocalSection[];
  weights: Record<string, number>;
}

// ---- arrangement.json ----
export interface Placement {
  id: number;
  role: string;
  slot_label: string;
  slot_index: number;
  section_id: number;
  source_start: number;
  source_end: number;
  target_start: number;
  stretch: number;
  gain_db: number;
  chop_file: string;
  tuned_file: string | null;
  enabled?: boolean; // frontend toggle; absent = enabled
}

export interface ArrangementSlot {
  label: string;
  index: number;
  start: number;
  end: number;
  filled: boolean;
}

export interface Arrangement {
  placements: Placement[];
  slots: ArrangementSlot[];
  summary: { chorus_section_id: number; bridge_section_id: number };
}

// ---- uniqueness_report.json ----
export interface UniquenessCheck {
  value: number;
  threshold: number;
  passed: boolean;
  exempt_loops?: boolean;
}

export interface UniquenessReport {
  passed: boolean;
  attempts: number;
  effective_similarity: number;
  checks: Record<string, UniquenessCheck>;
  summary: string;
}

// ---- song_manifest.json ----
export interface ManifestExport {
  file: string;
  format: string;
  lufs?: number;
  true_peak_db?: number;
}

export interface SongManifest {
  title: string;
  artist: string;
  bpm: number;
  key: string;
  created_at: string;
  sources: string[];
  reference_audio_in_output: boolean;
  uniqueness_report?: UniquenessReport;
  exports: ManifestExport[];
}

// ---- request bodies ----
export interface RenderOptions {
  retune_speed: number; // 0..100
  preset: string; // pop | hiphop | rnb | rock | acoustic
  loudness_preset: string; // streaming | loud | dynamic
  title?: string;
  artist?: string;
  include_stems?: boolean;
}

// Impact Score weights (spec 4.6) — 7 user-tunable weights.
export const SCORE_WEIGHT_KEYS = [
  'energy',
  'pitch_range',
  'pitch_height',
  'vibrato',
  'repetition',
  'brightness',
  'hookiness',
] as const;

export const DEFAULT_WEIGHTS: Record<string, number> = {
  energy: 0.2,
  pitch_range: 0.15,
  pitch_height: 0.1,
  vibrato: 0.15,
  repetition: 0.2,
  brightness: 0.1,
  hookiness: 0.1,
};
