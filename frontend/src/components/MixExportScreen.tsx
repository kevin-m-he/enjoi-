import { useState } from 'react';
import { mediaUrl } from '../lib/api';
import { useStore } from '../store';
import AudioPlayer from './AudioPlayer';
import Card from './Card';
import EmptyState from './EmptyState';
import JobProgressBar from './JobProgressBar';
import Slider from './Slider';

const MIX_PRESETS: [string, string][] = [
  ['pop', 'Pop'],
  ['hiphop', 'Hip-Hop / Trap'],
  ['rnb', 'R&B'],
  ['rock', 'Rock'],
  ['acoustic', 'Acoustic'],
];

const LOUDNESS_PRESETS: [string, string][] = [
  ['streaming', 'Streaming — −14 LUFS'],
  ['loud', 'Loud — −9 LUFS'],
  ['dynamic', 'Dynamic — −16 LUFS'],
];

const selectClasses =
  'w-full rounded-brutal border-3 border-ink bg-foam px-3 py-2.5 text-sm font-semibold text-ink outline-none shadow-brutal-sm transition focus:bg-foam-50';

export default function MixExportScreen() {
  const project = useStore((s) => s.project);
  const manifest = useStore((s) => s.manifest);
  const jobs = useStore((s) => s.jobs);
  const activeJobs = useStore((s) => s.activeJobs);
  const startRender = useStore((s) => s.startRender);

  const [retune, setRetune] = useState(35);
  const [preset, setPreset] = useState('pop');
  const [loudness, setLoudness] = useState('loud');
  const [title, setTitle] = useState('');
  const [artist, setArtist] = useState('');
  const [stems, setStems] = useState(false);

  const job = activeJobs.render ? jobs[activeJobs.render] : undefined;
  const rendering = job?.status === 'queued' || job?.status === 'running';

  if (!project) {
    return <EmptyState title="No project open" hint="Start from the Upload step." />;
  }

  const hasExports = !!project.exports?.length && !rendering;
  const v = encodeURIComponent(activeJobs.render ?? 'initial');
  const wavUrl = `${mediaUrl(project.id, 'exports/song.wav')}?v=${v}`;
  const mp3Url = `${mediaUrl(project.id, 'exports/song.mp3')}?v=${v}`;
  const wavMeta = manifest?.exports.find((e) => e.format === 'wav') ?? manifest?.exports[0];

  const build = () =>
    void startRender({
      retune_speed: retune,
      preset,
      loudness_preset: loudness,
      title: title.trim() || undefined,
      artist: artist.trim() || undefined,
      include_stems: stems,
    });

  return (
    <div className="space-y-6 pt-4">
      <div className="text-center">
        <h2 className="font-display text-4xl font-black uppercase tracking-tight text-ink">
          Mix it. <span className="text-pink">Ship it.</span>
        </h2>
        <p className="mx-auto mt-2 max-w-xl text-sm font-medium text-prussian-700">
          Autotune, mix and master your song to streaming loudness — then export WAV + MP3.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <Card title="Vocal tuning">
          <Slider
            label="Autotune strength"
            value={retune}
            min={0}
            max={100}
            step={1}
            onChange={setRetune}
            format={(x) => `${x}`}
            leftLabel="Natural (corrective)"
            rightLabel="Hard tune (T-Pain)"
            disabled={rendering}
          />
        </Card>

        <Card title="Mix & loudness">
          <div className="space-y-3">
            <div>
              <label className="mb-1 block text-xs font-bold uppercase text-prussian-700">Mix preset</label>
              <select
                className={selectClasses}
                value={preset}
                onChange={(e) => setPreset(e.target.value)}
                disabled={rendering}
              >
                {MIX_PRESETS.map(([val, label]) => (
                  <option key={val} value={val}>
                    {label}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="mb-1 block text-xs font-bold uppercase text-prussian-700">Master loudness</label>
              <select
                className={selectClasses}
                value={loudness}
                onChange={(e) => setLoudness(e.target.value)}
                disabled={rendering}
              >
                {LOUDNESS_PRESETS.map(([val, label]) => (
                  <option key={val} value={val}>
                    {label}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </Card>
      </div>

      <Card title="Song details">
        <div className="grid gap-4 sm:grid-cols-2">
          <div>
            <label className="mb-1 block text-xs font-bold uppercase text-prussian-700">Title</label>
            <input
              className={selectClasses}
              placeholder="My song"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              disabled={rendering}
            />
          </div>
          <div>
            <label className="mb-1 block text-xs font-bold uppercase text-prussian-700">Artist</label>
            <input
              className={selectClasses}
              placeholder="Your artist name"
              value={artist}
              onChange={(e) => setArtist(e.target.value)}
              disabled={rendering}
            />
          </div>
        </div>
        <label className="mt-4 flex cursor-pointer items-center gap-2 text-sm font-semibold text-prussian-700">
          <input
            type="checkbox"
            checked={stems}
            onChange={(e) => setStems(e.target.checked)}
            disabled={rendering}
            className="h-4 w-4 border-2 border-ink"
          />
          Also export stems (instrumental / tuned vocal) for remixing elsewhere
        </label>
      </Card>

      <div className="flex justify-center">
        <button
          onClick={build}
          disabled={rendering}
          className="rounded-brutal border-4 border-ink bg-pink px-10 py-4 font-display text-base font-black uppercase tracking-tight text-white shadow-brutal-lg transition active:translate-x-[6px] active:translate-y-[6px] active:shadow-none disabled:cursor-not-allowed disabled:opacity-50"
        >
          {rendering ? 'Building…' : '✦ Build My Song'}
        </button>
      </div>

      <JobProgressBar
        job={job}
        onRetry={build}
        prominent
        hint="Tuning vocals, mixing buses, mastering to target loudness, encoding WAV + MP3…"
      />

      {hasExports && (
        <Card title="Your song is ready" subtitle="Generated instrumental + your voice. 100% yours.">
          <div className="space-y-3">
            <AudioPlayer src={wavUrl} title="song.wav — 44.1 kHz / 24-bit master" />
            <AudioPlayer src={mp3Url} title="song.mp3 — 320 kbps" />
          </div>

          <div className="mt-4 flex flex-wrap gap-3">
            <a
              href={wavUrl}
              download="song.wav"
              className="rounded-brutal border-3 border-ink bg-pink px-4 py-2 text-sm font-extrabold uppercase text-white shadow-brutal-sm transition active:translate-x-[3px] active:translate-y-[3px] active:shadow-none"
            >
              ↓ Download WAV
            </a>
            <a
              href={mp3Url}
              download="song.mp3"
              className="rounded-brutal border-3 border-ink bg-foam px-4 py-2 text-sm font-extrabold uppercase text-ink shadow-brutal-sm transition active:translate-x-[3px] active:translate-y-[3px] active:shadow-none"
            >
              ↓ Download MP3
            </a>
          </div>

          {manifest && (
            <div className="mt-4 rounded-brutal border-3 border-ink bg-foam p-4 text-sm">
              <p className="font-display font-extrabold uppercase text-ink">
                {manifest.title || 'Untitled'}{' '}
                {manifest.artist && (
                  <span className="font-bold text-prussian-700">— {manifest.artist}</span>
                )}
              </p>
              <div className="mt-2 grid gap-1 text-xs font-semibold text-prussian-700 sm:grid-cols-2">
                <span>
                  Loudness:{' '}
                  <span className="tabular-nums text-ink">
                    {wavMeta?.lufs !== undefined ? `${wavMeta.lufs.toFixed(1)} LUFS` : '—'}
                  </span>
                </span>
                <span>
                  True peak:{' '}
                  <span className="tabular-nums text-ink">
                    {wavMeta?.true_peak_db !== undefined
                      ? `${wavMeta.true_peak_db.toFixed(1)} dBTP`
                      : '—'}
                  </span>
                </span>
                <span>
                  BPM: <span className="tabular-nums text-ink">{manifest.bpm}</span>
                </span>
                <span>
                  Key: <span className="text-ink">{manifest.key}</span>
                </span>
              </div>
              {manifest.uniqueness_report?.summary && (
                <p className="mt-3 text-xs font-extrabold text-pink">
                  {manifest.uniqueness_report.summary}
                </p>
              )}
              <p className="mt-1 text-[11px] font-medium text-prussian-700/70">
                Sources: {manifest.sources.join(' + ')} — reference audio in output:{' '}
                {manifest.reference_audio_in_output ? 'yes' : 'never'}
              </p>
            </div>
          )}

          <p className="mt-4 text-xs font-medium text-prussian-700/80">
            Use the download buttons above to save your song (stems and manifest included).
          </p>
        </Card>
      )}
    </div>
  );
}
