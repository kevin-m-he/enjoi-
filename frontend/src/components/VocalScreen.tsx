import { useRef, useState } from 'react';
import type { DragEvent } from 'react';
import { fmtTime, roleClasses, roleLabel } from '../lib/format';
import { useStore } from '../store';
import Card from './Card';
import EmptyState from './EmptyState';
import JobProgressBar from './JobProgressBar';

const ACCEPTED = ['.wav', '.mp3'];

export default function VocalScreen() {
  const project = useStore((s) => s.project);
  const vocalAnalysis = useStore((s) => s.vocalAnalysis);
  const jobs = useStore((s) => s.jobs);
  const activeJobs = useStore((s) => s.activeJobs);
  const uploadVocal = useStore((s) => s.uploadVocal);
  const toast = useStore((s) => s.toast);
  const setStep = useStore((s) => s.setStep);
  const health = useStore((s) => s.health);

  const inputRef = useRef<HTMLInputElement | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [lastFile, setLastFile] = useState<string | null>(null);

  const job = activeJobs.vocal ? jobs[activeJobs.vocal] : undefined;
  const processing = job?.status === 'queued' || job?.status === 'running';

  const handleFile = (file: File | undefined | null) => {
    if (!file) return;
    const lower = file.name.toLowerCase();
    if (!ACCEPTED.some((ext) => lower.endsWith(ext))) {
      toast('Please upload a .wav or .mp3 file.', 'error');
      return;
    }
    setLastFile(file.name);
    void uploadVocal(file);
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    handleFile(e.dataTransfer.files?.[0]);
  };

  if (!project) {
    return <EmptyState title="No project open" hint="Start from the Search step." />;
  }

  return (
    <div className="space-y-6 pt-4">
      <div className="text-center">
        <h2 className="font-display text-4xl font-black uppercase tracking-tight text-ink">
          Add your <span className="text-pink">one-take vocal</span>
        </h2>
        <p className="mx-auto mt-2 max-w-xl text-sm font-medium text-prussian-700">
          One continuous take, dry vocal (no reverb or effects), .wav or .mp3. We’ll transcribe it,
          find your best section and chop it for the arrangement.
        </p>
      </div>

      {/* drop zone */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => !processing && inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') inputRef.current?.click();
        }}
        className={`cursor-pointer rounded-brutal border-4 border-dashed px-6 py-14 text-center shadow-brutal transition ${
          dragOver
            ? 'border-pink bg-pink/10'
            : 'border-ink bg-foam-50 hover:bg-cyan/20'
        } ${processing ? 'pointer-events-none opacity-50' : ''}`}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".wav,.mp3,audio/wav,audio/x-wav,audio/mpeg"
          className="hidden"
          onChange={(e) => {
            handleFile(e.target.files?.[0]);
            e.target.value = '';
          }}
        />
        <div className="mx-auto mb-4 grid h-16 w-16 place-items-center rounded-brutal border-4 border-ink bg-cyan text-3xl shadow-brutal-sm">
          🎤
        </div>
        <p className="font-display text-lg font-extrabold uppercase tracking-tight text-ink">
          {dragOver ? 'Drop it!' : 'Drag & drop your vocal take here'}
        </p>
        <p className="mt-1 text-sm font-semibold text-prussian-700">
          or <span className="font-extrabold text-pink">browse</span> for a .wav / .mp3 file
        </p>
        {lastFile && (
          <p className="mt-3 text-xs font-bold text-prussian-700/70">Last upload: {lastFile}</p>
        )}
      </div>

      <JobProgressBar
        job={job}
        onRetry={() => inputRef.current?.click()}
        prominent
        hint="Cleaning, transcribing lyrics, splitting into phrases and scoring sections…"
      />

      {vocalAnalysis && !processing && (
        <>
          <div className="grid grid-cols-3 gap-4">
            {[
              { label: 'Take length', value: fmtTime(vocalAnalysis.duration_sec) },
              { label: 'Phrases detected', value: String(vocalAnalysis.phrases.length) },
              { label: 'Sections', value: String(vocalAnalysis.sections.length) },
            ].map((t) => (
              <div
                key={t.label}
                className="rounded-brutal border-4 border-ink bg-foam-50 p-4 shadow-brutal-sm"
              >
                <p className="text-xs font-extrabold uppercase tracking-wide text-prussian-700/70">
                  {t.label}
                </p>
                <p className="mt-1 font-display text-2xl font-black text-ink">{t.value}</p>
              </div>
            ))}
          </div>

          <Card title="Lyrics transcript" subtitle="Word-level transcription (Whisper, local)">
            {vocalAnalysis.lyrics.trim() ? (
              <p className="whitespace-pre-wrap rounded-sm border-3 border-ink bg-washi-200 p-4 text-sm font-medium leading-relaxed text-ink">
                {vocalAnalysis.lyrics}
              </p>
            ) : (
              <p className="rounded-sm border-3 border-ink bg-washi-200 p-4 text-sm font-medium text-prussian-700">
                No transcription available
                {health && !health.capabilities.whisper
                  ? ' — Whisper is not installed, so the take was segmented by energy only.'
                  : '.'}
              </p>
            )}
          </Card>

          <Card
            title="Detected sections"
            subtitle="Impact Score picks your chorus — the bridge is the chorus’s nearest neighbour"
          >
            <ul className="space-y-3">
              {[...vocalAnalysis.sections]
                .sort((a, b) => a.start - b.start)
                .map((sec) => (
                  <li
                    key={sec.id}
                    className="rounded-brutal border-3 border-ink bg-foam p-4 shadow-brutal-sm"
                  >
                    <div className="flex items-center justify-between gap-3">
                      <div className="flex min-w-0 items-center gap-3">
                        <span
                          className={`shrink-0 rounded-sm border-2 px-2.5 py-0.5 text-xs font-extrabold uppercase ${roleClasses(sec.role)}`}
                        >
                          {roleLabel(sec.role)}
                        </span>
                        <span className="text-xs font-bold tabular-nums text-prussian-700/70">
                          {fmtTime(sec.start)}–{fmtTime(sec.end)}
                        </span>
                      </div>
                      <span className="shrink-0 text-xs font-bold tabular-nums text-prussian-700">
                        Impact {Math.round(sec.impact_score * 100)}
                      </span>
                    </div>
                    <div className="mt-2 h-3 overflow-hidden rounded-sm border-2 border-ink bg-washi-200">
                      <div
                        className="h-full bg-pink"
                        style={{
                          width: `${Math.min(Math.max(sec.impact_score, 0), 1) * 100}%`,
                        }}
                      />
                    </div>
                    {sec.text && (
                      <p className="mt-2 line-clamp-2 text-sm font-medium text-prussian-700">
                        {sec.text}
                      </p>
                    )}
                  </li>
                ))}
            </ul>
          </Card>

          <div className="flex justify-end pb-6">
            <button
              onClick={() => setStep(4)}
              className="rounded-brutal border-4 border-ink bg-pink px-6 py-3 font-display text-sm font-black uppercase tracking-tight text-white shadow-brutal transition active:translate-x-[6px] active:translate-y-[6px] active:shadow-none"
            >
              Continue ▶ Arrange
            </button>
          </div>
        </>
      )}
    </div>
  );
}
