import { useEffect, useState } from 'react';
import AnalysisScreen from './components/AnalysisScreen';
import ArrangeScreen from './components/ArrangeScreen';
import Header from './components/Header';
import MixExportScreen from './components/MixExportScreen';
import SearchScreen from './components/SearchScreen';
import SimilarityScreen from './components/SimilarityScreen';
import Stepper from './components/Stepper';
import Toasts from './components/Toasts';
import VocalScreen from './components/VocalScreen';
import { HeroWave, FoamMark } from './components/WaveArt';
import { API_BASE } from './lib/api';
import { useJobSocket } from './lib/ws';
import { useStore } from './store';

function BackendDown() {
  const checkHealth = useStore((s) => s.checkHealth);
  const [checking, setChecking] = useState(false);

  const retry = async () => {
    setChecking(true);
    await checkHealth();
    setChecking(false);
  };

  return (
    <div className="flex min-h-screen flex-col items-center justify-center px-6 text-center">
      <div className="w-full max-w-md overflow-hidden rounded-brutal border-4 border-ink shadow-brutal-lg">
        <HeroWave className="h-40 w-full" />
        <div className="border-t-4 border-ink bg-foam-50 p-6">
          <h1 className="font-display text-3xl font-black uppercase tracking-tight text-ink">
            <span className="text-pink">Backend</span> not running
          </h1>
          <p className="mt-3 text-sm font-medium text-prussian-900">
            The local enjoi engine isn’t answering on{' '}
            <code className="rounded-sm border-2 border-ink bg-foam px-1.5 py-0.5 text-xs font-bold text-ink">
              {API_BASE.replace(/^https?:\/\//, '')}
            </code>
            . Start it from the project root with{' '}
            <code className="rounded-sm border-2 border-ink bg-pink px-1.5 py-0.5 text-xs font-bold text-white">
              scripts\dev.ps1
            </code>{' '}
            and we’ll reconnect automatically.
          </p>
          <button
            onClick={() => void retry()}
            disabled={checking}
            className="mt-6 rounded-brutal border-4 border-ink bg-pink px-6 py-3 text-sm font-extrabold uppercase tracking-tight text-white shadow-brutal transition active:translate-x-[6px] active:translate-y-[6px] active:shadow-none disabled:opacity-50"
          >
            {checking ? 'Checking…' : 'Retry now'}
          </button>
          <p className="mt-3 text-xs font-semibold text-prussian-900">
            Retrying automatically every few seconds…
          </p>
        </div>
      </div>
    </div>
  );
}

function Splash() {
  return (
    <div className="flex min-h-screen items-center justify-center">
      <div className="text-center">
        <FoamMark className="mx-auto mb-4 h-12 w-20 animate-foam-bob" />
        <h1 className="font-display text-5xl font-black tracking-tight">
          <span className="text-wave">enjoi</span> <span className="text-ink">享受</span>
        </h1>
        <p className="mt-3 animate-pulse text-sm font-bold uppercase tracking-tight text-prussian-700">
          Connecting to the engine…
        </p>
      </div>
    </div>
  );
}

export default function App() {
  const backendUp = useStore((s) => s.backendUp);
  const step = useStore((s) => s.step);
  const checkHealth = useStore((s) => s.checkHealth);

  useJobSocket();

  useEffect(() => {
    void checkHealth();
    const t = window.setInterval(() => void checkHealth(), 4000);
    return () => window.clearInterval(t);
  }, [checkHealth]);

  if (backendUp === null) {
    return (
      <>
        <Splash />
        <Toasts />
      </>
    );
  }

  if (backendUp === false) {
    return (
      <>
        <BackendDown />
        <Toasts />
      </>
    );
  }

  return (
    <div className="flex min-h-screen flex-col">
      <Header />
      <Stepper />
      <main className="mx-auto w-full max-w-6xl flex-1 px-6 pb-16">
        {step === 0 && <SearchScreen />}
        {step === 1 && <AnalysisScreen />}
        {step === 2 && <SimilarityScreen />}
        {step === 3 && <VocalScreen />}
        {step === 4 && <ArrangeScreen />}
        {step === 5 && <MixExportScreen />}
      </main>
      <VerseFooter />
      <Toasts />
    </div>
  );
}

/** Tiny dark-blue scripture pinned at the bottom-right of the page (in flow, so
 *  it doesn't float over the content as you scroll), with balanced line lengths. */
function VerseFooter() {
  return (
    <footer className="mt-auto flex w-full justify-end px-4 pb-2 pt-6">
      <p className="max-w-[300px] text-balance text-right text-[8px] leading-snug text-prussian-700/80">
        Isaiah 43:2, When you pass through the waters, I will be with you; and when you pass through
        the rivers, they will not sweep over you. When you walk through the fire, you will not be
        burned; the flames will not set you ablaze.
      </p>
    </footer>
  );
}
