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
      <div className="mb-6 grid h-20 w-20 place-items-center rounded-full border border-white/10 bg-white/5 text-4xl">
        🔌
      </div>
      <h1 className="text-3xl font-extrabold tracking-tight">
        <span className="text-grad">Backend</span> not running
      </h1>
      <p className="mt-3 max-w-md text-sm text-zinc-400">
        The local enjoi engine isn’t answering on{' '}
        <code className="rounded bg-black/40 px-1.5 py-0.5 text-xs text-zinc-300">
          127.0.0.1:8723
        </code>
        . Start it from the project root with{' '}
        <code className="rounded bg-black/40 px-1.5 py-0.5 text-xs text-amber-300">
          scripts\dev.ps1
        </code>{' '}
        and we’ll reconnect automatically.
      </p>
      <button
        onClick={() => void retry()}
        disabled={checking}
        className="mt-6 rounded-xl bg-gradient-to-r from-pink-500 to-amber-500 px-6 py-3 text-sm font-semibold text-white shadow-glow transition hover:opacity-90 disabled:opacity-50"
      >
        {checking ? 'Checking…' : 'Retry now'}
      </button>
      <p className="mt-3 text-xs text-zinc-600">Retrying automatically every few seconds…</p>
    </div>
  );
}

function Splash() {
  return (
    <div className="flex min-h-screen items-center justify-center">
      <div className="text-center">
        <h1 className="text-4xl font-black tracking-tight">
          <span className="text-grad">enjoi</span> <span className="text-zinc-200">享受</span>
        </h1>
        <p className="mt-3 animate-pulse text-sm text-zinc-500">Connecting to the engine…</p>
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
      <Toasts />
    </div>
  );
}
