// enjoi 享受 — Electron main process (CommonJS, no bundling needed)
const { app, BrowserWindow } = require('electron');
const path = require('node:path');
const fs = require('node:fs');
const { spawn } = require('node:child_process');

let mainWindow = null;
let backendProc = null;

/**
 * In PACKAGED mode the desktop shell is responsible for bringing the FastAPI
 * backend up (per docs/API_CONTRACT.md "Frontend contract"). In dev the
 * backend is assumed to already be running (scripts\dev.ps1).
 * Non-fatal: the renderer shows a friendly "backend not running" screen.
 */
function spawnBackendIfPackaged() {
  if (!app.isPackaged) return;
  try {
    const candidates = [
      path.join(process.resourcesPath || '', 'scripts', 'run_backend.ps1'),
      path.join(app.getAppPath(), '..', 'scripts', 'run_backend.ps1'),
      path.join(__dirname, '..', '..', 'scripts', 'run_backend.ps1'),
    ];
    const script = candidates.find((p) => {
      try {
        return fs.existsSync(p);
      } catch {
        return false;
      }
    });
    if (!script) return;
    backendProc = spawn(
      'powershell.exe',
      ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', script],
      { stdio: 'ignore', windowsHide: true }
    );
    backendProc.on('error', (err) => {
      console.warn('[enjoi] backend spawn failed (non-fatal):', err.message);
      backendProc = null;
    });
  } catch (err) {
    console.warn('[enjoi] backend spawn failed (non-fatal):', err && err.message);
  }
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    minWidth: 1080,
    minHeight: 720,
    backgroundColor: '#0c0a14',
    title: 'enjoi 享受',
    autoHideMenuBar: true,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      preload: path.join(__dirname, 'preload.cjs'),
    },
  });

  // Dev server only when explicitly requested (npm run dev sets
  // VITE_DEV_SERVER) or when there is no production build to load.
  const distIndex = path.join(__dirname, '..', 'dist', 'index.html');
  const useDevServer = !!process.env.VITE_DEV_SERVER || !fs.existsSync(distIndex);
  if (useDevServer) {
    mainWindow.loadURL('http://localhost:5173');
  } else {
    mainWindow.loadFile(distIndex);
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

app.whenReady().then(() => {
  spawnBackendIfPackaged();
  createWindow();
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  app.quit();
});

app.on('quit', () => {
  if (backendProc && !backendProc.killed) {
    try {
      backendProc.kill();
    } catch {
      /* already gone */
    }
  }
});
