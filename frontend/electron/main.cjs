// enjoi 享受 — Electron main process (CommonJS, no bundling needed)
const { app, BrowserWindow } = require('electron');
const path = require('node:path');
const fs = require('node:fs');
const { spawn } = require('node:child_process');

let mainWindow = null;
let backendProc = null;

/**
 * Bring the FastAPI backend up if it isn't already. Preferred path: spawn the
 * project venv's python directly (no PowerShell — shortcut-safe on systems
 * that restrict script hosts). Falls back to scripts\run_backend.ps1. If the
 * port is already in use the spawn dies quietly and the renderer connects to
 * the existing instance. Non-fatal either way: the renderer shows a friendly
 * "backend not running" screen with auto-retry.
 */
function ensureBackend() {
  try {
    const roots = [
      path.join(__dirname, '..', '..'),               // repo layout (frontend/electron/)
      path.join(process.resourcesPath || '', '..'),    // packaged layout
      process.resourcesPath || '',
    ];
    for (const root of roots) {
      const venvPython = path.join(root, 'backend', '.venv', 'Scripts', 'python.exe');
      const backendDir = path.join(root, 'backend');
      if (fs.existsSync(venvPython) && fs.existsSync(path.join(backendDir, 'main.py'))) {
        backendProc = spawn(
          venvPython,
          ['-m', 'uvicorn', 'main:app', '--host', '127.0.0.1', '--port', '8723'],
          { cwd: backendDir, stdio: 'ignore', windowsHide: true }
        );
        backendProc.on('error', () => {
          backendProc = null;
        });
        return;
      }
    }
    // Fallback: PowerShell launcher (packaged installs without a venv beside them)
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
  ensureBackend();
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
