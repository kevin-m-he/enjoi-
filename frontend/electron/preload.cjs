// enjoi 享受 — preload (placeholder: exposes nothing operational yet)
const { contextBridge } = require('electron');

contextBridge.exposeInMainWorld('appInfo', {
  name: 'enjoi-frontend',
  version: '0.1.0',
  electron: process.versions.electron,
});
