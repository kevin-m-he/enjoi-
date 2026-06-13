import { useEffect } from 'react';
import { api, WS_URL } from './api';
import { useStore } from '../store';
import type { Job } from './types';

/**
 * Connects to ws://127.0.0.1:8723/ws and feeds `{type:"job", job}` pushes into
 * the store. Auto-reconnects with a 1.5 s backoff. While the socket is down,
 * falls back to polling GET /api/jobs/{id} every 1.5 s for active jobs.
 */
export function useJobSocket(): void {
  useEffect(() => {
    let ws: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let disposed = false;

    const scheduleReconnect = () => {
      if (disposed || reconnectTimer !== null) return;
      reconnectTimer = setTimeout(() => {
        reconnectTimer = null;
        connect();
      }, 1500);
    };

    const connect = () => {
      if (disposed) return;
      try {
        ws = new WebSocket(WS_URL);
      } catch {
        scheduleReconnect();
        return;
      }
      ws.onopen = () => useStore.getState().setWsConnected(true);
      ws.onmessage = (ev) => {
        try {
          const data = JSON.parse(String(ev.data)) as { type?: string; job?: Job };
          if (data.type === 'job' && data.job) useStore.getState().upsertJob(data.job);
        } catch {
          /* ignore malformed frames */
        }
      };
      ws.onclose = () => {
        useStore.getState().setWsConnected(false);
        scheduleReconnect();
      };
      ws.onerror = () => {
        try {
          ws?.close();
        } catch {
          /* noop */
        }
      };
    };

    connect();

    const poll = setInterval(() => {
      const s = useStore.getState();
      if (s.wsConnected || s.backendUp === false) return;
      for (const id of Object.values(s.activeJobs)) {
        if (!id) continue;
        const known = s.jobs[id];
        if (known && known.status !== 'queued' && known.status !== 'running') continue;
        api
          .getJob(id)
          .then((job) => useStore.getState().upsertJob(job))
          .catch(() => undefined);
      }
    }, 1500);

    return () => {
      disposed = true;
      if (reconnectTimer !== null) clearTimeout(reconnectTimer);
      clearInterval(poll);
      try {
        ws?.close();
      } catch {
        /* noop */
      }
    };
  }, []);
}
