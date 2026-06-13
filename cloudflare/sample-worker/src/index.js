/**
 * enjoi sample host — serves the licensed loop library from a PRIVATE R2 bucket,
 * gated by a shared secret token so the raw .wav loops are never publicly
 * downloadable (licensing). Only the enjoi backend, holding the token, can fetch
 * them to GENERATE instrumentals; end users only ever receive the finished song.
 *
 * Bindings (wrangler.toml):
 *   SAMPLES       -> R2 bucket (bucket_name = "enjoi-samples")
 *   SAMPLE_TOKEN  -> secret  (wrangler secret put SAMPLE_TOKEN)
 */
const CORS = {
  'access-control-allow-origin': '*',
  'access-control-allow-methods': 'GET, OPTIONS',
  'access-control-allow-headers': 'x-enjoi-token, range',
};

export default {
  async fetch(request, env) {
    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: CORS });
    }
    if (request.method !== 'GET') {
      return new Response('Method not allowed', { status: 405, headers: CORS });
    }

    const url = new URL(request.url);
    const token = request.headers.get('x-enjoi-token') || url.searchParams.get('t') || '';
    if (!env.SAMPLE_TOKEN || token !== env.SAMPLE_TOKEN) {
      return new Response('Forbidden', { status: 403, headers: CORS });
    }

    const key = decodeURIComponent(url.pathname.replace(/^\/+/, ''));
    if (!key || !key.toLowerCase().endsWith('.wav') || key.includes('..')) {
      return new Response('Not found', { status: 404, headers: CORS });
    }

    const obj = await env.SAMPLES.get(key);
    if (!obj) {
      return new Response('Not found', { status: 404, headers: CORS });
    }

    const headers = new Headers(CORS);
    obj.writeHttpMetadata(headers);
    headers.set('content-type', 'audio/wav');
    headers.set('etag', obj.httpEtag);
    headers.set('cache-control', 'public, max-age=31536000, immutable');
    return new Response(obj.body, { headers });
  },
};
