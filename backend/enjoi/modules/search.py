"""YouTube search (spec §4.1): yt-dlp flat ``ytsearchN:`` extraction.

No API key required. Results are filtered to non-live entries no longer than
``config.MAX_REFERENCE_DURATION_SEC``.
"""
from __future__ import annotations

from ..core import config
from ..core.errors import PipelineError


def search_youtube(query: str, limit: int = 12) -> list[dict]:
    """Search YouTube and return up to ``limit`` SearchResult dicts.

    Each result: ``{video_id, title, channel, duration_sec, thumbnail_url,
    view_count, url}``. Live streams and videos longer than the reference
    duration cap are filtered out.
    """
    query = (query or "").strip()
    if not query:
        return []
    limit = max(1, min(int(limit), 30))

    try:
        import yt_dlp  # heavy import kept inside the function (contract rule 1)
    except Exception as exc:  # pragma: no cover - environment problem
        raise PipelineError("YouTube search is unavailable — yt-dlp is not installed.") from exc

    # Over-fetch so the duration/live filters can still fill `limit` slots.
    fetch = min(limit * 2, 40)
    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": "in_playlist",
        "noplaylist": True,
        "socket_timeout": 15,
        "retries": 2,
        "cachedir": False,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"ytsearch{fetch}:{query}", download=False)
    except Exception as exc:
        raise PipelineError(
            "YouTube search failed — check your internet connection and try again."
        ) from exc

    results: list[dict] = []
    for entry in (info or {}).get("entries") or []:
        if not entry:
            continue
        video_id = entry.get("id")
        duration = entry.get("duration")
        if not video_id or duration is None:
            continue  # live streams / channels report no duration
        try:
            duration = float(duration)
        except (TypeError, ValueError):
            continue
        if duration <= 0 or duration > config.MAX_REFERENCE_DURATION_SEC:
            continue
        if entry.get("live_status") in ("is_live", "is_upcoming", "post_live"):
            continue
        url = entry.get("url") or ""
        if not str(url).startswith("http"):
            url = f"https://www.youtube.com/watch?v={video_id}"
        results.append(
            {
                "video_id": str(video_id),
                "title": entry.get("title") or "",
                "channel": entry.get("channel") or entry.get("uploader") or "",
                "duration_sec": duration,
                "thumbnail_url": _thumbnail(entry, str(video_id)),
                "view_count": int(entry.get("view_count") or 0),
                "url": url,
            }
        )
        if len(results) >= limit:
            break
    return results


def _thumbnail(entry: dict, video_id: str) -> str:
    """Best thumbnail URL for a flat search entry."""
    thumb = entry.get("thumbnail")
    if thumb:
        return str(thumb)
    thumbs = entry.get("thumbnails") or []
    if thumbs:
        best = max(
            (t for t in thumbs if t.get("url")),
            key=lambda t: (t.get("width") or 0) * (t.get("height") or 0),
            default=None,
        )
        if best:
            return str(best["url"])
    return f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"
