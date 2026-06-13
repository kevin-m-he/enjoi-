# enjoi享受 — User Guide

enjoi turns a reference track plus **one** take of you singing into a finished, mixed, release-ready song. The whole flow is six screens, left to right. This guide walks through each one, then explains the Similarity slider and the Originality Check report.

---

## The six screens

### 1. Search

Type a song, artist, or vibe into the search bar at the top. Results appear as a grid of YouTube videos with thumbnail, title, channel, and duration.

- Pick something close to the *feel* you want — energy, genre, tempo. You're choosing a style reference, not a song to copy.
- Videos over 10 minutes are filtered out by default.
- Click a result to select it as your reference.

### 2. Analysis

The app fetches the reference's audio *for analysis only* and extracts its musical DNA: BPM, key, time signature, song structure (intro/verse/chorus/...), energy curve, groove, and instrumentation profile. You'll see a summary card when it's done.

What you should know:

- Only the **descriptors** (numbers and labels) are kept. The reference audio itself sits in a temporary sandbox and is **automatically deleted** after your instrumental is generated. It can never end up in your song.
- This step needs internet; everything after it runs fully offline.

### 3. Similarity

One slider, 0% to 100%: how closely should your new instrumental track the reference's *style*?

| Slider | What it means |
|---|---|
| 0% | Only the song **length** is kept. Everything else — tempo, key, structure, groove — is freely generated. |
| 25% | Loose family resemblance: tempo within ~15%, a related key, "has a chorus." |
| 50% | Same mode, tempo within ~7%, same number of sections, section-level energy match. |
| 75% | Same key family, tempo within ~3%, same section order, similar groove. |
| 100% | Same tempo, same key, same section order and lengths, matched energy curve, groove, and instrument palette. |
| **Any value** | **The melody and chord progression are always 100% original, and reference audio is never in the output.** |

The label under the slider tells you in plain words what your setting does, e.g. *"72% — same key, same structure, tempo within 3%, similar groove — melody & chords 100% original."*

Hit **Generate Instrumental**. A progress bar runs (a minute or three on a good GPU; much longer on CPU — grab a coffee), then a preview player appears. Behind the scenes every render also has to pass the Originality Check (see below) before you're allowed to hear it.

### 4. Vocal Upload

Drag and drop (or browse for) your one-take vocal: a single `.wav` or `.mp3` of you singing your song idea straight through. The app transcribes the lyrics, splits the take into phrases at breath points, and scores every section.

**Tips for a great one-take recording** (this is the single biggest quality lever):

- **Record dry.** No reverb, echo, or "studio" effects from your recording app — enjoi adds polished effects later, and baked-in reverb can't be removed.
- **Quiet room, close mic.** Soft furnishings help; keep a consistent 10–20 cm distance from the mic. Phone voice-memo apps in a closet work surprisingly well.
- **Sing continuously.** One pass, start to finish. Don't stop and restart — the app expects one continuous take and uses your breaths and pauses to find phrase boundaries.
- **Give your hook some welly.** The app detects your chorus partly by energy, pitch height, and repetition — so sing the part you think is the hook like you mean it, and sing it more than once if you can.
- **Stay roughly steady in tempo.** You don't need a metronome — the app can stretch each phrase by up to ±6% to sit on the beat — but wild tempo drift fights the auto-arranger.
- **Don't worry about pitch perfection.** That's what the autotune stage is for.

### 5. Auto-Arrange

The app shows what it found in your take:

- **Chorus** (marked with a star) — the section with the highest *Impact Score*: a blend of energy, pitch range and height, vibrato, repetition, brightness, and lyric "hookiness."
- **Verses** — the remaining material, kept in the order you sang it.
- **Bridge** — the section most similar in character to your chorus (without being it), placed to give a lift before the final chorus. Short take? The app builds the bridge from your runner-up chorus candidate, pitched down slightly and thinned in the mix for contrast.

Below that is a timeline of your vocal phrases placed on the new instrumental. Usually you can just accept it. If the app picked the "wrong" chorus:

- Use the preview buttons next to each section's score to listen.
- Drag any chop to a different slot on the timeline.
- Hit **Re-detect** (with adjusted weight sliders in advanced settings) to re-score.

Instrumental-only intros, outros, and turnarounds are left vocal-free on purpose — that's arrangement, not a bug.

### 6. Mix & Export

Three main controls:

- **Autotune strength** (0–100): 0 is gentle, natural correction; 100 is the hard-tuned T-Pain effect. Default 35 suits most pop.
- **Mix preset**: Pop, Hip-Hop/Trap, R&B, Rock, or Acoustic — pre-suggested from your reference's genre, switchable.
- **Master loudness**: Streaming (−14 LUFS, the default — correct for Spotify/Apple Music), Loud (−9), or Dynamic (−16).

Press **Build My Song**. The app pitch-corrects every phrase to the instrumental's key, runs the full mix chain (vocal EQ/compression/reverb/delay, instrumental ducking under your voice, mastering limiter), and exports:

- `song.wav` (44.1 kHz / 24-bit master) and `song.mp3` (320 kbps)
- optional **stems** (instrumental / tuned vocal) so you can remix elsewhere
- `song_manifest.json` + the Originality Check report — keep these with the song

Find everything in `%APPDATA%\enjoi\projects\<project>\exports\`.

---

## What the Originality Check report means

Every generated instrumental is automatically audited against the reference **before** you ever hear it. The export screen shows a summary like:

> *Originality check: passed — melody overlap 7%, no chord-run matches, no fingerprint matches.*

The four checks, in plain language:

| Check | Question it answers | Pass condition |
|---|---|---|
| Melody similarity | Does the new melody reuse the reference's note patterns? | Less than 25% of 6-note interval sequences shared |
| Harmony similarity | Does it copy the reference's chord progression? | No shared run longer than 4 chords (super-common loops like I–V–vi–IV don't count — those belong to everyone) |
| Chroma fingerprint | Does it *sound* harmonically like the same recording bar-by-bar? | Correlation stays below 0.80 |
| Audio fingerprint | Did any actual reference audio leak through? | Zero matches, ever |

If a render fails any check, the app quietly regenerates the offending sections with a different creative seed (you just see a slightly longer progress bar) — an infringing result is never offered to you.

**Why you can publish the result:** your exported song is built from exactly two sources — the newly generated instrumental and your own voice. The full report is saved as `uniqueness_report.json` next to your export, and `song_manifest.json` records the two-source provenance; both are useful evidence if a distributor asks where your music came from.

**The honest fine print:** no software can *guarantee* legal non-infringement. The check enforces measurable divergence on melody, harmony, and audio — which removes the practical copying risk — but it isn't legal advice. Also note that fetching YouTube audio for analysis may conflict with YouTube's Terms of Service; the cleanest path is using your own audio file as the reference where supported.

---

## Quick answers

- **Can I use two vocal takes?** Not in v1 — one take is the design. Sing it through twice within one recording instead; repetition actually helps chorus detection.
- **Can I write my own lyrics on screen?** Lyrics come from your singing (transcribed automatically); there's no lyric editor in v1.
- **Where are my projects?** `%APPDATA%\enjoi\projects\` — each song is a folder you can back up.
- **Something else broken?** See the Troubleshooting section in the main [README](../README.md).
