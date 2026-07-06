# Replicate Avatar Video Generator — Design

## Purpose

A local Python CLI tool that generates realistic avatar videos using Replicate-hosted models. Given a full-body photo (the avatar's identity) and a driving video (motion/expression source), it produces a video of the avatar performing the driving video's motion, with the driving video's audio carried over and lip-synced to the avatar's mouth movements.

Primary use case: personal content creation (e.g. social media, personal projects) — generating videos of a chosen persona driven by reference footage.

This is v1 (Mode 1: picture = identity, video = motion). A second mode — video = base footage, picture = identity swap over existing performance — is planned as a future addition and should reuse this tool's shared plumbing (Replicate client, upload/download, job/config handling) rather than being a separate codebase.

## Scope

**In scope for this spec:**
- Full-body avatar photo + driving video → full-body motion-transfer + lip-synced output video
- CLI-only interface (no web UI)
- Job-based, resumable, staged pipeline calling Replicate for the two heavy stages
- Local Replicate account/token setup instructions

**Out of scope (future work, not this spec):**
- Mode 2 (video-to-video identity/face swap over existing footage)
- Any web/GUI interface
- Batch processing of multiple avatar/video pairs in one invocation
- TTS-driven lip-sync (audio not sourced from the driving video)

## Architecture

Modular staged pipeline, each stage independently resumable with cached intermediate outputs on disk. Chosen over a single all-in-one model (less mature/reliable for full-body work) and over a monolithic two-stage script (which would force re-running the expensive body-motion stage if only the lip-sync stage fails).

```
avatar.jpg + driving.mp4
        │
        ▼
Stage 0 — Prep (local, free)
  validate inputs, extract audio track from driving video (ffmpeg)
  → audio.aac, meta.json
        │
        ▼
Stage 1 — Body Motion Transfer (Replicate)
  input: avatar photo + driving video
  output: stage1_motion.mp4 (silent, full-body motion matched to driving video)
  [cached — skipped on resume if already succeeded]
        │
        ▼
Stage 2 — Lip-Sync (Replicate)
  input: stage1_motion.mp4 + audio.aac
  output: stage2_lipsync.mp4
  [cached — retried alone on failure, without re-running Stage 1]
        │
        ▼
Stage 3 — Post-process (local, free)
  ffmpeg: confirm audio/video mux, normalize container/format
  → final.mp4
```

Each job runs in its own folder (`jobs/<job_id>/`) containing all intermediate and final artifacts plus a `meta.json` tracking per-stage status and which Replicate model/version was used for each stage.

## Components

- **`cli.py`** — argument parsing and entry point:
  `modelz generate --avatar <path> --driving <path> [--out <path>] [--resume <job_id>] [--dry-run]`
- **`replicate_client.py`** — thin wrapper around the Replicate API: uploads local files, creates predictions, polls for completion, downloads outputs. Shared by both stages now, and by Mode 2 later.
- **`pipeline/stage0_prep.py`** — input validation (file exists, readable as image/video via ffmpeg) and audio extraction.
- **`pipeline/stage1_motion.py`** — calls the body-motion-transfer model on Replicate.
- **`pipeline/stage2_lipsync.py`** — calls the lip-sync model on Replicate.
- **`pipeline/stage3_postprocess.py`** — local ffmpeg mux/format normalization.
- **`orchestrator.py`** — runs stages in order; before each stage, checks `meta.json` — if the stage is marked `done` and its output file exists, skips it; otherwise runs it and updates `meta.json`.
- **`config.py`** — loads `REPLICATE_API_TOKEN` from a local `.env` file (never hardcoded or committed).

## Data Flow / Resume Behavior

1. On `generate`, a new `job_id` is created and its folder set up (unless `--resume <job_id>` is passed, which loads the existing job folder and its `meta.json`).
2. Stage 0 always re-validates inputs cheaply (no API cost).
3. Stages 1–2 each check `meta.json` first: if already `done`, skip straight to the next stage. This means a failure at Stage 2 never triggers a re-run (and re-payment) of Stage 1.
4. On any stage failure (Replicate error, timeout, rate limit), that stage is marked `failed` with the error recorded in `meta.json`, and the CLI exits non-zero with a clear message pointing to `--resume <job_id>`.
5. Stage 3 runs locally and produces `final.mp4` in the job folder (and optionally copies/moves it to `--out` if specified).

## Setup

- A setup step (documented in README, optionally a setup script) walks through: creating a Replicate account, enabling billing (required since model runs are pay-per-use), generating an API token, and placing it in a local `.env` file (gitignored).
- Specific Replicate model choices for Stage 1 (body-motion transfer) and Stage 2 (lip-sync) are not pinned in this spec — the Replicate model catalog changes over time. The implementation plan should verify current, available, reasonably-priced candidates for each stage at build time (e.g. searching Replicate's catalog for full-body pose/motion-transfer models and dedicated audio-driven lip-sync models) and record the chosen model IDs/versions in `meta.json` per job for reproducibility.

## Error Handling

- All local input validation (file existence, readable as image/video) happens before any paid Replicate call, to avoid spending money on invalid inputs.
- Replicate call failures (timeout, model-side error, rate limiting) are caught, logged with the underlying error message, and recorded in that stage's `meta.json` entry as `failed` — they do not silently continue to the next stage.
- `--resume <job_id>` re-enters the orchestrator at the first non-`done` stage.

## Testing

- **Unit tests** (no real Replicate calls, client mocked): CLI argument parsing, input validation, ffmpeg audio-extraction helper, and the orchestrator's skip/resume logic given various `meta.json` states.
- **`--dry-run` mode**: validates the full pipeline wiring (input files, config/token presence, stage sequencing, job folder setup) without making any real Replicate API calls or incurring cost.
- **Manual smoke test**: one real end-to-end run with a short, low-cost driving video, checked by eye — video realism and lip-sync quality cannot be meaningfully asserted by an automated test, so this is an explicit manual verification step rather than part of the automated suite.

## Future Work (not this spec)

- Mode 2: video-to-video identity/face swap over existing footage, reusing `replicate_client.py`, `config.py`, and the job/`meta.json` orchestration pattern.
- Optional local web UI on top of the same pipeline.
- TTS-driven lip-sync mode (audio not sourced from the driving video).
