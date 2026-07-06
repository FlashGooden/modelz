# modelz

A local CLI that generates a realistic full-body avatar video from a photo and a
driving video, using Replicate-hosted models:

1. **Stage 1 — body motion transfer** (`zsxkib/mimic-motion`): animates the avatar
   photo to match the driving video's motion.
2. **Stage 2 — lip-sync** (`bytedance/latentsync`): syncs the driving video's audio
   to the animated avatar's mouth movements.

Each stage's output is cached to disk, so a failure in Stage 2 never forces you to
re-run (and re-pay for) Stage 1.

## Prerequisites

- Python 3.10+
- `ffmpeg` and `ffprobe` on your PATH (`brew install ffmpeg` on macOS)
- A Replicate account with billing enabled (models are pay-per-run)

## Setup

1. Create an account at replicate.com and enable billing under
   [replicate.com/account/billing](https://replicate.com/account/billing) — model
   runs cost money per generation (Stage 1 ~$0.98/run, Stage 2 ~$0.08/run at time
   of writing).
2. Generate an API token at
   [replicate.com/account/api-tokens](https://replicate.com/account/api-tokens).
3. Install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements-dev.txt
   ```
4. Copy `.env.example` to `.env` and paste your token in:
   ```bash
   cp .env.example .env
   # edit .env, set REPLICATE_API_TOKEN=<your token>
   ```

## Usage

```bash
python -m modelz generate --avatar avatar.jpg --driving driving.mp4 --out final.mp4
```

Validate your setup without spending money on real Replicate calls:

```bash
python -m modelz generate --avatar avatar.jpg --driving driving.mp4 --dry-run
```

If a run fails partway through, resume it (already-succeeded stages are skipped):

```bash
python -m modelz generate --avatar avatar.jpg --driving driving.mp4 --resume <job_id>
```

Generated files live under `jobs/<job_id>/` (intermediate stage outputs, `meta.json`
status, and `final.mp4`).

## Exit Codes

- `0` — success
- `1` — a runtime failure (e.g. missing Replicate API token, invalid input file, or a pipeline stage failed) — see the printed error message, and the resume hint if one is shown
- `2` — invalid command-line usage (e.g. missing a required flag) — this is standard `argparse` behavior

## Running Tests

```bash
pytest -v
```

## Manual Smoke Test

Automated tests mock every Replicate call, so they can't verify real output quality.
Before relying on this tool, run one real end-to-end job with a short (~5 second),
low-cost driving video and a full-body avatar photo, then inspect `final.mp4` by eye
for motion realism and lip-sync accuracy.
