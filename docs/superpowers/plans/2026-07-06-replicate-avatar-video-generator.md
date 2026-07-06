# Replicate Avatar Video Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python CLI (`modelz generate --avatar <photo> --driving <video>`) that produces a realistic full-body avatar video by running two Replicate models in sequence — body-motion transfer, then lip-sync — with each stage cached and resumable so a failure never forces re-paying for an already-successful stage.

**Architecture:** A modular staged pipeline (spec: [docs/superpowers/specs/2026-07-06-replicate-avatar-video-generator-design.md](../specs/2026-07-06-replicate-avatar-video-generator-design.md)). Stage 0 (local, free) validates inputs and extracts the driving video's audio track. Stage 1 (Replicate) runs body-motion transfer. Stage 2 (Replicate) runs lip-sync using Stage 1's output video + Stage 0's extracted audio. Stage 3 (local, free) muxes/normalizes the final file. An `orchestrator` module tracks per-stage status in a `meta.json` file per job so `--resume <job_id>` only re-runs stages that haven't succeeded yet.

**Tech Stack:** Python 3.10+, the `replicate` Python client, `ffmpeg`/`ffprobe` (system binaries, called via `subprocess`), `pytest` for tests.

**Verified Replicate models (as of 2026-07-06, sourced directly from each model's public `predict.py` on GitHub — re-verify if these models are deprecated by the time you implement):**
- Stage 1 — `zsxkib/mimic-motion` (wraps Tencent's MimicMotion). Inputs used here: `appearance_image` (Path), `motion_video` (Path). ~$0.98/run, ~12 min on an A100.
- Stage 2 — `bytedance/latentsync`. Inputs used here: `video` (Path), `audio` (Path). ~$0.08/run.

---

## File Structure

```
modelz/
├── .env.example                     # Create: template for REPLICATE_API_TOKEN
├── .gitignore                       # Modify: add .env and jobs/
├── README.md                        # Modify: setup + usage docs
├── requirements.txt                 # Create: runtime deps (replicate)
├── requirements-dev.txt             # Create: dev deps (pytest)
├── modelz/
│   ├── __init__.py                  # Create: empty
│   ├── __main__.py                  # Create: `python -m modelz` entrypoint
│   ├── errors.py                    # Create: shared exception hierarchy
│   ├── config.py                    # Create: REPLICATE_API_TOKEN loading
│   ├── replicate_client.py          # Create: Replicate call/upload/download wrapper
│   ├── orchestrator.py              # Create: job meta.json + stage sequencing/resume
│   ├── cli.py                       # Create: argparse entrypoint
│   └── pipeline/
│       ├── __init__.py              # Create: empty
│       ├── stage0_prep.py           # Create: input validation + audio extraction
│       ├── stage1_motion.py         # Create: body-motion Replicate call
│       ├── stage2_lipsync.py        # Create: lip-sync Replicate call
│       └── stage3_postprocess.py    # Create: local ffmpeg mux/normalize
└── tests/
    ├── __init__.py                  # Create: empty
    ├── conftest.py                  # Create: tiny_image/tiny_video/not_a_video fixtures
    ├── test_config.py               # Create
    ├── test_stage0_prep.py          # Create
    ├── test_replicate_client.py     # Create
    ├── test_stage1_motion.py        # Create
    ├── test_stage2_lipsync.py       # Create
    ├── test_stage3_postprocess.py   # Create
    ├── test_orchestrator_meta.py    # Create
    ├── test_orchestrator_pipeline.py # Create
    └── test_cli.py                  # Create
```

`jobs/` is created at runtime (gitignored) — one subfolder per job holding intermediate files and `meta.json`.

---

### Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `requirements-dev.txt`
- Create: `.env.example`
- Modify: `.gitignore`
- Create: `modelz/__init__.py`
- Create: `modelz/errors.py`
- Create: `modelz/pipeline/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Confirm system prerequisites**

Run: `ffmpeg -version && ffprobe -version && python3 --version`
Expected: version output for all three, with Python 3.10 or newer. If `ffmpeg`/`ffprobe` are missing, install them first (e.g. `brew install ffmpeg` on macOS) — the whole pipeline depends on them.

- [ ] **Step 2: Create runtime and dev dependency files**

`requirements.txt`:
```
replicate>=1.0.0
```

`requirements-dev.txt`:
```
-r requirements.txt
pytest>=8.0.0
```

- [ ] **Step 3: Create the `.env.example` template**

`.env.example`:
```
REPLICATE_API_TOKEN=your-replicate-api-token-here
```

- [ ] **Step 4: Update `.gitignore`**

Add to the existing `.gitignore` (which currently only has `.superpowers/`):
```
.env
jobs/
__pycache__/
*.pyc
```

- [ ] **Step 5: Create empty package `__init__.py` files**

Create `modelz/__init__.py` (empty file), `modelz/pipeline/__init__.py` (empty file), and `tests/__init__.py` (empty file).

- [ ] **Step 6: Create the shared exception hierarchy**

`modelz/errors.py`:
```python
class ModelzError(Exception):
    """Base class for expected modelz failures (as opposed to bugs)."""


class ConfigError(ModelzError):
    """Raised when required configuration (e.g. the API token) is missing."""


class InputValidationError(ModelzError):
    """Raised when a user-supplied file isn't a usable image/video."""


class StageFailedError(ModelzError):
    """Raised when a pipeline stage (Replicate call or local processing) fails."""
```

- [ ] **Step 7: Install dependencies**

Run: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements-dev.txt`
Expected: `replicate` and `pytest` install successfully.

- [ ] **Step 8: Commit**

```bash
git add requirements.txt requirements-dev.txt .env.example .gitignore modelz tests
git commit -m "chore: scaffold modelz package structure"
```

---

### Task 2: Config — Replicate API Token Loading

**Files:**
- Create: `modelz/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_config.py`:
```python
import pytest

from modelz import config
from modelz.errors import ConfigError


def test_load_api_token_from_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("REPLICATE_API_TOKEN=r8_test123\n")

    token = config.load_api_token(env_file)

    assert token == "r8_test123"


def test_load_api_token_uses_existing_environment_variable(tmp_path, monkeypatch):
    monkeypatch.setenv("REPLICATE_API_TOKEN", "r8_from_env")
    missing_env_file = tmp_path / ".env"

    token = config.load_api_token(missing_env_file)

    assert token == "r8_from_env"


def test_load_api_token_missing_raises_config_error(tmp_path, monkeypatch):
    monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)
    missing_env_file = tmp_path / ".env"

    with pytest.raises(ConfigError):
        config.load_api_token(missing_env_file)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'modelz.config'`

- [ ] **Step 3: Write the implementation**

`modelz/config.py`:
```python
import os
from pathlib import Path

from .errors import ConfigError


def load_api_token(env_path: Path | None = None) -> str:
    """Load REPLICATE_API_TOKEN from a .env file (if present) and the environment."""
    env_path = env_path or Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        raise ConfigError(
            "REPLICATE_API_TOKEN is not set. Copy .env.example to .env and add your token."
        )
    return token
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add modelz/config.py tests/test_config.py
git commit -m "feat: load Replicate API token from .env or environment"
```

---

### Task 3: Stage 0 — Input Validation & Audio Extraction

**Files:**
- Create: `modelz/pipeline/stage0_prep.py`
- Create: `tests/conftest.py`
- Test: `tests/test_stage0_prep.py`

- [ ] **Step 1: Create shared media fixtures**

`tests/conftest.py`:
```python
import subprocess

import pytest


@pytest.fixture
def tiny_image(tmp_path):
    path = tmp_path / "avatar.jpg"
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=blue:s=64x64",
            "-frames:v", "1", str(path),
        ],
        capture_output=True, check=True,
    )
    return path


@pytest.fixture
def tiny_video(tmp_path):
    path = tmp_path / "driving.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=red:s=64x64:d=1",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
            "-c:v", "libx264", "-c:a", "aac", "-shortest", str(path),
        ],
        capture_output=True, check=True,
    )
    return path


@pytest.fixture
def not_a_video(tmp_path):
    path = tmp_path / "not_video.txt"
    path.write_text("hello")
    return path
```

- [ ] **Step 2: Write the failing tests**

`tests/test_stage0_prep.py`:
```python
import subprocess

import pytest

from modelz.errors import InputValidationError
from modelz.pipeline import stage0_prep


def test_validate_image_accepts_real_image(tiny_image):
    stage0_prep.validate_image(tiny_image)  # should not raise


def test_validate_image_rejects_missing_file(tmp_path):
    with pytest.raises(InputValidationError):
        stage0_prep.validate_image(tmp_path / "missing.jpg")


def test_validate_image_rejects_non_media_file(not_a_video):
    with pytest.raises(InputValidationError):
        stage0_prep.validate_image(not_a_video)


def test_validate_video_accepts_real_video(tiny_video):
    stage0_prep.validate_video(tiny_video)  # should not raise


def test_validate_video_rejects_missing_file(tmp_path):
    with pytest.raises(InputValidationError):
        stage0_prep.validate_video(tmp_path / "missing.mp4")


def test_validate_video_rejects_non_media_file(not_a_video):
    with pytest.raises(InputValidationError):
        stage0_prep.validate_video(not_a_video)


def test_extract_audio_creates_file_with_audio_stream(tiny_video, tmp_path):
    dest = tmp_path / "out" / "audio.aac"

    result = stage0_prep.extract_audio(tiny_video, dest)

    assert result == dest
    assert dest.exists()
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(dest),
        ],
        capture_output=True, text=True,
    )
    assert "audio" in probe.stdout
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_stage0_prep.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'modelz.pipeline.stage0_prep'`

- [ ] **Step 4: Write the implementation**

`modelz/pipeline/stage0_prep.py`:
```python
import subprocess
from pathlib import Path

from ..errors import InputValidationError


def _is_readable_media(path: Path) -> bool:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path),
        ],
        capture_output=True, text=True,
    )
    return result.returncode == 0 and "video" in result.stdout


def validate_image(path: Path) -> None:
    if not path.exists():
        raise InputValidationError(f"Avatar image not found: {path}")
    if not _is_readable_media(path):
        raise InputValidationError(f"File is not a readable image: {path}")


def validate_video(path: Path) -> None:
    if not path.exists():
        raise InputValidationError(f"Driving video not found: {path}")
    if not _is_readable_media(path):
        raise InputValidationError(f"File is not a readable video: {path}")


def extract_audio(driving_video: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(driving_video), "-vn", "-acodec", "aac", str(dest)],
        capture_output=True, check=True,
    )
    return dest
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_stage0_prep.py -v`
Expected: 7 passed

- [ ] **Step 6: Commit**

```bash
git add modelz/pipeline/stage0_prep.py tests/conftest.py tests/test_stage0_prep.py
git commit -m "feat: validate avatar/driving inputs and extract driving audio"
```

---

### Task 4: Replicate Client Wrapper

**Files:**
- Create: `modelz/replicate_client.py`
- Test: `tests/test_replicate_client.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_replicate_client.py`:
```python
from unittest.mock import patch

from modelz import replicate_client


def test_run_model_opens_path_inputs_and_calls_replicate_run(tmp_path):
    file_input = tmp_path / "in.mp4"
    file_input.write_bytes(b"fake video bytes")

    with patch("modelz.replicate_client.replicate.run") as mock_run:
        mock_run.return_value = "https://replicate.delivery/out.mp4"

        result = replicate_client.run_model(
            "owner/model", {"video": file_input, "seed": 42}
        )

    assert result == "https://replicate.delivery/out.mp4"
    called_input = mock_run.call_args.kwargs["input"]
    assert called_input["seed"] == 42
    assert called_input["video"].name == file_input


def test_run_model_unwraps_list_output(tmp_path):
    file_input = tmp_path / "in.mp4"
    file_input.write_bytes(b"fake")

    with patch("modelz.replicate_client.replicate.run") as mock_run:
        mock_run.return_value = ["https://replicate.delivery/out.mp4"]

        result = replicate_client.run_model("owner/model", {"video": file_input})

    assert result == "https://replicate.delivery/out.mp4"


def test_download_saves_url_to_dest(tmp_path):
    dest = tmp_path / "out" / "final.mp4"

    with patch("modelz.replicate_client.urllib.request.urlretrieve") as mock_urlretrieve:
        result = replicate_client.download("https://replicate.delivery/out.mp4", dest)

    assert result == dest
    assert dest.parent.exists()
    mock_urlretrieve.assert_called_once_with(
        "https://replicate.delivery/out.mp4", str(dest)
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_replicate_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'modelz.replicate_client'`

- [ ] **Step 3: Write the implementation**

`modelz/replicate_client.py`:
```python
import urllib.request
from pathlib import Path

import replicate


def run_model(model: str, inputs: dict) -> str:
    """Run a Replicate model, opening any Path inputs as local file uploads."""
    opened = {}
    try:
        for key, value in inputs.items():
            opened[key] = open(value, "rb") if isinstance(value, Path) else value
        output = replicate.run(model, input=opened)
    finally:
        for value in opened.values():
            if hasattr(value, "close"):
                value.close()

    if isinstance(output, list):
        output = output[0]
    return str(output)


def download(source, dest: Path) -> Path:
    """Download a Replicate output (a URL string or a file-like object) to dest."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(source, "read"):
        dest.write_bytes(source.read())
    else:
        urllib.request.urlretrieve(str(source), str(dest))
    return dest
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_replicate_client.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add modelz/replicate_client.py tests/test_replicate_client.py
git commit -m "feat: add Replicate call/upload/download wrapper"
```

---

### Task 5: Stage 1 — Body Motion Transfer

**Files:**
- Create: `modelz/pipeline/stage1_motion.py`
- Test: `tests/test_stage1_motion.py`

- [ ] **Step 1: Write the failing test**

`tests/test_stage1_motion.py`:
```python
from unittest.mock import patch

from modelz.pipeline import stage1_motion


def test_run_calls_replicate_client_with_expected_inputs(tmp_path):
    appearance_image = tmp_path / "avatar.jpg"
    motion_video = tmp_path / "driving.mp4"
    dest = tmp_path / "stage1_motion.mp4"

    with patch(
        "modelz.pipeline.stage1_motion.replicate_client.run_model"
    ) as mock_run, patch(
        "modelz.pipeline.stage1_motion.replicate_client.download"
    ) as mock_download:
        mock_run.return_value = "https://replicate.delivery/motion.mp4"
        mock_download.return_value = dest

        result = stage1_motion.run(appearance_image, motion_video, dest)

    mock_run.assert_called_once_with(
        stage1_motion.MODEL_ID,
        {"appearance_image": appearance_image, "motion_video": motion_video},
    )
    mock_download.assert_called_once_with("https://replicate.delivery/motion.mp4", dest)
    assert result == dest
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stage1_motion.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'modelz.pipeline.stage1_motion'`

- [ ] **Step 3: Write the implementation**

`modelz/pipeline/stage1_motion.py`:
```python
from pathlib import Path

from .. import replicate_client

MODEL_ID = "zsxkib/mimic-motion"


def run(appearance_image: Path, motion_video: Path, dest: Path) -> Path:
    output = replicate_client.run_model(
        MODEL_ID,
        {"appearance_image": appearance_image, "motion_video": motion_video},
    )
    return replicate_client.download(output, dest)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_stage1_motion.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add modelz/pipeline/stage1_motion.py tests/test_stage1_motion.py
git commit -m "feat: add Stage 1 body-motion transfer via zsxkib/mimic-motion"
```

---

### Task 6: Stage 2 — Lip-Sync

**Files:**
- Create: `modelz/pipeline/stage2_lipsync.py`
- Test: `tests/test_stage2_lipsync.py`

- [ ] **Step 1: Write the failing test**

`tests/test_stage2_lipsync.py`:
```python
from unittest.mock import patch

from modelz.pipeline import stage2_lipsync


def test_run_calls_replicate_client_with_expected_inputs(tmp_path):
    video = tmp_path / "stage1_motion.mp4"
    audio = tmp_path / "audio.aac"
    dest = tmp_path / "stage2_lipsync.mp4"

    with patch(
        "modelz.pipeline.stage2_lipsync.replicate_client.run_model"
    ) as mock_run, patch(
        "modelz.pipeline.stage2_lipsync.replicate_client.download"
    ) as mock_download:
        mock_run.return_value = "https://replicate.delivery/lipsync.mp4"
        mock_download.return_value = dest

        result = stage2_lipsync.run(video, audio, dest)

    mock_run.assert_called_once_with(
        stage2_lipsync.MODEL_ID, {"video": video, "audio": audio}
    )
    mock_download.assert_called_once_with("https://replicate.delivery/lipsync.mp4", dest)
    assert result == dest
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stage2_lipsync.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'modelz.pipeline.stage2_lipsync'`

- [ ] **Step 3: Write the implementation**

`modelz/pipeline/stage2_lipsync.py`:
```python
from pathlib import Path

from .. import replicate_client

MODEL_ID = "bytedance/latentsync"


def run(video: Path, audio: Path, dest: Path) -> Path:
    output = replicate_client.run_model(MODEL_ID, {"video": video, "audio": audio})
    return replicate_client.download(output, dest)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_stage2_lipsync.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add modelz/pipeline/stage2_lipsync.py tests/test_stage2_lipsync.py
git commit -m "feat: add Stage 2 lip-sync via bytedance/latentsync"
```

---

### Task 7: Stage 3 — Post-process

**Files:**
- Create: `modelz/pipeline/stage3_postprocess.py`
- Test: `tests/test_stage3_postprocess.py`

- [ ] **Step 1: Write the failing test**

`tests/test_stage3_postprocess.py`:
```python
import subprocess

from modelz.pipeline import stage3_postprocess


def test_mux_produces_a_playable_output_file(tiny_video, tmp_path):
    dest = tmp_path / "out" / "final.mp4"

    result = stage3_postprocess.mux(tiny_video, dest)

    assert result == dest
    assert dest.exists()
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "csv=p=0", str(dest),
        ],
        capture_output=True, text=True,
    )
    assert probe.returncode == 0
    assert probe.stdout.strip() != ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stage3_postprocess.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'modelz.pipeline.stage3_postprocess'`

- [ ] **Step 3: Write the implementation**

`modelz/pipeline/stage3_postprocess.py`:
```python
import subprocess
from pathlib import Path


def mux(src: Path, dest: Path) -> Path:
    """Normalize the final container/format (re-mux without re-encoding)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-c", "copy", str(dest)],
        capture_output=True, check=True,
    )
    return dest
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_stage3_postprocess.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add modelz/pipeline/stage3_postprocess.py tests/test_stage3_postprocess.py
git commit -m "feat: add Stage 3 local post-process mux"
```

---

### Task 8: Orchestrator — Job Metadata Helpers

**Files:**
- Create: `modelz/orchestrator.py`
- Test: `tests/test_orchestrator_meta.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_orchestrator_meta.py`:
```python
from modelz import orchestrator


def test_new_job_id_is_unique():
    assert orchestrator.new_job_id() != orchestrator.new_job_id()


def test_load_meta_returns_default_structure_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator, "JOBS_DIR", tmp_path)

    meta = orchestrator.load_meta("job123")

    assert meta["job_id"] == "job123"
    for stage in ("prep", "motion", "lipsync", "postprocess"):
        assert meta["stages"][stage]["status"] == "pending"


def test_save_and_load_meta_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator, "JOBS_DIR", tmp_path)
    meta = orchestrator.load_meta("job123")
    meta["stages"]["prep"]["status"] = "done"

    orchestrator.save_meta("job123", meta)
    reloaded = orchestrator.load_meta("job123")

    assert reloaded["stages"]["prep"]["status"] == "done"


def test_mark_stage_sets_status_and_extra_fields():
    meta = {"stages": {"motion": {"status": "pending"}}}

    orchestrator.mark_stage(
        meta, "motion", "done", output="foo.mp4", model="zsxkib/mimic-motion"
    )

    assert meta["stages"]["motion"] == {
        "status": "done",
        "output": "foo.mp4",
        "model": "zsxkib/mimic-motion",
    }


def test_stage_done_true_only_when_status_done_and_output_file_exists(tmp_path):
    output_file = tmp_path / "out.mp4"
    output_file.write_bytes(b"data")
    meta = {"stages": {"motion": {"status": "done", "output": str(output_file)}}}

    assert orchestrator.stage_done(meta, "motion", "output") is True

    output_file.unlink()
    assert orchestrator.stage_done(meta, "motion", "output") is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_orchestrator_meta.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'modelz.orchestrator'`

- [ ] **Step 3: Write the implementation**

`modelz/orchestrator.py`:
```python
import json
import time
import uuid
from pathlib import Path

JOBS_DIR = Path("jobs")


def new_job_id() -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def _default_meta(job_id: str) -> dict:
    return {
        "job_id": job_id,
        "stages": {
            "prep": {"status": "pending"},
            "motion": {"status": "pending"},
            "lipsync": {"status": "pending"},
            "postprocess": {"status": "pending"},
        },
    }


def load_meta(job_id: str) -> dict:
    path = job_dir(job_id) / "meta.json"
    if not path.exists():
        return _default_meta(job_id)
    return json.loads(path.read_text())


def save_meta(job_id: str, meta: dict) -> None:
    path = job_dir(job_id) / "meta.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2))


def mark_stage(meta: dict, stage: str, status: str, **extra) -> None:
    meta["stages"][stage] = {"status": status, **extra}


def stage_done(meta: dict, stage: str, output_key: str) -> bool:
    entry = meta["stages"].get(stage, {})
    if entry.get("status") != "done":
        return False
    output_path = entry.get(output_key)
    return bool(output_path) and Path(output_path).exists()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_orchestrator_meta.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add modelz/orchestrator.py tests/test_orchestrator_meta.py
git commit -m "feat: add job meta.json helpers for stage tracking"
```

---

### Task 9: Orchestrator — Pipeline Execution & Resume

**Files:**
- Modify: `modelz/orchestrator.py`
- Test: `tests/test_orchestrator_pipeline.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_orchestrator_pipeline.py`:
```python
from unittest.mock import patch

import pytest

from modelz import orchestrator
from modelz.errors import StageFailedError


@pytest.fixture(autouse=True)
def isolated_jobs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator, "JOBS_DIR", tmp_path)
    yield tmp_path


def test_dry_run_only_runs_prep_stage(tiny_image, tiny_video):
    with patch("modelz.orchestrator.stage1_motion.run") as mock_motion, patch(
        "modelz.orchestrator.stage2_lipsync.run"
    ) as mock_lipsync, patch("modelz.orchestrator.stage3_postprocess.mux") as mock_mux:
        orchestrator.run_pipeline(
            avatar_image=tiny_image,
            driving_video=tiny_video,
            job_id="jobA",
            out_path=None,
            dry_run=True,
        )

    mock_motion.assert_not_called()
    mock_lipsync.assert_not_called()
    mock_mux.assert_not_called()
    meta = orchestrator.load_meta("jobA")
    assert meta["stages"]["prep"]["status"] == "done"


def test_full_pipeline_runs_all_stages_in_order(tiny_image, tiny_video, monkeypatch):
    monkeypatch.setenv("REPLICATE_API_TOKEN", "r8_test")

    def fake_motion(appearance_image, motion_video, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"motion")
        return dest

    def fake_lipsync(video, audio, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"lipsync")
        return dest

    def fake_mux(src, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"final")
        return dest

    with patch(
        "modelz.orchestrator.stage1_motion.run", side_effect=fake_motion
    ) as mock_motion, patch(
        "modelz.orchestrator.stage2_lipsync.run", side_effect=fake_lipsync
    ) as mock_lipsync, patch(
        "modelz.orchestrator.stage3_postprocess.mux", side_effect=fake_mux
    ) as mock_mux:
        result = orchestrator.run_pipeline(
            avatar_image=tiny_image,
            driving_video=tiny_video,
            job_id="jobB",
            out_path=None,
            dry_run=False,
        )

    mock_motion.assert_called_once()
    mock_lipsync.assert_called_once()
    mock_mux.assert_called_once()
    assert result.read_bytes() == b"final"


def test_resume_skips_already_done_motion_stage(tiny_image, tiny_video, monkeypatch, tmp_path):
    monkeypatch.setenv("REPLICATE_API_TOKEN", "r8_test")
    jdir = orchestrator.job_dir("jobC")
    jdir.mkdir(parents=True)
    existing_motion = jdir / "stage1_motion.mp4"
    existing_motion.write_bytes(b"already done")
    (jdir / "audio.aac").write_bytes(b"audio")

    meta = orchestrator.load_meta("jobC")
    orchestrator.mark_stage(meta, "prep", "done", audio=str(jdir / "audio.aac"))
    orchestrator.mark_stage(
        meta, "motion", "done", output=str(existing_motion), model="zsxkib/mimic-motion"
    )
    orchestrator.save_meta("jobC", meta)

    def fake_lipsync(video, audio, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"lipsync")
        return dest

    def fake_mux(src, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"final")
        return dest

    with patch("modelz.orchestrator.stage1_motion.run") as mock_motion, patch(
        "modelz.orchestrator.stage2_lipsync.run", side_effect=fake_lipsync
    ), patch("modelz.orchestrator.stage3_postprocess.mux", side_effect=fake_mux):
        orchestrator.run_pipeline(
            avatar_image=tiny_image,
            driving_video=tiny_video,
            job_id="jobC",
            out_path=None,
            dry_run=False,
        )

    mock_motion.assert_not_called()


def test_motion_stage_failure_marks_meta_and_raises(tiny_image, tiny_video, monkeypatch):
    monkeypatch.setenv("REPLICATE_API_TOKEN", "r8_test")

    with patch(
        "modelz.orchestrator.stage1_motion.run", side_effect=RuntimeError("boom")
    ):
        with pytest.raises(StageFailedError):
            orchestrator.run_pipeline(
                avatar_image=tiny_image,
                driving_video=tiny_video,
                job_id="jobD",
                out_path=None,
                dry_run=False,
            )

    meta = orchestrator.load_meta("jobD")
    assert meta["stages"]["motion"]["status"] == "failed"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_orchestrator_pipeline.py -v`
Expected: FAIL with `AttributeError: module 'modelz.orchestrator' has no attribute 'run_pipeline'` (and no `stage1_motion`/`stage2_lipsync`/`stage3_postprocess` attributes to patch yet)

- [ ] **Step 3: Add pipeline execution to the implementation**

Append to `modelz/orchestrator.py` (add these imports at the top of the file, alongside the existing `json`/`time`/`uuid`/`Path` imports, and add the function at the end):

```python
from . import config
from .errors import StageFailedError
from .pipeline import stage0_prep, stage1_motion, stage2_lipsync, stage3_postprocess


def run_pipeline(
    avatar_image: Path,
    driving_video: Path,
    job_id: str,
    out_path: Path | None,
    dry_run: bool,
) -> Path:
    meta = load_meta(job_id)
    jdir = job_dir(job_id)
    jdir.mkdir(parents=True, exist_ok=True)

    audio_path = jdir / "audio.aac"
    stage0_prep.validate_image(avatar_image)
    stage0_prep.validate_video(driving_video)
    if not audio_path.exists():
        stage0_prep.extract_audio(driving_video, audio_path)
    mark_stage(meta, "prep", "done", audio=str(audio_path))
    save_meta(job_id, meta)

    if dry_run:
        return jdir

    config.load_api_token()

    motion_path = jdir / "stage1_motion.mp4"
    if not stage_done(meta, "motion", "output"):
        try:
            stage1_motion.run(avatar_image, driving_video, motion_path)
        except Exception as exc:
            mark_stage(meta, "motion", "failed", error=str(exc))
            save_meta(job_id, meta)
            raise StageFailedError(f"Stage 1 (motion) failed: {exc}") from exc
        mark_stage(meta, "motion", "done", output=str(motion_path), model=stage1_motion.MODEL_ID)
        save_meta(job_id, meta)

    lipsync_path = jdir / "stage2_lipsync.mp4"
    if not stage_done(meta, "lipsync", "output"):
        try:
            stage2_lipsync.run(motion_path, audio_path, lipsync_path)
        except Exception as exc:
            mark_stage(meta, "lipsync", "failed", error=str(exc))
            save_meta(job_id, meta)
            raise StageFailedError(f"Stage 2 (lipsync) failed: {exc}") from exc
        mark_stage(meta, "lipsync", "done", output=str(lipsync_path), model=stage2_lipsync.MODEL_ID)
        save_meta(job_id, meta)

    final_path = jdir / "final.mp4"
    if not stage_done(meta, "postprocess", "output"):
        stage3_postprocess.mux(lipsync_path, final_path)
        mark_stage(meta, "postprocess", "done", output=str(final_path))
        save_meta(job_id, meta)

    if out_path:
        import shutil

        shutil.copy(final_path, out_path)
        return out_path
    return final_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_orchestrator_pipeline.py -v`
Expected: 4 passed

- [ ] **Step 5: Run the full test suite so far**

Run: `pytest -v`
Expected: all tests from Tasks 2–9 passing (roughly 20 tests)

- [ ] **Step 6: Commit**

```bash
git add modelz/orchestrator.py tests/test_orchestrator_pipeline.py
git commit -m "feat: sequence pipeline stages with cached resume in orchestrator"
```

---

### Task 10: CLI — Argument Parsing & Main Entrypoint

**Files:**
- Create: `modelz/cli.py`
- Create: `modelz/__main__.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_cli.py`:
```python
from pathlib import Path
from unittest.mock import patch

from modelz import cli
from modelz.errors import StageFailedError


def test_build_parser_requires_avatar_and_driving():
    parser = cli.build_parser()
    args = parser.parse_args(["generate", "--avatar", "a.jpg", "--driving", "d.mp4"])

    assert args.avatar == Path("a.jpg")
    assert args.driving == Path("d.mp4")
    assert args.dry_run is False
    assert args.job_id is None


def test_main_success_prints_job_id_and_output(tmp_path, capsys):
    with patch(
        "modelz.cli.orchestrator.new_job_id", return_value="job123"
    ), patch(
        "modelz.cli.orchestrator.run_pipeline", return_value=tmp_path / "final.mp4"
    ) as mock_run:
        exit_code = cli.main(["generate", "--avatar", "a.jpg", "--driving", "d.mp4"])

    assert exit_code == 0
    mock_run.assert_called_once()
    captured = capsys.readouterr()
    assert "job_id=job123" in captured.out
    assert "final.mp4" in captured.out


def test_main_failure_prints_error_and_resume_hint(capsys):
    with patch(
        "modelz.cli.orchestrator.new_job_id", return_value="job456"
    ), patch(
        "modelz.cli.orchestrator.run_pipeline",
        side_effect=StageFailedError("Stage 1 (motion) failed: boom"),
    ):
        exit_code = cli.main(["generate", "--avatar", "a.jpg", "--driving", "d.mp4"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Stage 1 (motion) failed: boom" in captured.err
    assert "--resume job456" in captured.err
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'modelz.cli'`

- [ ] **Step 3: Write the implementation**

`modelz/cli.py`:
```python
import argparse
import sys
from pathlib import Path

from . import orchestrator
from .errors import ModelzError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="modelz")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate an avatar video")
    generate.add_argument("--avatar", required=True, type=Path)
    generate.add_argument("--driving", required=True, type=Path)
    generate.add_argument("--out", type=Path, default=None)
    generate.add_argument("--resume", dest="job_id", default=None)
    generate.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    job_id = args.job_id or orchestrator.new_job_id()
    try:
        result = orchestrator.run_pipeline(
            avatar_image=args.avatar,
            driving_video=args.driving,
            job_id=job_id,
            out_path=args.out,
            dry_run=args.dry_run,
        )
    except ModelzError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(
            f"resume with: modelz generate --avatar {args.avatar} "
            f"--driving {args.driving} --resume {job_id}",
            file=sys.stderr,
        )
        return 1

    print(f"job_id={job_id}")
    print(f"output={result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

`modelz/__main__.py`:
```python
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cli.py -v`
Expected: 3 passed

- [ ] **Step 5: Run the full test suite**

Run: `pytest -v`
Expected: all tests passing (roughly 23 tests)

- [ ] **Step 6: Commit**

```bash
git add modelz/cli.py modelz/__main__.py tests/test_cli.py
git commit -m "feat: add modelz CLI entrypoint"
```

---

### Task 11: Setup & Usage Documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Write the README**

`README.md`:
```markdown
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

## Running Tests

```bash
pytest -v
```

## Manual Smoke Test

Automated tests mock every Replicate call, so they can't verify real output quality.
Before relying on this tool, run one real end-to-end job with a short (~5 second),
low-cost driving video and a full-body avatar photo, then inspect `final.mp4` by eye
for motion realism and lip-sync accuracy.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add setup, usage, and manual smoke test instructions"
```

---

## Manual Smoke Test (run once, after Task 11)

This is not an automated test — Replicate charges real money per call and video
realism can't be asserted programmatically.

1. Set up `.env` with a real `REPLICATE_API_TOKEN` (see README).
2. Pick a short (~5s) driving video with clear audio, and a full-body avatar photo.
3. Run: `python -m modelz generate --avatar <photo> --driving <video> --out smoke_test.mp4`
4. Confirm `smoke_test.mp4` plays, the avatar's motion roughly follows the driving
   video, and the audio is present and reasonably lip-synced.
5. Deliberately interrupt a run (e.g. Ctrl-C after Stage 1 logs) and confirm
   `--resume <job_id>` picks up at Stage 2 rather than re-running Stage 1
   (check `jobs/<job_id>/meta.json` to confirm `motion` stays `"status": "done"`).
