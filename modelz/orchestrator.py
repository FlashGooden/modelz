import json
import shutil
import time
import uuid
from pathlib import Path

from . import config
from .errors import StageFailedError
from .pipeline import stage0_prep, stage1_motion, stage2_lipsync, stage3_postprocess

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
    path.write_text(json.dumps(meta, indent=2, default=str))


def mark_stage(meta: dict, stage: str, status: str, **extra) -> None:
    meta["stages"][stage] = {"status": status, **extra}


def stage_done(meta: dict, stage: str, output_key: str) -> bool:
    entry = meta["stages"].get(stage, {})
    if entry.get("status") != "done":
        return False
    output_path = entry.get(output_key)
    return bool(output_path) and Path(output_path).exists()


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
        try:
            stage3_postprocess.mux(lipsync_path, final_path)
        except Exception as exc:
            mark_stage(meta, "postprocess", "failed", error=str(exc))
            save_meta(job_id, meta)
            raise StageFailedError(f"Stage 3 (postprocess) failed: {exc}") from exc
        mark_stage(meta, "postprocess", "done", output=str(final_path))
        save_meta(job_id, meta)

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(final_path, out_path)
        return out_path
    return final_path
