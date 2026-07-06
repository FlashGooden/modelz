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
