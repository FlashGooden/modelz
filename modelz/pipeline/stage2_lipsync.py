from pathlib import Path

from .. import replicate_client

MODEL_ID = "bytedance/latentsync"


def run(video: Path, audio: Path, dest: Path) -> Path:
    output = replicate_client.run_model(MODEL_ID, {"video": video, "audio": audio})
    return replicate_client.download(output, dest)
