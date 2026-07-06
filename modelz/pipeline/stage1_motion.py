from pathlib import Path

from .. import replicate_client

MODEL_ID = "zsxkib/mimic-motion"


def run(appearance_image: Path, motion_video: Path, dest: Path) -> Path:
    output = replicate_client.run_model(
        MODEL_ID,
        {"appearance_image": appearance_image, "motion_video": motion_video},
    )
    return replicate_client.download(output, dest)
