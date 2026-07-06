from pathlib import Path

from .. import replicate_client

MODEL_ID = "wan-video/wan-2.2-animate-replace"


def run(appearance_image: Path, motion_video: Path, dest: Path) -> Path:
    output = replicate_client.run_model(
        MODEL_ID,
        {"character_image": appearance_image, "video": motion_video},
    )
    return replicate_client.download(output, dest)
