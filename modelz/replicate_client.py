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
