import argparse
import sys
from pathlib import Path

from . import orchestrator
from .errors import ModelzError, StageFailedError


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
        if isinstance(exc, StageFailedError):
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
