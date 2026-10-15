
"""Reject private files from Git commits."""

import sys
from pathlib import PurePosixPath

PRIVATE_DIRECTORIES = {
    "FREDL-library",
    "private",
    "secrets",
}

PRIVATE_SUFFIXES = {
    ".pdf",
    ".env",
    ".key",
    ".pem",
}


def main() -> int:
    blocked = []

    for filename in sys.argv[1:]:
        path = PurePosixPath(filename)

        if any(part in PRIVATE_DIRECTORIES for part in path.parts):
            blocked.append(filename)
        elif path.suffix.lower() in PRIVATE_SUFFIXES:
            blocked.append(filename)
        # A dotfile such as ".env" has no suffix in pathlib.
        elif path.name.lower() == ".env" or path.name.startswith(".env."):
            blocked.append(filename)

    if blocked:
        print("Blocked private files:")
        for filename in blocked:
            print(f"  {filename}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
