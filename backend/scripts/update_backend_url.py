import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "backend_url.json"


def validate_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("URL must include http(s):// and a host.")
    return value.rstrip("/")


def detect_branch() -> str:
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        branch = result.stdout.strip()
        if branch:
            return branch
    except Exception:
        pass
    return "server_main"


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/update_backend_url.py <base_url>")
        return 1

    try:
        base_url = validate_url(sys.argv[1].strip())
    except ValueError as exc:
        print(f"Invalid URL: {exc}")
        return 1

    payload = {
        "base_url": base_url,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "notes": "Android app reads this file to discover the current backend URL.",
    }
    branch = detect_branch()

    CONFIG_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Updated {CONFIG_PATH.name} -> {base_url}")
    print("Next steps:")
    print("  git add backend_url.json")
    print('  git commit -m "Update backend URL"')
    print(f"  git push origin {branch}")
    print("Android should read:")
    print(f"  https://raw.githubusercontent.com/Ai-pre/MUTON/refs/heads/{branch}/backend_url.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
