"""Create SafePay's access secrets in ~/.safepay (outside the repository, mode 600).

  api_keys        one key per bank system that may call /v1/screen and friends
  analyst_tokens  one token per analyst, as "token name"
  demo_password   the password for the public demo (user name: judge)

Existing files are kept, so running this twice never locks anyone out.
Usage: python3 -m safepay.scripts.make_keys [--show-demo-password]
"""
from __future__ import annotations

import argparse
import secrets
import string

from safepay.app.security import HOME

WORDS = string.ascii_lowercase + string.digits


def write(name: str, content: str) -> bool:
    path = HOME / name
    if path.exists():
        return False
    HOME.mkdir(parents=True, exist_ok=True)
    HOME.chmod(0o700)
    path.write_text(content)
    path.chmod(0o600)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-demo-password", action="store_true")
    args = ap.parse_args()
    made = {
        "api_keys": write("api_keys", f"# one key per line\n{secrets.token_urlsafe(32)}\n"),
        "analyst_tokens": write("analyst_tokens", "# token name\n" + "".join(
            f"{secrets.token_urlsafe(32)} {who}\n" for who in ("analyst-1", "analyst-2"))),
        # Easy to type from a submission page: three groups of four characters.
        "demo_password": write("demo_password", "-".join(
            "".join(secrets.choice(WORDS) for _ in range(4)) for _ in range(3)) + "\n"),
    }
    for name, new in made.items():
        print(f"{HOME / name}: {'created' if new else 'kept (already existed)'}")
    if args.show_demo_password:
        print("public demo login: judge /", (HOME / "demo_password").read_text().strip())


if __name__ == "__main__":
    main()
