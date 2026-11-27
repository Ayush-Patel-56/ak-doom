"""Release check: is the SUBMITTED brain really the trained champion? Run it next to validate.py.

Why: `validate.py` passes even when submission/weights/champion.pt is a 132-byte Git LFS pointer
(a clone without LFS). DoomAgent then falls back to the random policy on purpose (it must never crash),
so the validator's dry run still completes, with warnings only. This check fails loudly instead.

Checks (exit code 1 if any FAIL):
1. every Git LFS file is real content, not a pointer (champion.pt and the curated maps)
2. champion.pt's SHA-256 equals the one recorded in champion.json
3. DoomAgent loads it as a trained brain ("brain:champion.pt", memory on), not the random fallback
4. the loaded agent reaches the exit of the template map two_rooms.wad (the trained champion does)
5. the checked-out commit is the tip of origin/main and the tree is clean (what the scorer will use)

    uv run python scripts/release_check.py            # all checks
    uv run python scripts/release_check.py --skip-git # 1-4 only (e.g. inside an unpacked copy)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHAMPION = REPO / "submission" / "weights" / "champion.pt"
CHAMPION_JSON = REPO / "submission" / "weights" / "champion.json"
LFS_HEADER = b"version https://git-lfs"


def is_lfs_pointer(path: Path) -> bool:
    with path.open("rb") as f:
        return f.read(len(LFS_HEADER)) == LFS_HEADER


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def check_lfs() -> tuple[bool, str]:
    try:
        files = [ln.split(" ", 2)[-1] for ln in git("lfs", "ls-files").splitlines() if ln.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        files = [str(CHAMPION.relative_to(REPO))]
    pointers = [f for f in files if (REPO / f).is_file() and is_lfs_pointer(REPO / f)]
    missing = [f for f in files if not (REPO / f).is_file()]
    if pointers or missing:
        return False, (f"{len(pointers)} LFS pointer(s) and {len(missing)} missing file(s), e.g. "
                       f"{(pointers + missing)[:3]}; run `git lfs install && git lfs pull`")
    return True, f"{len(files)} LFS files are real content"


def check_hash() -> tuple[bool, str]:
    if not CHAMPION.is_file():
        return False, "submission/weights/champion.pt is missing"
    want = json.loads(CHAMPION_JSON.read_text(encoding="utf-8")).get("weights_sha256", "")
    got = sha256(CHAMPION)
    if got != want:
        return False, f"champion.pt SHA-256 {got[:12]} != champion.json {want[:12]}"
    return True, f"SHA-256 {got[:12]} matches champion.json ({CHAMPION.stat().st_size:,} bytes)"


def check_agent() -> tuple[bool, str, object]:
    sys.path.insert(0, str(REPO / "submission"))
    from doom_env import action_space, observation_space
    from agent import DoomAgent

    agent = DoomAgent(observation_space(), action_space(), "cpu")
    name = getattr(agent, "policy_name", "?")
    if not name.startswith("brain:") or "+memory" not in name:
        return False, f"DoomAgent loaded {name!r}, not the trained memory brain (random fallback?)", None
    return True, f"DoomAgent policy is {name!r}", agent


def check_plays(agent) -> tuple[bool, str]:
    from doom_env import make_env

    env = make_env(str(REPO / "maps" / "two_rooms.wad"), "MAP01", episode_ticks=2100)
    try:
        obs, _ = env.reset(seed=0)
        agent.reset()
        steps, done, info = 0, False, {}
        while not done:
            obs, _, terminated, truncated, info = env.step(agent.act(obs))
            steps += 1
            done = terminated or truncated
        exited = bool(env.privileged_state()["exit_reached"])
    finally:
        env.close()
    if not exited:
        return False, f"did not reach the two_rooms exit in {steps} decisions (the champion does it in ~82)"
    return True, f"reached the two_rooms exit in {steps} decisions"


def check_git() -> tuple[bool, str]:
    try:
        git("fetch", "-q", "origin")
        head, main = git("rev-parse", "HEAD"), git("rev-parse", "origin/main")
        dirty = git("status", "--porcelain", "--untracked-files=no")
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        return False, f"git check failed: {exc}"
    if head != main:
        return False, f"HEAD {head[:7]} is not origin/main {main[:7]}: the scorer uses the last commit on main"
    if dirty:
        return False, "uncommitted changes to tracked files"
    return True, f"HEAD = origin/main = {head[:7]}, tree clean"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-git", action="store_true")
    a = ap.parse_args(argv)
    results: list[tuple[str, bool, str]] = []
    ok, msg = check_lfs()
    results.append(("LFS content", ok, msg))
    ok, msg = check_hash()
    results.append(("champion hash", ok, msg))
    agent = None
    if ok:
        ok, msg, agent = check_agent()
        results.append(("trained brain loaded", ok, msg))
    if agent is not None:
        ok, msg = check_plays(agent)
        results.append(("plays the template map", ok, msg))
    if not a.skip_git:
        ok, msg = check_git()
        results.append(("final commit", ok, msg))
    for name, ok, msg in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {msg}")
    failed = [n for n, ok, _ in results if not ok]
    print("RELEASE CHECK " + ("PASSED" if not failed else f"FAILED ({', '.join(failed)})"))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
