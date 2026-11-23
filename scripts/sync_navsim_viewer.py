"""Synchronize the workspace viewer into the deployable NAVSIM branch bundle."""
import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def selected(source):
    files = [source / name for name in ("run_viewer.py", "README.md", "requirements.txt")]
    files += list((source / "epviz").rglob("*.py"))
    files += [p for p in (source / "web").rglob("*") if p.is_file()]
    return sorted(p for p in files if p.is_file() and "__pycache__" not in p.parts
                  and not p.name.endswith((".log", ".pid", ".tmp")))


def deployment_content(original):
    """Ship runtime documentation without references to local-only tests."""
    if original.name != "README.md":
        return original.read_bytes()
    paragraphs = original.read_text(encoding="utf-8").split("\n\n")
    result = []
    in_validation = False
    for paragraph in paragraphs:
        if paragraph.startswith("## 验证"):
            in_validation = True
        elif in_validation and paragraph.startswith("## "):
            in_validation = False
        if in_validation or paragraph.startswith(("NAVSIM 回归测试：", "可选浏览器回归：")):
            continue
        result.append(paragraph)
    return ("\n\n".join(result).rstrip() + "\n").encode("utf-8")


def sync(source, check=False):
    source = Path(source).resolve()
    target = (ROOT / "tools/ep_agent_viewer").resolve()
    if not target.is_relative_to(ROOT.resolve()) or target == source:
        raise ValueError("Viewer bundle must be inside the NAVSIM worktree")
    files = selected(source)
    if not (source / "run_viewer.py").is_file() or not files:
        raise ValueError("Expected the standalone ep_agent_viewer source")
    changed = []
    for original in files:
        output = target / original.relative_to(source)
        content = deployment_content(original)
        digest = hashlib.sha256(content).digest()
        if not output.is_file() or hashlib.sha256(output.read_bytes()).digest() != digest:
            changed.append(original.relative_to(source).as_posix())
            if not check:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(content)
    # Deliberately never delete files in either directory.
    print("Viewer files:", len(files), "changed:", len(changed), "check:", check)
    if check and changed:
        raise RuntimeError("Viewer bundle is out of date: " + ", ".join(changed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(ROOT.parent.parent / "ep_agent_viewer"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    sync(args.source, args.check)
