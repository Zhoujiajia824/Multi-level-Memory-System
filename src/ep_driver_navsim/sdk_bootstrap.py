"""Platform bootstrap; executes the pinned official entry point unchanged."""
import os
import pathlib
import runpy
import sys
import types


def platform_setup():
    if os.name == "nt":
        # Official Linux-produced scene pickles contain pathlib.PosixPath.
        # Interpret serialized paths using native Windows filesystem semantics.
        # No trajectory, annotation, map geometry or score is transformed here.
        pathlib.PosixPath = pathlib.WindowsPath
    if os.name == "nt" and "fcntl" not in sys.modules:
        # nuPlan imports fcntl unconditionally, although complete local maps do
        # not use it. Refuse any attempt to download/lock rather than pretend
        # that Windows provides POSIX advisory locks. Scoring math is untouched.
        module = types.ModuleType("fcntl")
        module.LOCK_EX, module.LOCK_UN = 2, 8
        def unsupported_lock(*args, **kwargs):
            raise RuntimeError("nuPlan map downloading/locking requires Linux; supply complete local maps")
        module.flock = unsupported_lock
        sys.modules["fcntl"] = module
    return {"platform": sys.platform, "windows_import_guard": os.name == "nt",
            "serialized_posix_paths_use_native_path": os.name == "nt",
            "scoring_source_modified": False}


if __name__ == "__main__":
    platform_setup()
    entry = sys.argv.pop(1)
    sys.argv[0] = entry
    replay_settings = os.environ.get("EP_NAVSIM_REPLAY_SETTINGS")
    if replay_settings:
        import json
        from .score_replay import install_score_replay
        install_score_replay(json.loads(pathlib.Path(replay_settings).read_text(encoding="utf-8")))
    runpy.run_path(entry, run_name="__main__")
