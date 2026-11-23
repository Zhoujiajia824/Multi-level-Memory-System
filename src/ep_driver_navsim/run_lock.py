"""OS-owned output-directory exclusion, independent of PID files or score math."""
from contextlib import contextmanager
import json
import os

from nine_grid.paths import filesystem_path


class _Lease:
    def __init__(self, stream):
        self.stream = stream

    def child_options(self):
        # flock belongs to the open file description. On Linux the SDK keeps
        # that lock even if the launcher is killed, until its own last copy of
        # the descriptor closes. Do not explicitly LOCK_UN in the parent.
        return {"pass_fds": (self.stream.fileno(),)} if os.name != "nt" else {}


@contextmanager
def evaluation_lock(settings):
    root = filesystem_path(settings["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".ep_navsim_evaluation.lock"
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            if stream.seek(0, os.SEEK_END) == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            acquire = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            release = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            acquire = lambda: fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            release = lambda: None  # close below; inherited child may still own it
        try:
            acquire()
        except OSError:
            raise RuntimeError("Another evaluation owns this output_dir; use a different directory") from None
        try:
            # Advisory metadata is written only while holding the OS lock. A
            # stale PID never prevents reuse after the owning process exits.
            stream.seek(0)
            stream.truncate()
            stream.write(json.dumps(dict(pid=os.getpid(), scope="one output_dir only")).encode("utf-8"))
            stream.flush()
            frozen = settings.get("settings_path")
            if frozen and json.loads(filesystem_path(frozen).read_text(encoding="utf-8")) != settings:
                raise ValueError("Frozen run_settings changed before evaluation; refusing mixed settings")
            yield _Lease(stream)
        finally:
            stream.seek(0)
            release()
