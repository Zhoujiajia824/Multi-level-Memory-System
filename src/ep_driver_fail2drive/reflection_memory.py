"""Explicit, route-frozen, read-only memory for reflection only.

No default directory, online writing, migration or fallback library is used.
The original feature compatibility checks and three-channel retrieval apply.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

from nine_grid.contracts import RetrievalResult, plain
from nine_grid.memory import MemoryStore
from nine_grid.paths import filesystem_path
from nine_grid.retrieval import retrieve


ROOT = Path(__file__).resolve().parents[2]


def retrieval_record_evidence(result):
    """Only the human-readable historical content, not internal asset locations."""
    from nine_grid.models import _historical_situation
    records = []
    for record in result.records:
        item = dict(id=record['id'], revision=record.get('revision', 1))
        if record.get('record_origin') == 'offline_retrospective':
            item.update(record_origin='offline_retrospective', text=record['offline_text'])
        else:
            answer = record.get('answer', {})
            item.update(situation=_historical_situation(record.get('situation', {})),
                answer={key: answer[key] for key in
                        ('attention', 'local_goal', 'explanation', 'scene', 'decision') if key in answer})
        records.append(plain(item))
    return records


def _publication_hashes(directory):
    return {name: (hashlib.sha256((directory/name).read_bytes()).hexdigest()
                   if (directory/name).is_file() else None)
            for name in ('manifest.json', 'freeze.json')}


def _check_launch_identity(directory, expected):
    if expected is None:
        return
    if (not isinstance(expected, dict) or not isinstance(expected.get('root'), str)
            or filesystem_path(expected['root']) != directory
            or not isinstance(expected.get('metadata_sha256'), dict)
            or not expected['metadata_sha256']):
        raise ValueError('Invalid reflection memory launch identity or mismatched root')
    targets = {}
    for name in expected['metadata_sha256']:
        if not isinstance(name, str) or not name or Path(name).is_absolute():
            raise ValueError('Reflection memory identity must use relative metadata paths')
        target = filesystem_path(directory/name)
        if not target.is_relative_to(directory):
            raise ValueError('Reflection memory launch metadata escapes its root')
        targets[name] = target
    metadata_paths = [directory/name for name in ('manifest.json', 'freeze.json', 'invalidations.jsonl')
                      if (directory/name).is_file()]
    records = directory/'records'
    if records.is_dir():
        metadata_paths.extend(records.glob('*/record.json'))
    if any(not filesystem_path(path).is_relative_to(directory) for path in metadata_paths):
        raise ValueError('Reflection memory launch metadata escapes its root')
    current_names = {path.relative_to(directory).as_posix() for path in metadata_paths}
    if current_names != set(expected['metadata_sha256']):
        raise ValueError('Reflection memory metadata set changed since launch fingerprint')
    for name, digest in expected['metadata_sha256'].items():
        target = targets[name]
        if (not isinstance(digest, str) or not target.is_file()
                or hashlib.sha256(target.read_bytes()).hexdigest() != digest):
            raise ValueError('Reflection memory metadata differs from the launch fingerprint: '+name)


class ReflectionMemoryBank:
    """Load one explicitly selected library; queries never change that snapshot."""
    def __init__(self, directory, *, enabled, inactive_reason=None, store_factory=None,
                 expected_identity=None):
        self.enabled = False
        self.directory = None
        self.records = []
        self.snapshot_id = ''
        self.evidence = dict(enabled=False, configured=directory is not None,
            read_only=True, writing_enabled=False, fallback_library_used=False,
            reason=inactive_reason or ('reflection_disabled' if not enabled else 'not_configured'),
            directory=None, record_count=0, records=[], excluded_records=[])
        # A_base and reflection-off must not resolve, stat or open the supplied
        # path, even when a private configuration still contains a legacy root.
        if not enabled or directory is None:
            return
        if not isinstance(directory, (str, Path)) or not str(directory).strip():
            raise ValueError('reflection_memory_dir must be an explicit nonempty path or null')
        path = Path(directory).expanduser()
        self.directory = filesystem_path(path if path.is_absolute() else ROOT/path)
        if not self.directory.is_dir():
            raise ValueError('Configured reflection memory directory is missing or not a directory: '+str(self.directory))
        _check_launch_identity(self.directory, expected_identity)
        before = _publication_hashes(self.directory)
        store = (store_factory or MemoryStore)(self.directory, write_dir=None)
        try:
            self.records = list(store.snapshot())
        except (ValueError, OSError, KeyError, TypeError) as exc:
            raise ValueError('Cannot read configured reflection memory: '+str(exc)) from exc
        after = _publication_hashes(self.directory)
        _check_launch_identity(self.directory, expected_identity)
        if before != after:
            raise ValueError('Reflection memory publication changed while loading its snapshot')
        if not self.records:
            audit = plain(getattr(store, 'audit', []))
            raise ValueError('Configured reflection memory has no readable records; '
                             'no fallback library will be used. Exclusions: '+str(audit))
        self.enabled = True
        self.snapshot_id = store.snapshot_id
        self.evidence.update(enabled=True, reason='explicit_reflection_only_library',
            directory=str(self.directory), snapshot_id=self.snapshot_id,
            format=('offline_frozen' if any(value is not None for value in after.values())
                    else 'online_records_read_only'), publication_sha256=after,
            launch_identity_verified=expected_identity is not None,
            record_count=len(self.records),
            records=[dict(id=record['id'], revision=record.get('revision', 1)) for record in self.records],
            excluded_records=plain(getattr(store, 'audit', [])))

    def query(self, representations, config):
        if not self.enabled:
            return RetrievalResult(representations.frame_id, time=representations.time,
                snapshot_id=self.snapshot_id,
                diagnostics=dict(reason=self.evidence['reason'], memory_scope='reflection_only'))
        result = retrieve(representations, self.records, snapshot_id=self.snapshot_id, config=config)
        result.diagnostics['memory_scope'] = 'reflection_only'
        result.diagnostics['snapshot_id'] = self.snapshot_id
        return result

    def snapshot_evidence(self):
        """Archive metadata only; never serialize arrays or expose library paths to VLM."""
        return deepcopy(self.evidence)
