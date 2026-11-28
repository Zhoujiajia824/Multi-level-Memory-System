"""Opt-in ownership of one local collection simulator, never an existing service."""
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time

PLATFORM = os.name


def _process_rows():
    if not Path('/proc').is_dir():
        raise RuntimeError('Owned process-group cleanup requires Linux /proc')
    rows = []
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            raw = path.read_text()
            fields = raw[raw.rfind(')')+2:].split()
            rows.append(dict(pid=int(path.parent.name), state=fields[0], group=int(fields[2]),
                             session=int(fields[3]), birth=int(fields[19])))
        except (OSError, ValueError, IndexError):
            continue
    return rows


class SimulatorManagementError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class DedicatedSimulator:
    def __init__(self, root, log_directory, health, *, host, port, gpu_uuid, gpu_pci,
                 restart_limit=8, startup_timeout=300.):
        self.root, self.logs = Path(root).resolve(), Path(log_directory)
        self.health, self.host, self.port = health, host, int(port)
        if host not in ('127.0.0.1', 'localhost') or not 1024 <= self.port <= 65533:
            raise ValueError('Managed CARLA requires a local host and three unprivileged ports')
        if not isinstance(gpu_uuid, str) or not re.fullmatch(
                r'GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', gpu_uuid, re.I):
            raise ValueError('Managed CARLA requires an explicit NVIDIA GPU UUID')
        pci = re.fullmatch(r'(?:pci-)?([0-9a-f]{4})[:_]([0-9a-f]{2})[:_]([0-9a-f]{2})[._]([0-7])',
                           gpu_pci or '', re.I)
        if pci is None:
            raise ValueError('Managed CARLA requires the explicit GPU PCI address, e.g. pci-0000_c1_00_0')
        if not 0 <= restart_limit <= 68 or not 0 < startup_timeout <= 900:
            raise ValueError('Managed CARLA restart/startup limits are out of range')
        if not (self.root / 'CarlaUE4.sh').is_file():
            raise ValueError('Managed CARLA root must contain CarlaUE4.sh')
        self.gpu_uuid = gpu_uuid
        self.gpu_pci = 'pci-' + '_'.join(part.lower() for part in pci.groups())
        self.restart_limit, self.startup_timeout = restart_limit, startup_timeout
        self.restarts, self.launches, self.process, self.handle = 0, 0, None, None
        self.group, self.birth = None, None

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        try:
            self.stop()
        except Exception as cleanup:
            self.event('stop_failed', error=str(cleanup))
            if error is None:
                raise

    def event(self, name, **details):
        self.logs.mkdir(parents=True, exist_ok=True)
        with (self.logs / 'events.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(dict(event=name, wall_time=time.time(),
                restarts=self.restarts, launches=self.launches, **details), ensure_ascii=False) + '\n')

    def _ports_free(self):
        for port in range(self.port, self.port + 3):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
                connection.settimeout(.2)
                if connection.connect_ex(('127.0.0.1', port)) == 0:
                    raise SimulatorManagementError('simulator_port_in_use',
                        f'Port {port} already has a listener; existing services will not be taken over or stopped')
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    probe.bind(('127.0.0.1', port))
                except OSError as error:
                    raise SimulatorManagementError('simulator_port_in_use',
                        f'Port {port} is occupied; existing services will not be taken over or stopped') from error

    def _device(self):
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=uuid,pci.bus_id,memory.free,memory.total',
                '--format=csv,noheader,nounits'], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                encoding='utf-8', errors='replace', timeout=5., check=True)
            for line in result.stdout.splitlines():
                fields = [value.strip() for value in line.split(',')]
                if len(fields) != 4 or fields[0].lower() != self.gpu_uuid.lower():
                    continue
                domain, bus, device = fields[1].lower().split(':')
                normalized = f'pci-{int(domain,16):04x}_{bus}_{device.replace(".","_")}'
                if normalized != self.gpu_pci:
                    raise ValueError('The GPU UUID and PCI address identify different devices')
                return dict(uuid=fields[0], pci=self.gpu_pci, memory_free_mib=fields[2],
                            memory_total_mib=fields[3], memory_threshold_applied=False)
            raise ValueError('The requested GPU UUID was not reported by nvidia-smi')
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            raise SimulatorManagementError('simulator_gpu_identity_unverified', str(error)) from error

    def start(self):
        if PLATFORM != 'posix':
            raise SimulatorManagementError('managed_simulator_platform', 'Managed CarlaUE4.sh requires Linux')
        if self.process is not None and self.process.poll() is None:
            raise SimulatorManagementError('simulator_already_owned', 'The owned simulator is already running')
        self._ports_free()
        device = self._device()
        self.logs.mkdir(parents=True, exist_ok=True)
        indices = [int(p.stem.split('_')[-1]) for p in self.logs.glob('start_*.log')
                   if p.stem.split('_')[-1].isdecimal()]
        self.launches = max(indices, default=0) + 1
        log = self.logs / f'start_{self.launches:03d}.log'
        env = dict(os.environ)
        for key in ('DISPLAY', 'NODEVICE_SELECT', 'MESA_VK_DEVICE_SELECT'):
            env.pop(key, None)
        env.update(CUDA_VISIBLE_DEVICES=self.gpu_uuid, DRI_PRIME=self.gpu_pci,
            VK_INSTANCE_LAYERS='VK_LAYER_MESA_device_select', MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE='1')
        command = [str(self.root / 'CarlaUE4.sh'), '-RenderOffScreen', '-quality-level=Epic', '-nosound',
            '-graphicsadapter=0', f'-carla-rpc-port={self.port}', f'-carla-streaming-port={self.port+1}',
            f'-carla-secondary-port={self.port+2}', '-carla-primary-port=0']
        self.handle = log.open('wb')
        try:
            self.process = subprocess.Popen(command, cwd=str(self.root), env=env, stdin=subprocess.DEVNULL,
                stdout=self.handle, stderr=subprocess.STDOUT, start_new_session=True)
            self.group = self.process.pid
            self.birth = next((row['birth'] for row in _process_rows() if row['pid'] == self.group), None)
            self.event('started', pid=self.process.pid, log=log.name, command=command,
                gpu_uuid=self.gpu_uuid, gpu_pci=self.gpu_pci, device=device)
            deadline = time.monotonic() + self.startup_timeout
            last = None
            while time.monotonic() < deadline:
                code = self.process.poll()
                if code is not None:
                    raise SimulatorManagementError('owned_simulator_exited',
                        f'Owned CARLA exited with code {code}; see {log}')
                remaining = deadline - time.monotonic()
                last = self.health(self.host, self.port, min(5., max(.1, remaining)))
                if last['ready']:
                    self.event('ready', pid=self.process.pid, health=last)
                    return
                if last.get('failure_code') == 'simulator_version_mismatch':
                    raise SimulatorManagementError('simulator_version_mismatch', last['error'])
                time.sleep(min(1., max(0., deadline-time.monotonic())))
            raise SimulatorManagementError('simulator_startup_timeout',
                f'Owned CARLA did not become RPC-ready within {self.startup_timeout:g}s; see {log}; {last}')
        except BaseException as error:
            self.event('start_failed', error=type(error).__name__ + ': ' + str(error))
            try:
                self.stop()
            except Exception as cleanup:
                self.event('stop_failed', error=str(cleanup), original_error=str(error))
            raise

    def prepare(self):
        if self.process is None:
            self.start()
        elif self.process.poll() is not None:
            self.restart('owned simulator exited before the next capture')

    def restart(self, reason):
        if self.restarts >= self.restart_limit:
            raise SimulatorManagementError('simulator_restart_budget_exhausted',
                f'Owned CARLA restart limit {self.restart_limit} exhausted')
        self.event('restart_requested', reason=reason)
        self.stop()
        self.restarts += 1
        self.start()

    def _members(self):
        rows = _process_rows()
        leader = next((row for row in rows if row['pid'] == self.group), None)
        if leader and ((self.birth is None or leader['birth'] != self.birth)
                       or leader['group'] != self.group or leader['session'] != self.group):
            raise SimulatorManagementError('simulator_ownership_unverified', 'Owned launcher identity changed')
        members = [row for row in rows if row['group'] == self.group and row['state'] != 'Z']
        if leader is None and not members and self.process.poll() is None:
            raise SimulatorManagementError('simulator_ownership_unverified', 'Live owned launcher absent from /proc')
        if any(row['session'] != self.group or (self.birth is not None and row['birth'] < self.birth)
               for row in members):
            raise SimulatorManagementError('simulator_ownership_unverified', 'Process-group ownership changed')
        return members

    def stop(self):
        if self.group is not None:
            for value, grace in ((signal.SIGINT, 5.), (signal.SIGTERM, 3.), (getattr(signal, 'SIGKILL', 9), 3.)):
                self.process.poll()  # Reap the launcher, but also check surviving engine children.
                if not self._members():
                    break
                try:
                    os.killpg(self.group, value)
                except ProcessLookupError:
                    pass
                deadline = time.monotonic()+grace
                while self._members() and time.monotonic() < deadline:
                    self.process.poll()
                    time.sleep(.05)
            if self._members():
                raise SimulatorManagementError('owned_simulator_stop_failed', 'Owned CARLA process group did not exit')
            self.event('stopped', pid=self.group, exit_code=self.process.poll())
            self.group, self.birth = None, None
        if self.handle is not None:
            self.handle.close()
            self.handle = None
