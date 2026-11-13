#!/usr/bin/env python3
"""Build-independent supervisor for sync.sh --test; JSON is the raw evidence."""
import argparse
import collections
import contextlib
import datetime
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import pwd
import re
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time

CASES = [(30, False, False, False), (50, False, False, False), (70, False, False, False),
         (50, True, True, False), (50, True, True, True)]
CONFIG = {'version': 1, 'mode': '1920x1080@60 SDR', 'fps': 60, 'warmup_seconds': 10,
          'case_seconds': 8, 'mouse_rate': 240, 'mouse_radius': 160, 'mouse_period': 2,
          'cursor_hide_delay_ms': 600000,
          'ambient_capabilities': ['CAP_SYS_NICE'],
          'night': [.75, .06944445, .95], 'cases': CASES,
          'mangohud': 'fps,frametime,cpu_stats,frame_timing,position=top-left,font_size=20,fps_sampling_period=500'}


def run(args, **kwargs):
    return subprocess.run(args, check=True, text=True, **kwargs)


def output(args):
    return subprocess.check_output(args, text=True).strip()


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def machine_state():
    patterns = ['/sys/class/thermal/thermal_zone*/temp', '/sys/devices/system/cpu/cpufreq/policy*/scaling_governor',
                '/sys/devices/system/cpu/cpufreq/policy*/scaling_cur_freq', '/sys/class/devfreq/*/cur_freq']
    result = {}
    for pattern in patterns:
        for filename in glob.glob(pattern):
            with contextlib.suppress(OSError):
                result[filename] = Path(filename).read_text().strip()
    return result


def connected_displays():
    result = {}
    for status in Path('/sys/class/drm').glob('card*-*/status'):
        if status.read_text().strip() == 'connected':
            edid = (status.parent / 'edid').read_bytes()
            if edid:
                result[status.parent.name] = hashlib.sha256(edid).hexdigest()
    return result


def active_modes():
    result = []
    seen = set()
    for state in Path('/sys/kernel/debug/dri').glob('*/state'):
        identity = (state.stat().st_dev, state.stat().st_ino)
        if identity in seen:
            continue
        seen.add(identity)
        for block in re.split(r'(?=^crtc\[)', state.read_text(), flags=re.M):
            if block.startswith('crtc[') and re.search(r'^\s+active=1$', block, re.M):
                mode = re.search(r'mode: "([^"]+)": (\d+)', block)
                if mode:
                    result.append([mode[1], int(mode[2])])
    return result


def process(pid):
    line = Path(f'/proc/{pid}/stat').read_text()
    fields = line.rsplit(')', 1)[1].split()
    return {'pid': pid, 'comm': line.split('(', 1)[1].rsplit(')', 1)[0], 'parent': int(fields[1]),
            'ticks': int(fields[11]) + int(fields[12]), 'start': int(fields[19])}


def descendants(pid):
    processes = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        with contextlib.suppress(OSError, ValueError):
            p = process(int(path.parent.name)); processes[p['pid']] = p
    selected = {pid}
    while True:
        found = {p['pid'] for p in processes.values() if p['parent'] in selected}
        if found <= selected:
            return [p for n, p in processes.items() if n in selected]
        selected |= found


def gpu_counters(pid):
    clients = {}
    for info in Path(f'/proc/{pid}/fdinfo').glob('*'):
        try:
            text = info.read_text()
            driver = re.search(r'^drm-driver:\s*(\S+)', text, re.M)
            client = re.search(r'^drm-client-id:\s*(\d+)', text, re.M)
            if not driver or not client:
                continue
            # Duplicate fds may reference the same DRM client.
            device = os.readlink(f'/proc/{pid}/fd/{info.name}')
            key = (device, driver[1], client[1])
            values = {k: int(v) for k, v in re.findall(r'^(drm-engine-[^:]+):\s*(\d+)\s+ns', text, re.M)}
            values.update({k: int(v) for k, v in re.findall(r'^(v3d-jobs-[^:]+):\s*(\d+)', text, re.M)})
            clients['|'.join(key)] = values
        except (OSError, ValueError):
            continue
    return clients


def snapshot(groups):
    result = {}
    for name, pids in groups.items():
        result[name] = []
        for pid in pids:
            p = process(pid)
            p['gpu'] = gpu_counters(pid)
            result[name].append(p)
    return result


def summarize(before, after, seconds, events):
    result = {}
    for name, processes in after.items():
        old = {p['pid']: p for p in before[name]}
        ticks = 0; engines = collections.defaultdict(lambda: {'ns': 0, 'jobs': None})
        for p in processes:
            previous = old[p['pid']]
            if p['start'] != previous['start']:
                raise RuntimeError('process restarted during measurement')
            ticks += p['ticks'] - previous['ticks']
            if set(p['gpu']) != set(previous['gpu']):
                raise RuntimeError('DRM clients changed during measurement')
            for client, counters in p['gpu'].items():
                for key, value in counters.items():
                    delta = value - previous['gpu'][client].get(key, value)
                    if delta < 0:
                        raise RuntimeError('GPU counter moved backwards')
                    if key.startswith('drm-engine-'):
                        engines[key[len('drm-engine-'):]]['ns'] += delta
                    elif key.startswith('v3d-jobs-'):
                        engine = engines[key[len('v3d-jobs-'):]]
                        engine['jobs'] = (engine['jobs'] or 0) + delta
        result[name] = {'cpu_percent': 100 * ticks / os.sysconf('SC_CLK_TCK') / seconds, 'engines': {}}
        for name2, e in sorted(engines.items()):
            result[name]['engines'][name2] = {**e, 'ms_per_second': e['ns'] / 1e6 / seconds,
                'jobs_per_second': e['jobs'] / seconds if e['jobs'] is not None else None,
                'ms_per_job': e['ns'] / 1e6 / e['jobs'] if e['jobs'] else None}
    intervals = [(b['time'] - a['time']) * 1000 for a, b in zip(events, events[1:])]
    gaps = collections.Counter((b['seq'] - a['seq']) % (1 << 32) for a, b in zip(events, events[1:]))
    if len(intervals) < 2:
        raise RuntimeError('too few KMS presentations')
    intervals.sort()
    return {'processes': result, 'presentation_fps': len(events) / seconds,
            'median_ms': statistics.median(intervals), 'p95_ms': intervals[math.ceil(.95 * len(intervals)) - 1],
            'sequence_gaps': dict(gaps), 'missed_refreshes': sum((gap - 1) * count for gap, count in gaps.items() if gap > 1)}


class Trace:
    def __init__(self, name):
        self.path = Path('/sys/kernel/tracing/instances') / name
        self.path.mkdir()
        (self.path / 'tracing_on').write_text('0')
        (self.path / 'trace_clock').write_text('mono')
        (self.path / 'buffer_size_kb').write_text('2048')
        for event in ('drm_vblank_event_queued', 'drm_vblank_event_delivered'):
            (self.path / f'events/drm/{event}/enable').write_text('1')

    def start(self):
        (self.path / 'trace').write_text('')
        (self.path / 'tracing_on').write_text('1')

    def stop(self, tids):
        (self.path / 'tracing_on').write_text('0')
        text = (self.path / 'trace').read_text()
        for stat in (self.path / 'per_cpu').glob('cpu*/stats'):
            if re.search(r'(?:^|\n)overrun:\s*[1-9]', stat.read_text()):
                raise RuntimeError('trace buffer overran')
        pattern = r'-(\d+)\s+\[\d+\].*?\s(\d+\.\d+): drm_vblank_event_(queued|delivered): file=(\S+), crtc=(\d+), seq=(\d+)'
        records = [{'pid': int(p), 'time': float(t), 'event': e, 'file': f, 'crtc': int(c), 'seq': int(s)}
                   for p, t, e, f, c, s in re.findall(pattern, text)]
        clients = {(e['file'], e['crtc']) for e in records if e['event'] == 'queued' and e['pid'] in tids}
        if not clients:
            # Atomic page flips may emit delivered events without queued events.
            # The isolated session must have exactly one presenting DRM client.
            clients = {(e['file'], e['crtc']) for e in records if e['event'] == 'delivered'}
            if len(clients) != 1:
                raise RuntimeError('cannot identify a unique KMS presentation stream')
        events = [e for e in records if e['event'] == 'delivered' and (e['file'], e['crtc']) in clients]
        return events, text

    def close(self):
        (self.path / 'tracing_on').write_text('0')
        for event in ('drm_vblank_event_queued', 'drm_vblank_event_delivered'):
            (self.path / f'events/drm/{event}/enable').write_text('0')
        self.path.rmdir()


def command(stage, text):
    with socket.socket(socket.AF_UNIX) as sock:
        sock.settimeout(3)
        sock.connect(str(stage / 'control.sock'))
        sock.sendall((text + '\n').encode())
        reply = b''
        while not reply.endswith(b'\n'):
            part = sock.recv(4096)
            if not part:
                raise RuntimeError('fixture disconnected')
            reply += part
        result = json.loads(reply)
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result


def report(data):
    lines = [f"Gamescope benchmark: {data['status'].upper()}"]
    if data.get('diagnostic'):
        lines.append('DIAGNOSTIC: Wayland protocol logging enabled; timings are not comparable.')
    for key in ('host', 'revision', 'dirty_diff_sha256', 'binary_sha256'):
        if key in data:
            lines.append(f'{key}: {data[key]}')
    config = data.get('configuration', CONFIG)
    lines += [f"Mode: {config['mode']}; Xwayland; no Steam",
              f"Timing: MangoHud first frame -> {config['warmup_seconds']}s warmup once -> "
              f"{len(config['cases'])} x {config['case_seconds']}s cases",
              f"Animation: {config['fps']} FPS; pointer: {config['mouse_rate']} Hz, "
              f"{config['mouse_radius']}px radius, {config['mouse_period']}s period; "
              f"cursor hide: {config['cursor_hide_delay_ms']}ms",
              'Night amount/hue/saturation: ' + '/'.join(map(str, config['night'])),
              'MangoHud: ' + config['mangohud']]
    states = [data.get('machine_before', {}), data.get('machine_after', {})]
    states += [case.get('machine', {}) for case in data.get('results', [])]
    temperatures = []; cpu_clocks = []; gpu_clocks = []; governors = set()
    for state in states:
        for path, value in state.items():
            if path.endswith('/temp'):
                temperatures.append(float(value) / 1000)
            elif path.endswith('/scaling_cur_freq'):
                cpu_clocks.append(float(value) / 1000)
            elif path.endswith('/cur_freq'):
                gpu_clocks.append(float(value) / 1e6)
            elif path.endswith('/scaling_governor'):
                governors.add(value)
    def span(values, unit):
        return f'{min(values):.1f}-{max(values):.1f}{unit}' if values else 'unavailable'
    if any(states):
        lines.append(f"Telemetry: temperature {span(temperatures, 'C')}; CPU {span(cpu_clocks, 'MHz')}; "
                     f"GPU {span(gpu_clocks, 'MHz')}; governor {','.join(sorted(governors)) or 'unavailable'}")
    lines += ['CPU: % of one core. GPU: per-engine busy ms/s, not frame latency.', '']
    lines += ['Damage HUD Mouse Night | FPS  median/p95 ms  missed | CPU G/X/H/App % | GPU ms/s (G)',
              '-----------------------|--------------------------------|-----------------|-------------']
    for case in data.get('results', []):
        s = case['summary']; processes = s['processes']
        cpu = '/'.join(f"{processes[n]['cpu_percent']:.1f}" for n in ('gamescope', 'xwayland', 'mangoapp', 'app'))
        engines = processes['gamescope']['engines']
        gpu = ', '.join(f"{k}={v['ms_per_second']:.1f}" for k, v in engines.items() if v['ns']) or 'idle'
        flags = ' '.join('on ' if case[k] else 'off' for k in ('overlay', 'mouse', 'night'))
        lines.append(f"{case['damage']:>3}%   {flags} | {s['presentation_fps']:5.1f}  {s['median_ms']:5.1f}/{s['p95_ms']:5.1f} {s['missed_refreshes']:7} | {cpu} | {gpu}")
        jobs = ', '.join(f"{k}: {v['jobs_per_second']:.1f} jobs/s, " +
                         (f"{v['ms_per_job']:.2f} ms/job" if v['ms_per_job'] is not None else 'n/a ms/job')
                         for k, v in engines.items() if v['jobs'])
        if jobs:
            lines.append('    ' + jobs)
    if data.get('error'):
        lines += ['', 'ERROR: ' + data['error']]
    if data.get('cleanup_errors'):
        lines += ['', 'Cleanup: ' + '; '.join(data['cleanup_errors'])]
    if data.get('damage_diagnostic'):
        lines += ['', 'Wayland damage rectangles (% of screen, count): ' +
                  ', '.join(f'{key}% x{count}' for key, count in data['damage_diagnostic'].items())]
    return '\n'.join(lines) + '\n'


def remote(args):
    from benchmark_mouse import Mouse
    stage = Path(args.stage).resolve()
    account = pwd.getpwnam(args.user)
    data = json.loads((stage / 'metadata.json').read_text())
    data.update(status='failed', results=[], host=socket.gethostname(), configuration=CONFIG,
                started=datetime.datetime.now(datetime.timezone.utc).isoformat(), machine_before=machine_state())
    data['kernel'] = output(['uname', '-r'])
    if Path('/usr/bin/rpm').exists():
        data['packages'] = subprocess.run(['rpm', '-q', 'mesa-vulkan-drivers', 'mesa-dri-drivers', 'mangohud'],
                                          text=True, capture_output=True).stdout.splitlines()
    unit = f'gamescope-test-{os.getpid()}'
    restore_unit = unit + '-restore'
    previous = output(['systemctl', 'is-active', 'plasmalogin']) if subprocess.run(['systemctl', 'is-active', '--quiet', 'plasmalogin']).returncode == 0 else 'inactive'
    mouse = trace = connection = service = logger = None
    (stage / 'gamescope.log').touch()
    heartbeat = {'time': time.monotonic(), 'eof': False}
    interrupted = threading.Event()
    display_reference = None
    display_check_time = 0

    def heartbeats():
        for line in sys.stdin:
            heartbeat['time'] = time.monotonic()
        heartbeat['eof'] = True
    threading.Thread(target=heartbeats, daemon=True).start()
    for sig in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: interrupted.set())

    def alive():
        nonlocal display_check_time
        if interrupted.is_set() or heartbeat['eof'] or time.monotonic() - heartbeat['time'] > 10:
            raise RuntimeError('interrupted or SSH heartbeat lost')
        if service and service.poll() is not None:
            raise RuntimeError('Gamescope exited prematurely; see gamescope.log')
        if mouse and not mouse.thread.is_alive():
            raise RuntimeError('mouse input worker stopped')
        if display_reference is not None and time.monotonic() - display_check_time > 1:
            display_check_time = time.monotonic()
            if connected_displays() != display_reference:
                raise RuntimeError('display disconnected or EDID changed during the run')

    def pause(seconds, observer=None):
        end = time.monotonic() + seconds
        next_observation = 0
        while time.monotonic() < end:
            alive()
            if observer and time.monotonic() >= next_observation:
                observer()
                next_observation = time.monotonic() + .25
            time.sleep(min(.05, max(0, end - time.monotonic())))

    def wait_for(predicate, description, timeout=30):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            alive()
            value = predicate()
            if value:
                return value
            time.sleep(.1)
        raise RuntimeError('timed out waiting for ' + description)

    def child_command(arguments):
        return run(['runuser', '-u', args.user, '--', 'env', f'XDG_RUNTIME_DIR=/run/user/{account.pw_uid}',
                    f'DISPLAY={fixture["display"]}', *arguments], cwd=stage, stdout=subprocess.DEVNULL)

    def overlay_window():
        atom = connection.intern_atom('GAMESCOPE_EXTERNAL_OVERLAY')
        for w in connection.screen().root.query_tree().children:
            prop = w.get_full_property(atom, X.AnyPropertyType)
            if prop and len(prop.value) and prop.value[0] and w.get_attributes().map_state == X.IsViewable:
                return w
        return None

    def overlay_visible():
        window = overlay_window()
        if window is None:
            return False
        geometry = window.get_geometry()
        try:
            image = window.get_image(0, 0, geometry.width, geometry.height, X.ZPixmap, 0xffffffff)
        except xerror.BadMatch:
            return False  # Unmapped or remapping between the two requests.
        # mangoapp uses an ARGB32 window. Nonzero alpha confirms actual HUD pixels.
        if image.depth != 32:
            raise RuntimeError('MangoHud window is not ARGB32')
        return any(image.data[3::4])

    def state(damage, hud, moving, night):
        result = command(stage, f'state {damage} {int(night)}')
        if result['damage_percent'] != damage or result['night'] != night:
            raise RuntimeError('fixture state did not apply')
        child_command(['mangohudctl', 'set', 'no_display', 'false' if hud else 'true'])
        wait_for(lambda: overlay_visible() == hud, 'MangoHud visibility', 5)
        mouse.enabled = moving
        pause(1)  # Keep state-change redraws outside the measured interval.
        return result

    try:
        from Xlib import X, display, error as xerror
        if os.geteuid() != 0:
            raise RuntimeError('remote benchmark supervisor requires root')
        if previous == 'active':
            run(['systemd-run', '--quiet', '--unit=' + restore_unit, '--on-active=150',
                 '--timer-property=AccuracySec=1s', '/usr/bin/systemctl', 'start', 'plasmalogin'])
        # ExecStopPost restores the normal session even if the supervisor is killed.
        run(['systemctl', 'stop', 'plasmalogin'])
        pause(2)
        stable = {'value': None, 'since': time.monotonic()}
        def stable_display():
            value = connected_displays()
            if not value or value != stable['value']:
                stable.update(value=value, since=time.monotonic())
                return False
            return value if time.monotonic() - stable['since'] >= 1 else False
        display_reference = wait_for(stable_display, 'a stable connected display')
        data['displays'] = display_reference
        log = open(stage / 'gamescope.log', 'w')
        arguments = ['systemd-run', '--quiet', '--unit=' + unit, '--collect', '--wait', '--pipe',
                     '-p', 'User=' + args.user, '-p', 'PAMName=login', '-p', 'TTYPath=/dev/tty1',
                     '-p', 'StandardInput=tty', '-p', 'RuntimeMaxSec=120', '-p', 'TimeoutStopSec=5',
                     '-p', 'AmbientCapabilities=CAP_SYS_NICE',
                     '-p', 'WorkingDirectory=' + str(stage)]
        if previous == 'active':
            arguments += ['-p', 'ExecStopPost=+/usr/bin/systemctl start plasmalogin']
        for setting in [f'XDG_RUNTIME_DIR=/run/user/{account.pw_uid}', 'XDG_SESSION_TYPE=wayland',
                        'XDG_SEAT=seat0', 'XDG_VTNR=1', 'LIBSEAT_BACKEND=logind',
                        'XDG_CONFIG_HOME=' + str(stage / 'config'), 'MANGOHUD_CONFIG=' + CONFIG['mangohud']]:
            arguments += ['--setenv=' + setting]
        if data.get('diagnostic'):
            arguments += ['--setenv=WAYLAND_DEBUG=server']
        arguments += ['/usr/bin/env', str(stage / 'gamescope'), '--backend', 'drm', '-w', '1920', '-h', '1080',
                      '-W', '1920', '-H', '1080', '-r', '60', '--cursor-scale-height', '1080',
                      '--hide-cursor-delay', str(CONFIG['cursor_hide_delay_ms']),
                      '--mangoapp', '--', str(stage / 'gamescope-damage'), '--duration', '110', '--control',
                      str(stage / 'control.sock'), '--ready', str(stage / 'ready.json')]
        data['launch_command'] = arguments
        # systemd-run --pipe cannot pass a regular output file as a service fd.
        service = subprocess.Popen(arguments, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        def save_log():
            with log:
                for line in service.stdout:
                    log.write(line); log.flush()
        logger = threading.Thread(target=save_log, daemon=True)
        logger.start()
        startup_deadline = time.monotonic() + 30
        fixture = wait_for(lambda: json.loads((stage / 'ready.json').read_text()) if (stage / 'ready.json').exists() else None,
                           'benchmark window', max(0, startup_deadline - time.monotonic()))
        if (fixture['width'], fixture['height']) != (1920, 1080):
            raise RuntimeError('unexpected Xwayland dimensions')
        os.seteuid(account.pw_uid)
        try:
            connection = display.Display(fixture['display'])
        finally:
            os.seteuid(0)
        overlay = wait_for(overlay_window, 'MangoHud external-overlay window', max(0, startup_deadline - time.monotonic()))
        wait_for(overlay_visible, 'MangoHud first rendered frame', max(0, startup_deadline - time.monotonic()))
        data['active_modes'] = active_modes()
        if data['active_modes'] != [['1920x1080', 60]]:
            raise RuntimeError('unexpected active KMS modes: ' + repr(data['active_modes']))
        print('MangoHud rendered; warming up for 10 seconds', flush=True)
        mainpid = int(output(['systemctl', 'show', unit, '-p', 'MainPID', '--value']))
        mouse = Mouse(rate=CONFIG['mouse_rate'], radius=CONFIG['mouse_radius'], period=CONFIG['mouse_period'])
        state(50, True, True, True)
        pause(CONFIG['warmup_seconds'])
        children = descendants(mainpid)
        xwayland = []
        for path in Path('/proc').glob('[0-9]*/cmdline'):
            with contextlib.suppress(OSError, ValueError):
                pid = int(path.parent.name)
                argv = path.read_bytes().split(b'\0')
                if process(pid)['comm'] == 'Xwayland' and fixture['display'].encode() in argv and path.stat().st_uid == account.pw_uid:
                    xwayland.append(pid)
        overlay_pid = overlay.get_full_property(connection.intern_atom('_NET_WM_PID'), X.AnyPropertyType)
        groups = {'gamescope': [mainpid], 'app': [fixture['pid']],
                  'xwayland': xwayland,
                  'mangoapp': [int(overlay_pid.value[0])] if overlay_pid and len(overlay_pid.value)
                              else [p['pid'] for p in children if p['comm'] == 'mangoapp']}
        data['process_groups'] = groups
        data['discovered_processes'] = children
        if any(not value for value in groups.values()):
            raise RuntimeError('missing benchmark process group: ' + repr(groups))
        if not any(gpu_counters(mainpid).values()):
            raise RuntimeError('GPU fdinfo timing counters unavailable')
        data['process_groups'] = groups
        trace = Trace(unit)
        tids = {int(p.name) for p in Path(f'/proc/{mainpid}/task').iterdir()}
        for index, (damage, hud, moving, night) in enumerate(CASES, 1):
            fixture = state(damage, hud, moving, night)
            print(f'Case {index}/5: damage={damage}% overlay={hud} mouse={moving} night={night}', flush=True)
            initial = command(stage, 'status'); before = snapshot(groups); pointer = connection.screen().root.query_pointer()
            mouse_before = mouse.events
            pointer_samples = []
            def observe_pointer():
                p = connection.screen().root.query_pointer()
                pointer_samples.append([time.monotonic(), p.root_x, p.root_y])
            trace.start(); start = time.monotonic()
            pause(CONFIG['case_seconds'], observe_pointer)
            end = time.monotonic(); events, trace_text = trace.stop(tids)
            (stage / f'case-{index}.trace').write_text(trace_text)
            after = snapshot(groups); final = command(stage, 'status')
            if final['frames'] == initial['frames']:
                raise RuntimeError('fixture did not animate')
            case = {'damage': damage, 'overlay': hud, 'mouse': moving, 'night': night, 'seconds': end - start,
                    'fixture': fixture, 'frames_submitted': final['frames'] - initial['frames'],
                    'mouse_events': mouse.events - mouse_before, 'pointer_start': [pointer.root_x, pointer.root_y],
                    'pointer_samples': pointer_samples,
                    'before': before, 'after': after, 'presentations': events,
                    'summary': summarize(before, after, end - start, [e for e in events if start <= e['time'] <= end]),
                    'machine': machine_state()}
            if moving and not case['mouse_events']:
                raise RuntimeError('mouse events missing')
            moved = len({tuple(p[1:]) for p in pointer_samples}) > 1
            if moved != moving:
                raise RuntimeError('observed pointer movement does not match the case')
            (stage / f'case-{index}.trace').write_text(trace_text)
            data['results'].append(case)
            atomic_json(stage / 'result.json', data)
        data['status'] = 'complete'
    except Exception as error:
        data['error'] = f'{type(error).__name__}: {error}'
        print(data['error'], file=sys.stderr, flush=True)
    finally:
        errors = []
        for action in [lambda: trace.close() if trace else None,
                       lambda: mouse.close() if mouse else None,
                       lambda: connection.close() if connection else None]:
            try:
                action()
            except Exception as error:
                errors.append(str(error))
        subprocess.run(['systemctl', 'stop', unit], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if service:
            with contextlib.suppress(subprocess.TimeoutExpired):
                service.wait(timeout=8)
        if logger:
            logger.join(timeout=2)
        if previous == 'active':
            if subprocess.run(['systemctl', 'start', 'plasmalogin']).returncode:
                errors.append('could not restore plasmalogin')
            subprocess.run(['systemctl', 'stop', restore_unit + '.timer', restore_unit + '.service'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if errors:
            data['cleanup_errors'] = errors; data['status'] = 'failed'
        data['machine_after'] = machine_state()
        if data.get('diagnostic') and (stage / 'gamescope.log').exists():
            protocol = (stage / 'gamescope.log').read_text(errors='replace')
            boxes = re.findall(r'wl_surface(?:@|#)\d+\.damage(?:_buffer)?\(\s*(-?\d+),\s*(-?\d+),\s*(\d+),\s*(\d+)\)', protocol)
            areas = collections.Counter(round(100 * int(w) * int(h) / (1920 * 1080), 1)
                                        for _, _, w, h in boxes if .2 <= int(w) * int(h) / (1920 * 1080) <= .9)
            data['damage_diagnostic'] = dict(sorted(areas.items()))
            if not areas and not data.get('error'):
                data.update(status='failed', error='no partial client damage found in the Wayland protocol log')
        atomic_json(stage / 'result.json', data)
        (stage / 'result.txt').write_text(report(data))
        for filename in ('result.json', 'result.txt', 'gamescope.log'):
            os.chown(stage / filename, account.pw_uid, account.pw_gid)
    return 0 if data['status'] == 'complete' else 1


def local(args):
    destination = Path(args.result).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = {'status': 'failed', 'results': [], 'host': args.host, 'configuration': CONFIG,
            'diagnostic': args.diagnose_damage}
    stage = None; ssh = None; stopped = threading.Event()
    try:
        data['revision'] = output(['git', 'rev-parse', 'HEAD'])
        diff = subprocess.check_output(['git', 'diff', '--binary', 'HEAD', '--', 'src', 'layer', 'protocol', 'sync.sh', 'tools', 'tests'])
        extra = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard', '-z', '--', 'src', 'tools', 'tests'])
        for filename in extra.decode().split('\0'):
            if filename:
                diff += filename.encode() + b'\0' + Path(filename).read_bytes()
        data['dirty_diff_sha256'] = hashlib.sha256(diff).hexdigest() if diff else 'clean'
        data['binary_sha256'] = hashlib.sha256((Path(args.build_dir) / 'src/gamescope').read_bytes()).hexdigest()
        stage = output(['ssh', args.host, 'mktemp -d /var/tmp/gamescope-test.XXXXXXXX'])
        user = output(['ssh', args.host, 'id -un'])
        metadata = destination.with_name(destination.name + '.metadata.tmp')
        atomic_json(metadata, data)
        try:
            arch = output(['ssh', args.host, 'uname -m'])
            files = [str(Path(args.build_dir) / 'src/gamescope'), str(Path(args.build_dir) / 'src/gamescope-damage'),
                     str(Path(args.build_dir) / f'layer/libVkLayer_FROG_gamescope_wsi_{arch}.so'),
                     __file__, str(Path(__file__).with_name('benchmark_mouse.py'))]
            data['artifact_sha256'] = {Path(filename).name: hashlib.sha256(Path(filename).read_bytes()).hexdigest()
                                       for filename in files}
            atomic_json(metadata, data)
            run(['scp', *files, f'{args.host}:{stage}/'])
            run(['scp', str(metadata), f'{args.host}:{stage}/metadata.json'])
        finally:
            metadata.unlink(missing_ok=True)
        remote_command = shlex.join(['sudo', '-n', 'python3', stage + '/benchmark.py', 'remote', '--stage', stage, '--user', user])
        ssh = subprocess.Popen(['ssh', '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=2', args.host, remote_command],
                               stdin=subprocess.PIPE, text=True)
        def heartbeat():
            while not stopped.is_set():
                try:
                    ssh.stdin.write('alive\n'); ssh.stdin.flush()
                except (BrokenPipeError, OSError, ValueError):
                    return
                stopped.wait(1)
        worker = threading.Thread(target=heartbeat, daemon=True); worker.start()
        status = ssh.wait(); stopped.set(); worker.join(timeout=2)
        run(['scp', f'{args.host}:{stage}/result.json', str(destination) + '.json'])
        data = json.loads(Path(str(destination) + '.json').read_text())
        if status and data['status'] == 'complete':
            data.update(status='failed', error=f'SSH/remote exit status {status}')
        run(['scp', f'{args.host}:{stage}/gamescope.log', str(destination) + '.log'])
        data['remote_artifacts'] = stage
    except BaseException as error:
        data.update(status='failed', error=f'{type(error).__name__}: {error}')
        if ssh and ssh.poll() is None:
            stopped.set(); ssh.stdin.close()
            with contextlib.suppress(subprocess.TimeoutExpired):
                ssh.wait(timeout=15)
            if ssh.poll() is None:
                ssh.terminate()
        if stage:
            data['remote_artifacts'] = stage
    finally:
        stopped.set()
        atomic_json(str(destination) + '.json', data)
        temporary = destination.with_name(destination.name + '.tmp')
        temporary.write_text(report(data)); temporary.replace(destination)
        print(report(data))
        print(f'Results: {destination}')
    return 0 if data['status'] == 'complete' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='mode', required=True)
    p = commands.add_parser('local')
    p.add_argument('--host', required=True); p.add_argument('--result', required=True); p.add_argument('--build-dir', required=True)
    p.add_argument('--diagnose-damage', action='store_true', help='log Wayland damage separately; not a performance comparison')
    p = commands.add_parser('remote')
    p.add_argument('--stage', required=True); p.add_argument('--user', required=True)
    p = commands.add_parser('failure')
    p.add_argument('--result', required=True); p.add_argument('--error', required=True)
    args = parser.parse_args()
    if args.mode == 'failure':
        data = {'status': 'failed', 'results': [], 'error': args.error}
        destination = Path(args.result).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(report(data))
        atomic_json(str(destination) + '.json', data)
        return 0
    return local(args) if args.mode == 'local' else remote(args)


if __name__ == '__main__':
    sys.exit(main())
