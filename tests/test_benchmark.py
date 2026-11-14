import copy
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import benchmark
from benchmark_mouse import Mouse


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.before = {'gamescope': [{'pid': 42, 'start': 99, 'ticks': 100, 'gpu': {
            'render|v3d|1': {'drm-engine-csd': 10, 'v3d-jobs-csd': 1}}}]}
        self.after = copy.deepcopy(self.before)
        p = self.after['gamescope'][0]
        p['ticks'] += 2 * os.sysconf('SC_CLK_TCK')
        p['gpu']['render|v3d|1'] = {'drm-engine-csd': 2000000010, 'v3d-jobs-csd': 5}
        self.events = [{'time': t, 'seq': seq} for t, seq in [(0, 0xffffffff), (1, 0), (2, 2)]]

    def test_counter_units_and_sequence_wrap(self):
        result = benchmark.summarize(self.before, self.after, 2, self.events)
        process = result['processes']['gamescope']
        self.assertEqual(process['cpu_percent'], 100)
        self.assertEqual(process['engines']['csd']['ms_per_second'], 1000)
        self.assertEqual(process['engines']['csd']['jobs_per_second'], 2)
        self.assertEqual(process['engines']['csd']['ms_per_job'], 500)
        self.assertEqual(result['missed_refreshes'], 1)

    def test_reject_restarted_process(self):
        self.after['gamescope'][0]['start'] += 1
        with self.assertRaisesRegex(RuntimeError, 'restarted'):
            benchmark.summarize(self.before, self.after, 2, self.events)

    def test_reject_changed_gpu_clients(self):
        self.after['gamescope'][0]['gpu'] = {}
        with self.assertRaisesRegex(RuntimeError, 'clients changed'):
            benchmark.summarize(self.before, self.after, 2, self.events)

    def test_reject_missing_presentations(self):
        with self.assertRaisesRegex(RuntimeError, 'too few'):
            benchmark.summarize(self.before, self.after, 2, [])

    def test_failed_report_with_spaces(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result with spaces.txt'
            subprocess.run([sys.executable, str(ROOT / 'tools/benchmark.py'), 'failure',
                            '--result', str(path), '--error', 'simulated startup failure'], check=True)
            self.assertIn('FAILED', path.read_text())
            self.assertEqual(json.loads(Path(str(path) + '.json').read_text())['status'], 'failed')

    def test_report_marks_native_output_and_missing_gpu_timing(self):
        summary = benchmark.summarize(self.before, self.after, 2, self.events)
        process = dict(summary['processes']['gamescope'], engines={})
        summary['processes'] = {name: process for name in ('gamescope', 'xwayland', 'mangoapp', 'app')}
        text = benchmark.report({'status': 'complete', 'active_modes': [['1920x1200', 60]],
                                 'gpu_timing_available': False, 'results': [dict(
                                     damage=50, overlay=False, mouse=False, night=False, summary=summary)]})
        self.assertIn('1920x1200', text)
        self.assertIn('CPU and presentation measurements only', text)
        self.assertIn('unavailable', text)
        self.assertNotIn('idle', text)

    def test_sync_rejects_arguments_before_connecting(self):
        for args in [['--test'], ['--test', ''], ['--test', 'result.txt', '--appid', '1', 'pi']]:
            result = subprocess.run(['sh', str(ROOT / 'sync.sh'), *args], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b'--test', result.stderr)

    def test_invalid_mouse_parameters_do_not_open_device(self):
        for kwargs in [{'rate': 0}, {'radius': -1}, {'period': float('nan')}]:
            with self.assertRaises(ValueError):
                Mouse(**kwargs)

    def test_regular_sync_and_appid_keep_deployment_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / 'bin'
            binary.mkdir()
            stub = binary / 'stub'
            stub.write_text('#!' + sys.executable + '''
import json, os, pathlib, sys
name = pathlib.Path(sys.argv[0]).name
with open(os.environ['CALL_LOG'], 'a') as log:
    log.write(json.dumps([name, *sys.argv[1:]]) + '\\n')
if name == 'ssh':
    if sys.argv[-2:] == ['uname', '-m']:
        print(os.environ['TARGET_ARCH'])
    else:
        pathlib.Path(os.environ['REMOTE_BODY']).write_text(sys.stdin.read())
''')
            stub.chmod(0o755)
            for name in ('ssh', 'meson', 'ninja', 'scp'):
                (binary / name).symlink_to(stub)
            env = {**os.environ, 'PATH': str(binary) + ':' + os.environ['PATH'],
                   'CALL_LOG': str(root / 'calls'), 'REMOTE_BODY': str(root / 'remote'),
                   'TARGET_ARCH': platform.machine()}
            for args, appid in [([], '0'), (['--appid', '123'], '123')]:
                (root / 'calls').write_text('')
                subprocess.run(['sh', str(ROOT / 'sync.sh'), *args, 'pi'], cwd=root, env=env, check=True)
                calls = [json.loads(line) for line in (root / 'calls').read_text().splitlines()]
                self.assertEqual(sum(call[0] == 'scp' for call in calls), 2)
                self.assertEqual(calls[-1][-2:], [appid, platform.machine()])
                self.assertIn('sudo install -Z -m755 ~/gamescope /usr/bin/gamescope', (root / 'remote').read_text())


if __name__ == '__main__':
    unittest.main()
