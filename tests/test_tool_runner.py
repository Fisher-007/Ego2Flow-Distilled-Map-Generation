"""Exercise configured execution and native launch arguments without model weights."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_map import load_config, initialize
from map_generation.city_workflow import read_json, write_json
from run_tool import launch_spec, registry


class ToolRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / 'build.json'
        self.raw = read_json(ROOT / 'configs/build.example.json')
        self.raw['input'].update(bbox_city_xy=[0, 0, 100, 100], data_root=str(self.root / 'data'))
        self.raw['resources']['cpus'] = str(min(os.sched_getaffinity(0)))
        self.raw['output'] = str(self.root / 'run')
        write_json(self.path, self.raw)
        self.config = load_config(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def test_registry_dispatch_and_installed_model_paths(self):
        for tool in registry(ROOT):
            spec = launch_spec(self.config, tool['id'], ['--help'])
            self.assertEqual(spec['argv'][:3], ['taskset', '-c', self.raw['resources']['cpus']])
            self.assertEqual(spec['environment']['SUMO_HOME'], self.config['sumo_home'])
        sample = launch_spec(self.config, 'generate_corridor', ['sample', '--decision', 'd.json'])
        self.assertEqual(sample['argv'][3], self.config['scenario_dreamer_python'])
        self.assertEqual(sample['argv'][-2:], ['--source-dir', self.config['scenario_dreamer_source']])
        visual = launch_spec(self.config, 'cosmos_transfer', ['--model=/override'])
        self.assertNotIn('--model', visual['argv'])
        self.assertEqual(visual['argv'][-2:], ['--guardrail-root', self.config['cosmos_guardrail_root']])
        blend = launch_spec(self.config, 'export_architecture', ['--plan', 'plan.json'])
        self.assertEqual(blend['argv'][3], self.config['blender'])
        self.assertIn('--python-exit-code', blend['argv'])

    def test_actual_tool_cli_works_from_another_directory(self):
        log = self.root / 'data/sensor/train/synthetic'
        lane = dict(is_intersection=True, left_lane_boundary=[dict(x=0, y=0, z=0)],
                    right_lane_boundary=[dict(x=1, y=1, z=0)])
        write_json(log / 'map/log_map_archive_fixture____ATX_0.json', {'lane_segments': {'1': lane}})
        (log / 'city_SE3_egovehicle.feather').write_bytes(b'synthetic existence fixture')
        record = self.root / 'command'
        catalog = self.root / 'catalog.json'
        command = [sys.executable, str(ROOT / 'run_tool.py'), 'run', '--config', str(self.path),
                   '--tool', 'index_local_sources', '--record-dir', str(record), '--',
                   '--data-root', self.config['data_root'], '--city', 'ATX', '--output', str(catalog)]
        result = subprocess.run(command, cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(read_json(catalog)['indexed_logs'], 1)
        self.assertEqual(read_json(record / 'result.json')['exit_code'], 0)
        self.assertEqual(read_json(record / 'command.json')['cwd'], str(ROOT))

    def test_initialize_copies_workspace_and_uses_private_native_project(self):
        carla = self.root / 'carla'
        project = carla / 'Unreal/CarlaUE4'
        write_json(project / 'CarlaUE4.uproject', {'synthetic': True})
        (project / 'Config').mkdir()
        (project / 'Config/DefaultEngine.ini').write_text('; synthetic')
        for name in ('Binaries', 'Plugins', 'Source', 'Content/Carla'):
            (project / name).mkdir(parents=True)
        (carla / 'PythonAPI').mkdir()
        (carla / 'Util/BuildTools').mkdir(parents=True)
        (carla / 'Util/BuildTools/Import.py').write_text('# synthetic')
        self.raw['software']['carla_root'] = str(carla)
        write_json(self.path, self.raw)
        source = self.root / 'source'
        for name in ('map_generation', 'map_testing', 'scripts', 'native', 'examples', 'configs/workflows'):
            shutil.copytree(ROOT / name, source / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        with patch('build_map.ROOT', source):
            workflow = initialize(load_config(self.path))
        private = self.root / 'run/native/carla'
        self.assertEqual(workflow.config['carla_root'], str(private))
        self.assertEqual(workflow.config['environment']['CARLA_ROOT'], str(private))
        self.assertTrue((private / 'Unreal/CarlaUE4/Content/Carla').is_symlink())
        self.assertFalse((private / 'Unreal/CarlaUE4/Config').is_symlink())
        spec = launch_spec(workflow.config, 'generate_corridor', ['--help'])
        self.assertEqual(spec['cwd'], str(self.root / 'run/workspace'))
        self.assertEqual(spec['environment']['PYTHONPATH'], spec['cwd'])
        self.assertTrue((Path(spec['cwd']) / 'map_testing/dynamic.py').is_file())
        self.assertFalse((workflow.workspace / 'docs').exists())
        request = workflow.request()
        self.assertNotIn('docs/', json.dumps(request.messages))
        self.assertIn('# Construction recipe', request.messages[0]['content'])

    def test_carla_launch_only_selects_prepared_cache_configuration(self):
        carla, engine, cache = [self.root / name for name in ('carla', 'engine', 'cache')]
        write_json(carla / 'Unreal/CarlaUE4/CarlaUE4.uproject', {'synthetic': True})
        executable = engine / 'Engine/Binaries/Linux/UE4Editor'
        executable.parent.mkdir(parents=True)
        executable.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
        executable.chmod(0o755)
        env = {**os.environ, **self.config['environment'], 'CARLA_ROOT': str(carla), 'UE4_ROOT': str(engine),
               'MAP_CPUSET': self.raw['resources']['cpus'], 'MAP_GPU': '0', 'MAP_DDC_DIR': str(cache),
               'MAP_RUNTIME_DIR': str(self.root / 'runtime')}
        command = ['bash', str(ROOT / 'scripts/env/run_carla.sh'), '/Game/SyntheticMap']
        before = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
        self.assertNotIn('-DefEngineIni=', before.stdout)
        cache.mkdir()
        (cache / 'DefaultEngine.ini').write_text('; synthetic')
        after = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
        self.assertIn('-DefEngineIni=' + str(cache / 'DefaultEngine.ini'), after.stdout)


if __name__ == '__main__':
    unittest.main()
