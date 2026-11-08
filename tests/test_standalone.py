"""Offline workflow integration tests using synthetic inputs."""
import ast
import base64
from contextlib import contextmanager, redirect_stdout
import io
import json
import os
import re
import warnings
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from map_generation.agent_provider import AgentRequest, OpenAICompatibleProvider, ProviderError
from map_generation.city_workflow import CityWorkflow, STAGES, REQUIRED_ARTIFACTS, read_json, write_json
from scripts.workflow.run import main as runner_main
from scripts.data.index_local_sources import index
from build_map import load_config


def action(kind='tool', **kwargs):
    result = dict(kind=kind, reason='Synthetic integration fixture', tool='', arguments_json='{}',
                  artifacts=[], target_stage='', findings=[], summary='')
    result.update(kwargs)
    return result


class Reply(io.BytesIO):
    def __init__(self, value):
        super().__init__(json.dumps(value).encode())
        self.status = 200
        self.headers = {}


def completion(value):
    return Reply({'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(value)}}]})


@contextmanager
def cwd(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class PackageTests(unittest.TestCase):
    def test_source_and_tool_dependency_closure(self):
        for path in ROOT.rglob('*.py'):
            text = path.read_text()
            tree = ast.parse(text, filename=str(path))
            for module in re.findall(r'\bscripts\.[a-z_][a-z_0-9]*\.[a-z_][a-z_0-9]*', text):
                self.assertTrue((ROOT / (module.replace('.', '/') + '.py')).is_file(), (path, module))
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and not node.level:
                    modules = [node.module or '']
                for module in modules:
                    if module.startswith(('map_generation.', 'map_testing.', 'scripts.')):
                        local = ROOT / module.replace('.', '/')
                        self.assertTrue(local.with_suffix('.py').is_file() or local.is_dir(), (path, module))
        registry = read_json(ROOT / 'configs/workflows/city_tools.json')
        for row in registry['tools']:
            entry = row['entrypoint']
            target = entry['target'].replace('.', '/') + '.py' if entry['kind'] == 'module' else entry['target']
            self.assertTrue((ROOT / target).is_file(), target)
        for path in ROOT.rglob('*.sh'):
            subprocess.run(['bash', '-n', str(path)], check=True, capture_output=True)

    def test_folder_runs_without_its_parent(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / 'map_builder'
            shutil.copytree(ROOT, destination, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'runs', '*.local.json', 'docs'))
            env = {**os.environ, 'PYTHONPATH': '', 'PYTHONDONTWRITEBYTECODE': '1'}
            result = subprocess.run([sys.executable, 'build_map.py', '--help'], cwd=destination,
                                    env=env, text=True, capture_output=True, check=True)
            self.assertIn('prepare-native', result.stdout)
            subprocess.run([sys.executable, 'test_map.py', '--help'], cwd=destination,
                           env=env, text=True, capture_output=True, check=True)
            subprocess.run([sys.executable, 'run_tool.py', '--help'], cwd=destination,
                           env=env, text=True, capture_output=True, check=True)
            listing = subprocess.run([sys.executable, 'run_tool.py', 'list'], cwd=destination,
                                     env=env, text=True, capture_output=True, check=True)
            self.assertEqual(len(json.loads(listing.stdout)), 144)
            inventory = subprocess.run([sys.executable, '-m', 'scripts.workflow.inventory', '--stage', 'structure', '--signatures'],
                                       cwd=destination, env=env, text=True, capture_output=True, check=True)
            tools = json.loads(inventory.stdout)['tools']
            self.assertTrue(tools)
            self.assertTrue(all(t['source_exists'] for t in tools))
            self.assertFalse(any(p.is_symlink() for p in destination.rglob('*')))

    def test_configuration_resolves_paths_and_roles(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'config.json'
            config = read_json(ROOT / 'configs/build.example.json')
            config['input']['bbox_city_xy'] = [-500, -500, 500, 500]
            config['input']['data_root'] = 'data'
            config['agent']['roles']['static_review'] = {'model': 'fixture-review-model'}
            write_json(path, config)
            actual = load_config(path)
            self.assertEqual(actual['data_root'], str(Path(temporary) / 'data'))
            self.assertEqual(actual['python'], sys.executable)
            self.assertEqual(actual['agent']['roles']['static_review']['model'], 'fixture-review-model')
            self.assertNotIn('api_key', actual['agent'])
            self.assertNotIn('components', actual)

    def test_local_index_keeps_source_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lane = dict(is_intersection=True, left_lane_boundary=[dict(x=0, y=0, z=2), dict(x=10, y=0, z=2)],
                        right_lane_boundary=[dict(x=0, y=3, z=2), dict(x=10, y=3, z=2)])
            for split in ('train', 'val'):
                log = root / 'sensor' / split / ('synthetic_' + split)
                write_json(log / 'map/log_map_archive_fixture____ATX_0.json', {'lane_segments': {'1': lane}})
                (log / 'city_SE3_egovehicle.feather').write_bytes(b'synthetic existence fixture')
            result = index(root, 'ATX', ['train', 'val'])
            self.assertEqual({row['split'] for row in result['candidates']}, {'train', 'val'})
            self.assertEqual(result['candidates'][0]['bbox_city_xy'], [0, 0, 10, 3])
            self.assertEqual(len(index(root, 'ATX', ['train', 'val'], ['synthetic_val'])['candidates']), 1)

    def test_single_area_selection(self):
        from scripts.data.prepare_city_selection import prepare
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = root / 'av2/sensor/train/synthetic_source'
            lane = dict(id=1, lane_type='VEHICLE', left_lane_boundary=[dict(x=0, y=0, z=2)],
                        right_lane_boundary=[dict(x=0, y=3, z=2)])
            vector = log / 'map/log_map_archive_fixture____ATX_0.json'
            write_json(vector, {'lane_segments': {'1': lane}})
            (log / 'city_SE3_egovehicle.feather').write_bytes(b'synthetic existence fixture')
            observation = dict(log_id='synthetic_source', split='train', nearest_timestamp_ns=1000, ego_distance_m=0)
            write_json(root / 'catalog.json', {'city': 'ATX', 'candidates': [dict(observation, vector_map=str(vector.relative_to(root / 'av2')))]})
            write_json(root / 'candidates.json', {'city': 'ATX', 'candidates': [dict(candidate_id='junction', sources=[observation],
                       junction_bbox_city_xy=[-10,-10,10,10], center_city_xy=[0,0])]})
            write_json(root / 'selection.json', dict(city='ATX', selected_candidate_ids=['junction'], bbox_city_xy=[-500,-500,500,500], selection_rationale='Synthetic input'))
            with redirect_stdout(io.StringIO()):
                prepare(root / 'selection.json', root / 'catalog.json', root / 'candidates.json', root, root / 'selected')
            record = read_json(root / 'selected/area_selection.json')
            self.assertIsNone(record['minimum_area_center_separation_m'])
            self.assertEqual(len(record['areas']), 1)
            self.assertEqual(read_json(root / 'selected/city_config.json')['coordinates']['origin_city_xyz'][2], 2)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.workspace = self.directory / 'workspace'
        for name in ('map_generation', 'map_testing', 'scripts', 'configs/workflows'):
            shutil.copytree(ROOT / name, self.workspace / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        self.run = self.directory / 'run'
        self.config = dict(run_id='synthetic', map_name='synthetic', task='Offline state-machine test', workspace=str(self.workspace),
                           dynamic_policy={'name': 'synthetic-policy'},
                           agent=dict(provider='http', base_url='https://fixture.invalid/v1', model='fixture', api_key_env='FIXTURE_KEY'),
                           resources={'map': dict(cpus=str(min(os.sched_getaffinity(0))), gpu=0, port=25000)},
                           environment={'PYTHONPATH': str(self.workspace), 'PYTHONDONTWRITEBYTECODE': '1'})
        self.workflow = CityWorkflow.create(self.run, self.config)

    def tearDown(self):
        self.temporary.cleanup()

    def invoke(self, transport, max_actions):
        with warnings.catch_warnings(), patch('map_generation.agent_provider.urlopen', side_effect=transport), \
             patch.dict(os.environ, {'FIXTURE_KEY': 'synthetic-key', 'PYTHONDONTWRITEBYTECODE': '1'}), \
             patch.object(sys, 'argv', ['run', 'resume', '--run', str(self.run), '--max-actions', str(max_actions)]), \
             cwd(self.workspace), redirect_stdout(io.StringIO()):
            # Poll workers that persist across runner invocations.
            warnings.filterwarnings('ignore', message=r'subprocess \d+ is still running', category=ResourceWarning)
            runner_main()

    def test_role_instructions_are_loaded_without_docs(self):
        self.assertFalse((self.workspace / 'docs').exists())
        method = (self.workspace / 'configs/workflows/instructions/method.md').read_text()
        for stage, expected, omitted in (
            ('source_selection', ('construction.md',), ('review.md', 'dynamic_policy.md')),
            ('static_review', ('review.md',), ('construction.md', 'dynamic_policy.md')),
            ('dynamic_review', ('review.md', 'dynamic_policy.md'), ('construction.md',)),
        ):
            with self.subTest(stage=stage):
                workflow = CityWorkflow.create(self.directory / stage, self.config)
                workflow.state['stage'] = stage
                request = workflow.request()
                system = request.messages[0]['content']
                packet = json.loads(request.messages[1]['content'][0]['text'])
                self.assertIn(method, system)
                self.assertNotIn('docs/', json.dumps(request.messages))
                for name in expected:
                    path = 'configs/workflows/instructions/' + name
                    self.assertIn(path, packet['instruction_files'])
                    self.assertIn((self.workspace / path).read_text(), system)
                for name in omitted:
                    self.assertNotIn('configs/workflows/instructions/' + name, packet['instruction_files'])
                for path in packet['instruction_files']:
                    self.assertIn((self.workspace / path).read_text(), system)

    def test_pending_request_preserves_loaded_instructions(self):
        request = self.workflow.request()
        path = self.workspace / 'configs/workflows/instructions/method.md'
        path.write_text(path.read_text() + '\nSynthetic instruction update.\n')
        resumed = CityWorkflow(self.run)
        self.assertEqual(resumed.request().messages, request.messages)
        resumed.apply(action(tool='list', arguments_json='{}'), request.request_id)
        following = CityWorkflow(self.run).request()
        self.assertIn('Synthetic instruction update.', following.messages[0]['content'])

    def test_api_image_worker_failure_and_saved_request_resume(self):
        picture = self.directory / 'pixel.png'
        picture.write_bytes(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a2ioAAAAASUVORK5CYII='))
        def first(request, **kwargs):
            self.assertEqual(request.full_url, 'https://fixture.invalid/v1/chat/completions')
            self.assertEqual(request.get_header('Authorization'), 'Bearer synthetic-key')
            payload = json.loads(request.data)
            self.assertTrue(payload['response_format']['json_schema']['strict'])
            return completion(action(tool='image', arguments_json=json.dumps({'path': str(picture)})))
        self.invoke(first, 1)
        with self.assertRaises(ProviderError):
            self.invoke(lambda *a, **k: (_ for _ in ()).throw(URLError('synthetic interruption')), 1)
        pending = read_json(self.run / 'state.json')['pending_request']
        saved_request = (self.run / 'requests' / (pending + '.json')).read_bytes()
        def launch(request, **kwargs):
            payload = json.loads(request.data)
            images = [v for v in payload['messages'][1]['content'] if v['type'] == 'image_url']
            self.assertTrue(images[0]['image_url']['url'].startswith('data:image/png;base64,'))
            return completion(action(tool='run', arguments_json=json.dumps(dict(resource='map', cpu_only=True,
                argv=[sys.executable, '-c', 'import os,sys; print("worker executed", os.environ["CUDA_VISIBLE_DEVICES"]); sys.exit(7)'],
                purpose='Synthetic failing child process'))))
        self.invoke(launch, 1)
        self.assertEqual((self.run / 'requests' / (pending + '.json')).read_bytes(), saved_request)
        job = read_json(self.run / 'state.json')['jobs'][0]
        self.invoke(lambda *a, **k: completion(action(tool='poll', arguments_json=json.dumps({'job_id': job, 'wait_seconds': 20}))), 1)
        self.assertEqual(read_json(self.run / 'jobs' / job / 'result.json')['exit_code'], 7)
        spec = read_json(self.run / 'jobs' / job / 'spec.json')
        self.assertEqual(spec['argv'][:2], ['taskset', '-c'])
        self.assertEqual(spec['env']['CUDA_VISIBLE_DEVICES'], '')
        self.assertIn('worker executed', (self.run / 'jobs' / job / 'output.log').read_text())
        self.assertEqual(len(read_json(self.run / 'state.json')['jobs']), 1)
        self.assertTrue((self.run / 'provider_errors' / (pending + '.json')).is_file())

    def advance_fixture(self, workflow):
        request = workflow.request()
        stage = workflow.state['stage']
        evidence = self.directory / 'synthetic_evidence.txt'
        evidence.write_text('Synthetic workflow fixture; no map-quality claim.')
        artifacts = []
        for name in REQUIRED_ARTIFACTS[stage]:
            path = self.directory / f'{stage}_{name}.json'
            value = dict(status='pass', revision=workflow.state['revision'], evidence=[str(evidence)]) if name in {'static_review','dynamic_review'} else {'synthetic': True}
            if name == 'dynamic_policy_results':
                from test_testing_interfaces import dynamic_result
                receipt = self.directory / 'synthetic_execution.json'
                write_json(receipt, dict(status='completed', exit_code=0))
                value = dynamic_result('synthetic', workflow.state['revision'], 'synthetic-policy', str(evidence))
                value['execution'] = str(receipt)
            write_json(path, value)
            artifacts.append(dict(name=name, path=str(path)))
        workflow.apply(action('advance', artifacts=artifacts), request.request_id)

    def test_roles_feedback_revision_and_complete(self):
        workflow = self.workflow
        workflow.event('private_reasoning_fixture', {'text': 'constructor-only synthetic content'})
        for stage in STAGES[:7]:
            self.assertEqual(workflow.state['stage'], stage)
            self.advance_fixture(workflow)
        request = workflow.request()
        self.assertEqual(workflow.role(), 'static_review')
        packet = json.loads(request.messages[1]['content'][0]['text'])
        self.assertNotIn('constructor-only synthetic content', json.dumps(packet['recent_events']))
        observation = self.directory / 'observed.txt'
        observation.write_text('Synthetic surface discontinuity fixture')
        workflow.apply(action('feedback', target_stage='structure', findings=[dict(id='gap', description='Synthetic gap',
                       required_change='Synthetic repair', evidence=[str(observation)])]), request.request_id)
        self.assertEqual(workflow.state['revision'], 2)
        self.assertEqual(workflow.state['stage'], 'structure')
        self.assertEqual(set(workflow.state['completed_stages']), set(STAGES[:4]))
        self.assertTrue((self.run / 'revisions/revision-0001.json').is_file())
        while workflow.state['status'] != 'complete':
            self.advance_fixture(workflow)
        self.assertEqual(set(workflow.state['review_requests']), {'static_review','dynamic_review'})
        workflow.reopen({'text': 'Synthetic new observation'})
        self.assertEqual(workflow.state['status'], 'running')
        self.assertEqual(workflow.role(), 'decision')

    def test_stale_review_is_not_accepted(self):
        workflow = self.workflow
        workflow.state['stage'] = 'static_review'
        workflow.state['revision'] = 3
        evidence = self.directory / 'evidence.txt'
        evidence.write_text('synthetic')
        report = self.directory / 'review.json'
        write_json(report, dict(status='pass', revision=2, evidence=[str(evidence)]))
        workflow.request()
        with self.assertRaisesRegex(ValueError, 'current construction revision'):
            workflow.advance(action('advance', artifacts=[dict(name='static_review', path=str(report)), dict(name='native_views', path=str(evidence))]))

    def test_tool_call_api_mode(self):
        req = self.workflow.request()
        req = AgentRequest(req.request_id, req.model, req.messages, req.response_schema, response_mode='tool_call')
        expected = action('blocked', summary='Synthetic question')
        def transport(request, **kwargs):
            payload = json.loads(request.data)
            self.assertEqual(payload['tool_choice']['function']['name'], 'submit_action')
            return Reply({'choices': [{'finish_reason': 'tool_calls', 'message': {'tool_calls': [{'function': {'name': 'submit_action', 'arguments': json.dumps(expected)}}]}}]})
        with patch('map_generation.agent_provider.urlopen', side_effect=transport):
            response = OpenAICompatibleProvider(base_url='https://fixture.invalid/v1', api_key=None).complete(req)
        self.assertEqual(response.action, expected)


if __name__ == '__main__':
    unittest.main()
