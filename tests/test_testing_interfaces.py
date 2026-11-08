"""Adapter integration tests using synthetic subprocess fixtures."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from map_generation.city_workflow import CityWorkflow, read_json, write_json
from map_testing.adapters import check_adapter, execute
from map_testing.dynamic import run as run_dynamic, validate_workflow_artifact
from map_testing.paired import run as run_paired, paired_inputs
from map_testing.scoring import FRAME, score


def dynamic_result(name, revision, policy, evidence):
    return dict(kind='dynamic_policy', execution_status='completed', map_name=name, revision=revision,
                policy=dict(name=policy, checkpoint='synthetic', settings={}),
                learned_control=dict(request_ids=[0], control_steps=10),
                traffic=dict(requested=3, admitted=2, completed=1, unadmitted=1, in_system=1,
                             terminated_unfinished=0, observation_seconds=10,
                             waiting_vehicle_seconds=2, in_system_vehicle_seconds=9),
                model_failures=['Synthetic controller failure retained'], limitations=['Synthetic fixture'],
                evidence=[evidence])


ADAPTER = '''
import argparse, json, os, sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--request');p.add_argument('--output');a=p.parse_args()
r=json.loads(Path(a.request).read_text());c=json.loads(Path(r['adapter_config']).read_text())
out=Path(a.output);folder=out.parent
def save(path,value): path.write_text(json.dumps(value))
(folder/'evidence.txt').write_text('Synthetic fixture; no real map or model. '+os.environ['MAP_CPUSET'])
assert os.sched_getaffinity(0)=={int(c['cpu'])}
assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
for key,value in c.get('expected_environment',{}).items(): assert os.environ[key]==value, key
base=dict(execution_status='completed',map_name=r['map_name'],revision=r['revision'],evidence=['evidence.txt'])
if c.get('fail'):
    save(out,base); print('synthetic failure'); sys.exit(7)
if r['kind']=='dynamic_policy':
    base.update(c['dynamic_result'])
elif r['kind']=='prepare_pair':
    image=folder/'synthetic.png';image.write_bytes(b'synthetic placeholder')
    views=[dict(camera='front',timestamp_ns=10,image=str(image),intrinsics=[[1,0,0],[0,1,0],[0,0,1]],camera_to_ego=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])]
    cases=[dict(case_id=k,log_id='synthetic',timestamp_ns=10,context={'past_speed':1},views=views) for k in ['a','b']]
    bundle=dict(coordinate_frame='anchor_ego_forward_left_m',future_times_s=[0.5,1],cases=cases)
    save(folder/'real.json',bundle); save(folder/'sim.json',bundle)
    ref=dict(bundle,cases=[dict(case_id=k,trajectory_xy_m=[[0,0],[0,0]]) for k in ['a','b']])
    save(folder/'reference.json',ref)
    base.update(real_inputs='real.json',sim_inputs='sim.json',reference='reference.json',limitations=['synthetic'])
else:
    assert r['kind']=='predict' and 'reference' not in r and 'artifacts' not in r and 'cases' not in r
    bundle=json.loads(Path(r['inputs']).read_text())
    assert all('trajectory_xy_m' not in row for row in bundle['cases'])
    distance=1 if r['side']=='real' else 2
    rows=[dict(case_id='a',valid=True,trajectory_xy_m=[[distance,0],[distance,0]])]
    if r['side']=='real': rows.append(dict(case_id='b',valid=False,trajectory_xy_m=None,error='synthetic parse failure'))
    base.update(side=r['side'],coordinate_frame=bundle['coordinate_frame'],future_times_s=bundle['future_times_s'],
        model=dict(name='synthetic',checkpoint='synthetic',settings={'seed':0}),cases=rows)
save(out,base)
'''


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cpu = min(os.sched_getaffinity(0))
        self.resource = dict(cpus=str(self.cpu), gpu=0, port=25040)
        self.script = self.root / 'user_adapter.py'
        self.script.write_text(ADAPTER)
        self.settings = self.root / 'settings.json'
        write_json(self.settings, {'cpu': self.cpu})
        self.adapter = dict(name='synthetic-policy', command=[sys.executable, str(self.script)],
                            cwd=str(self.root), config=str(self.settings))
        self.run = self.root / 'construction'
        runtime = self.root / 'runtime.json'
        write_json(runtime, {'synthetic': True})
        self.state = dict(status='complete', stage='delivery', revision=2,
                          config=dict(map_name='SyntheticMap', carla_root=str(self.root),
                                      resources={'map': self.resource}, dynamic_policy=self.adapter),
                          artifacts={'runtime_manifest': {'path': str(runtime), 'revision': 2}})
        write_json(self.run / 'state.json', self.state)

    def tearDown(self):
        self.temp.cleanup()

    def test_external_pipeline_scores_failures_and_preserves_construction(self):
        cases = self.root / 'cases.json'; write_json(cases, {'synthetic': True})
        config = dict(output=str(self.root / 'paired'), cases=str(cases), resources=self.resource,
                      prepare=self.adapter, predictor=self.adapter, environment={'PYTHONPATH': str(ROOT)})
        self.state['config']['environment'] = {'UE4_ROOT': '/synthetic/engine', 'DISPLAY': ':synthetic'}
        write_json(self.run / 'state.json', self.state)
        write_json(self.settings, dict(cpu=self.cpu, expected_environment={
            'UE4_ROOT': '/synthetic/engine', 'DISPLAY': ':synthetic',
            'MAP_RUNTIME_DIR': str(self.root / 'paired/runtime')}))
        old = (self.run / 'state.json').read_bytes()
        result = run_paired(config, self.run)
        scores = read_json(result.parent / 'scores.json')
        self.assertEqual(scores['requested_cases'], 2)
        self.assertEqual(scores['paired_valid_cases'], 1)
        self.assertEqual(scores['paired_cases_summary']['real_vs_reference']['ade_m'], 1)
        self.assertEqual(scores['paired_cases_summary']['sim_vs_reference']['fde_m'], 2)
        self.assertEqual(scores['paired_cases_summary']['sim_vs_real']['ade_m'], 1)
        self.assertIn('sim', scores['cases'][1]['failures'])
        self.assertEqual(old, (self.run / 'state.json').read_bytes())
        for step in ['prepare','real','sim']:
            self.assertEqual(read_json(result.parent / step / 'execution.json')['exit_code'], 0)
        with self.assertRaises(FileExistsError):
            run_paired(config, self.run)

    def test_horizon_scores_use_all_samples_up_to_each_endpoint(self):
        reference = dict(coordinate_frame=FRAME, future_times_s=[0.5, 1.0],
                         cases=[dict(case_id='a', trajectory_xy_m=[[0, 0], [0, 0]])])
        model = dict(name='synthetic', checkpoint='synthetic', settings={})
        real = dict(reference, model=model, cases=[dict(case_id='a', valid=True, trajectory_xy_m=[[3, 0], [0, 4]])])
        sim = dict(reference, model=model, cases=[dict(case_id='a', valid=True, trajectory_xy_m=[[0, 0], [0, 0]])])
        metrics = score(real, sim, reference)['paired_cases_summary']['sim_vs_real']
        self.assertEqual(metrics['by_horizon'], [dict(time_s=0.5, ade_m=3, fde_m=3),
                                                 dict(time_s=1.0, ade_m=3.5, fde_m=4)])
        self.assertEqual(metrics['ade_m'], 3.5)

    def test_failed_process_keeps_receipt_and_cannot_be_success(self):
        write_json(self.settings, {'cpu': self.cpu, 'fail': True})
        out = self.root / 'failed'
        with self.assertRaisesRegex(RuntimeError, 'exited 7'):
            execute(self.adapter, dict(map_name='SyntheticMap', revision=2), out, self.resource)
        self.assertEqual(read_json(out / 'execution.json')['exit_code'], 7)
        self.assertEqual(read_json(out / 'execution.json')['status'], 'failed')
        self.assertIn('synthetic failure', (out / 'output.log').read_text())
        with self.assertRaisesRegex(ValueError, 'Configure an installed adapter'):
            check_adapter(None)

    def test_dynamic_control_and_current_revision_gate_with_explicit_reuse(self):
        self.state.update(stage='dynamic_review', status='running')
        write_json(self.run / 'state.json', self.state)
        request = dict(map_name='SyntheticMap', revision=2, map_asset='/Game/MAPSyntheticMap',
                       scope='synthetic only', observation_seconds=10)
        for key in ['runtime_manifest', 'scene', 'routes', 'scenario']:
            request[key] = self.state['artifacts']['runtime_manifest']['path']
        request_path = self.root / 'request.json'; write_json(request_path, request)
        fixture = dynamic_result('SyntheticMap', 2, 'synthetic-policy', 'evidence.txt')
        write_json(self.settings, {'cpu':self.cpu, 'dynamic_result':fixture})
        result = run_dynamic(self.run, request_path, self.root / 'dynamic')
        validate_workflow_artifact(result, self.state['config'], 2)
        with self.assertRaisesRegex(ValueError, 'current map and revision'):
            validate_workflow_artifact(result, self.state['config'], 3)
        reuse = self.root / 'reuse.json'
        write_json(reuse, dict(kind='dynamic_policy_reuse', map_name='SyntheticMap', revision=3,
                   previous_result=str(result), unchanged_scope='Synthetic unaffected scope',
                   evidence=[str(result.parent / 'evidence.txt')]))
        validate_workflow_artifact(reuse, self.state['config'], 3)
        fixture['learned_control']['control_steps'] = 0
        write_json(self.settings, {'cpu':self.cpu, 'dynamic_result':fixture})
        with self.assertRaisesRegex(ValueError, 'actually control'):
            run_dynamic(self.run, request_path, self.root / 'no_policy_control')
        self.assertFalse((self.root / 'no_policy_control/result.json').exists())

    def test_workflow_does_not_accept_basic_driving_alone(self):
        config = dict(self.state['config'], workspace=str(ROOT), task='Synthetic', run_id='fixture')
        config.pop('dynamic_policy')
        workflow = CityWorkflow.create(self.root / 'workflow', config)
        workflow.state['stage'] = 'dynamic_review'
        evidence = self.root / 'evidence.txt'; evidence.write_text('synthetic')
        report = self.root / 'report.json'
        write_json(report, dict(status='pass',revision=1,evidence=[str(evidence)]))
        artifacts = [dict(name=k,path=str(report)) for k in workflow.required_artifacts('dynamic_review')]
        with self.assertRaisesRegex(ValueError, 'Configure dynamic_policy'):
            workflow.advance({'artifacts': artifacts})

    def test_pairing_checks_context_calibration_and_causality(self):
        out = self.root / 'prepare'
        execute(self.adapter, dict(kind='prepare_pair',map_name='SyntheticMap',revision=2), out, self.resource)
        real, sim, ref = [out / name for name in ['real.json','sim.json','reference.json']]
        value = read_json(sim); value['cases'][0]['context']['past_speed'] = 2; write_json(sim,value)
        with self.assertRaisesRegex(ValueError, 'non-image context'):
            paired_inputs(real,sim,ref)
        value = read_json(real); value['cases'][0]['views'][0]['intrinsics'][0][0]=2; write_json(sim,value)
        with self.assertRaisesRegex(ValueError, 'calibration'):
            paired_inputs(real,sim,ref)
        value = read_json(real); value['cases'][0]['views'][0]['timestamp_ns']=11
        write_json(real,value); write_json(sim,value)
        with self.assertRaisesRegex(ValueError, 'causal anchor'):
            paired_inputs(real,sim,ref)

    def test_scoring_retains_zero_coverage_and_rejects_changed_models(self):
        reference = dict(coordinate_frame=FRAME,future_times_s=[1],cases=[dict(case_id='a',trajectory_xy_m=[[1,0]])])
        prediction = dict(coordinate_frame=FRAME,future_times_s=[1],model=dict(name='fixture',checkpoint='fixture',settings={}),cases=[])
        result = score(prediction,prediction,reference)
        self.assertEqual(result['paired_valid_cases'],0)
        self.assertIsNone(result['all_valid_summary']['real_vs_reference']['ade_m'])
        different = deepcopy(prediction); different['model']['settings']['seed']=1
        with self.assertRaisesRegex(ValueError, 'same recorded model'):
            score(prediction,different,reference)

    def test_templates_do_not_fake_a_model_run(self):
        for filename, kind in [('learned_policy.py','dynamic_policy'),('paired_test.py','predict')]:
            request = self.root / 'template_request.json'
            write_json(request,dict(kind=kind,adapter_config=str(self.settings)))
            output = self.root / (kind + '.json')
            result = subprocess.run([sys.executable,str(ROOT/'examples/adapters'/filename),
                                     '--request',str(request),'--output',str(output)],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('NotImplementedError',result.stderr)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
