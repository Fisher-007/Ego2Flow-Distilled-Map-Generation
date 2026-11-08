"""Sample, repair, and assemble source-conditioned Scenario Dreamer road segments."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from scripts.generation.refine_coarse_conditioned import (
    add_initial_seam, configuration, decode, prepare, refinement_data, save)
from scripts.generation.optimize_seam_latent import optimize, inspect_unprojected, PARAMETERS
from map_generation.boundary_shape import BoundaryShape
from map_generation.information_graph import compile_conditions, check_road_candidate


def assemble_candidate(scene, spec, out, conditions=None):
    report = json.loads((out/'repair_review_input.json').read_text())['geometry']
    lanes = {v['id']: v for v in scene['fusion_map']['lanes']}
    seam = lanes[spec['seam_id']]
    incoming = next(v for v in lanes.values() if seam['id'] in v['successors'])
    outgoing = lanes[seam['successors'][0]]
    ends = np.asarray([incoming['centerline'][-1], outgoing['centerline'][0]])
    tangents = np.asarray([np.diff(incoming['centerline'][-2:],axis=0)[0,:2],np.diff(outgoing['centerline'][:2],axis=0)[0,:2]])
    tangents /= np.linalg.norm(tangents,axis=1,keepdims=True)
    points = np.asarray(report['centerline_xy_enu_m'])
    t = np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))];t /= t[-1]
    aligned = points+(1-t[:,None])*(ends[0,:2]-points[0])+t[:,None]*(ends[1,:2]-points[-1])
    # Fit a low-frequency interior shape with fixed source endpoints.
    knots = np.linspace(0,1,5)
    selected = np.column_stack([np.interp(knots,t,aligned[:,i]) for i in range(2)])
    base = BoundaryShape(ends[:,:2],tangents,knots,np.zeros((3,2)))
    shape = BoundaryShape(ends[:,:2],tangents,knots,selected[1:-1]-base.base(knots[1:-1]))
    parameter = np.linspace(0,1,401);xy = shape(parameter);der = shape(parameter,1)
    normal = np.c_[-der[:,1],der[:,0]];normal /= np.linalg.norm(normal,axis=1,keepdims=True)
    width = np.interp(parameter,[0,1],seam.get('end_widths_m',[incoming['width_m'],outgoing['width_m']]))
    envelope = Polygon(np.r_[xy+normal*width[:,None]/2,(xy-normal*width[:,None]/2)[::-1]])
    buildings = unary_union([Polygon(b['exterior'],b['holes']) for b in scene['buildings']])
    from scripts.generation.generate_coarse_conditioned import curvature
    checks={'simple_xy':bool(LineString(xy).is_simple),'valid_width_envelope':bool(envelope.is_valid),
        'inside_building_m':LineString(xy).intersection(buildings).length,
        'max_curvature_per_m':curvature(xy),'max_endpoint_alignment_m':float(np.linalg.norm(aligned-points,axis=1).max()),
        'endpoints_fixed_m':float(np.linalg.norm(xy[[0,-1]]-ends[:,:2],axis=1).max()),
        'interior_knots':3,'source_topology_retained':True,
        'conditioning':'Arc-length-linear endpoint alignment then 3 interior samples and exact clamped boundaries; deterministic, separately reported.'}
    save(out/'assembly_review_input.json',checks)
    if not checks['simple_xy'] or not checks['valid_width_envelope'] or checks['inside_building_m']>.01:
        raise ValueError('Assembled shape fails the static geometry check; retain reports')
    station=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))];station/=station[-1]
    xyz=np.c_[xy,np.interp(station,[0,1],ends[:,2])]
    candidate=deepcopy(seam)
    candidate.update(source='Scenario Dreamer SDEdit with feedback latent repair',centerline=xyz.tolist(),boundary_shape=shape.as_dict(),
        end_widths_m=seam.get('end_widths_m',[incoming['width_m'],outgoing['width_m']]),
        generation_provenance={'sampling':str(out/'sampling.json'),'repair':str(out/'repair_review_input.json'),
          'assembly':str(out/'assembly_review_input.json'),'width_and_height':'Retained source endpoint evidence; deterministic interpolation, not model predictions.'})
    updated=deepcopy(scene);idx=next(i for i,v in enumerate(updated['fusion_map']['lanes']) if v['id']==seam['id'])
    updated['fusion_map']['lanes'][idx]=candidate
    if conditions is not None:
        preservation=check_road_candidate(scene,updated,conditions)
        save(out/'graph_preservation.json',preservation)
        if not preservation['passed']:
            raise ValueError('Candidate changed graph-protected source lanes or traffic relations')
    save(out/'candidate_scene.json',updated)
    save(out/'candidate.json',{'lane':candidate,'static_geometry':checks,
        'other_lanes_unchanged':all(a==b for i,(a,b) in enumerate(zip(scene['fusion_map']['lanes'],updated['fusion_map']['lanes'])) if i!=idx),
        'all_successors_unchanged':all(a['successors']==b['successors'] for a,b in zip(scene['fusion_map']['lanes'],updated['fusion_map']['lanes']))})
    print(json.dumps(checks,indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['sample', 'repair', 'assemble'])
    p.add_argument('--decision', type=Path, required=True)
    p.add_argument('--source-dir', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, help='Read the configured LDM checkpoint from an alternate path')
    p.add_argument('--autoencoder-checkpoint', type=Path, help='Read the configured autoencoder checkpoint from an alternate path')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--information-graph',type=Path)
    p.add_argument('--graph-task',default='source_faithful_simulation')
    args = p.parse_args()
    decision = json.loads(args.decision.read_text()); spec = decision['road']
    scene_path = Path(decision['source_scene']); scene = json.loads(scene_path.read_text())
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    conditions=None
    if args.information_graph:
        graph=json.loads(args.information_graph.read_text())
        conditions=compile_conditions(graph,scene,'task:'+args.graph_task)
        if spec['seam_id'] not in {r['source_id'] for r in conditions['road']['generate']}:
            raise ValueError('The graph does not select this evidence gap for generation')
        save(out/'graph_conditions.json',conditions)
    if args.action == 'assemble':
        assemble_candidate(scene,spec,out,conditions); return
    sys.path.insert(0, str(args.source_dir.resolve()))
    import torch
    import pytorch_lightning as pl
    from models.scenario_dreamer_ldm import ScenarioDreamerLDM
    from utils.train_helpers import set_latent_stats
    pl.seed_everything(spec['seed'], workers=True)
    torch.set_float32_matmul_precision('medium')
    cfg, ae_cfg = configuration(args.source_dir.resolve()); cfg = set_latent_stats(cfg)
    prepared = prepare(scene, spec['seam_id'], cfg.dataset)
    seeded, _ = add_initial_seam(prepared)
    configured_checkpoint = Path(os.environ['MAP_SD_CHECKPOINT']) if os.environ.get('MAP_SD_CHECKPOINT') else Path(cfg.eval.save_dir)/cfg.eval.run_name/'last.ckpt'
    read_checkpoint = args.checkpoint or configured_checkpoint
    configured_autoencoder = cfg.model.autoencoder_path
    if args.autoencoder_checkpoint:
        cfg.model.autoencoder_path = str(args.autoencoder_checkpoint)
    loading = {'configured_checkpoint':str(configured_checkpoint),'read_checkpoint':str(read_checkpoint),
        'configured_autoencoder':str(configured_autoencoder),'read_autoencoder':str(cfg.model.autoencoder_path),
        'status':'loading','started_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'scope':'Optional checkpoint read-location overrides only; model settings, cfg_ae, source conditions and sampling parameters are unchanged.'}
    save(out/'model_loading.json',loading)
    loading_started = time.monotonic()
    model = ScenarioDreamerLDM.load_from_checkpoint(
        str(read_checkpoint), cfg=cfg, cfg_ae=ae_cfg,
        map_location='cpu').to('cuda').eval().requires_grad_(False)
    torch.cuda.synchronize()
    loading.update(status='loaded',load_seconds=time.monotonic()-loading_started,
        loaded_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    save(out/'model_loading.json',loading)
    print(json.dumps(loading),flush=True)
    post_load_started = time.monotonic()
    with torch.no_grad():
        data = refinement_data(model, seeded, len(prepared['context']))
    mask = data['lane'].mask; index = int(torch.where(~mask)[0].item())
    tangents = np.array([np.diff(prepared['incoming']['centerline'][-2:],axis=0)[0,:2],
                         np.diff(prepared['outgoing']['centerline'][:2],axis=0)[0,:2]])
    tangents /= np.linalg.norm(tangents,axis=1,keepdims=True)
    buildings = unary_union([Polygon(b['exterior'], b['holes']) for b in scene['buildings']])
    if args.action == 'sample':
        with torch.no_grad(), model.ema.average_parameters():
            agents, lanes = data['agent'].latents.clone(), data['lane'].latents.clone()
            start = spec['start_t']; xa = agents[:,None].clone()
            t = torch.full((len(lanes),),start,device=model.device,dtype=torch.long)
            xl = model.diff_model.q_sample(lanes[:,None],t,torch.randn_like(lanes[:,None]))
            xl[mask] = lanes[mask,None]
            for step in reversed(range(start+1)):
                ta = torch.full((len(agents),),step,device=model.device,dtype=torch.long)
                tl = torch.full((len(lanes),),step,device=model.device,dtype=torch.long)
                xa,xl = model.diff_model.p_sample(xa,xl,data,ta,tl)
                xa = torch.clip(xa,-cfg.model.diffusion_clip,cfg.model.diffusion_clip)
                xl = torch.clip(xl,-cfg.model.diffusion_clip,cfg.model.diffusion_clip)
                xa[:] = agents[:,None]; xl[mask] = lanes[mask,None]
            raw = decode(model,data,xa[:,0],xl[:,0],out,'raw')
            torch.save({'agents':xa[:,0].cpu(),'lanes':xl[:,0].cpu(),'mask':mask.cpu(),
                        'ids':data['lane'].ids.cpu()},out/'sample_latents.pt')
            review_input = inspect_unprojected(raw,prepared,buildings,tangents)
            save(out/'sampling.json',{'decision':str(args.decision.resolve()),'method':'Official Scenario Dreamer SDEdit-style local refinement',
                'start_t':start,'reverse_steps':start+1,'from_pure_noise':False,
                'initialization':'Provisional endpoint interpolation in a real AV2/OSM evidence gap; not measured gap truth.',
                'known_latent_max_change':float((xl[mask,0]-lanes[mask]).abs().max()),
                'source_context':prepared['context'],'selected_gap':prepared['seam'],
                'signal_coefficient':float(model.diff_model.sqrt_alphas_cumprod[start]),
                'noise_coefficient':float(model.diff_model.sqrt_one_minus_alphas_cumprod[start])})
            save(out/'raw_review_input.json',review_input)
            torch.cuda.synchronize()
            save(out/'execution_metrics.json',{'action':'sample','loading':loading,'post_load_action_seconds':time.monotonic()-post_load_started,'cuda_peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'seed':spec['seed'],'reverse_steps':start+1})
            print(json.dumps(review_input,indent=2))
        return
    # Read the requested latent-repair parameters.
    revision = spec['revision']
    saved = torch.load(out/'sample_latents.pt',map_location=model.device,weights_only=True)
    if not torch.equal(saved['ids'],data['lane'].ids): raise ValueError('Source lane order changed')
    endpoints = torch.as_tensor((prepared['ends'][:,:2]-prepared['center'])@prepared['rotation'].T,device=model.device,dtype=torch.float32)
    model_tangents = torch.as_tensor(tangents@prepared['rotation'].T,device=model.device,dtype=torch.float32)
    raw, repaired, changes = optimize(model,data,saved['agents'],saved['lanes'],index,endpoints,model_tangents,out/'repair')
    report = inspect_unprojected(repaired,prepared,buildings,tangents)
    save(out/'repair_review_input.json',{'revision':revision,'parameters':PARAMETERS,'changes':changes,'geometry':report})
    torch.cuda.synchronize()
    save(out/'repair_execution_metrics.json',{'action':'repair','loading':loading,'post_load_action_seconds':time.monotonic()-post_load_started,'cuda_peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'seed':spec['seed']})
    print(json.dumps(report,indent=2))



if __name__ == '__main__': main()
