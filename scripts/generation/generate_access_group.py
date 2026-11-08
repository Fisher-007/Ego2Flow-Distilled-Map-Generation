"""Condition official Scenario Dreamer on a reviewed group of inferred paths."""
import argparse
import json
import os
from pathlib import Path
import sys
import shutil
import numpy as np
from shapely.geometry import LineString,box
from shapely.ops import unary_union
from collections import deque
from scripts.generation.refine_coarse_conditioned import configuration,refinement_data,decode,save


def model_input(condition,cfg):
    from utils.lane_graph_helpers import resample_polyline
    lanes={l['id']:l for l in condition['lanes']};unknown=set(condition['generated_path_ids'])
    required=unknown|{r['neighbor_id'] for r in condition['boundaries'] if r['fixed']}
    bbox=condition['region']['bbox_local_xy'];center=(np.asarray(bbox[:2])+bbox[2:])/2
    first=lanes[condition['generated_path_ids'][0]];delta=np.asarray(first['centerline'][-1])[:2]-np.asarray(first['centerline'][0])[:2]
    angle=np.pi/2-np.arctan2(delta[1],delta[0]);rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    mutable_points=np.concatenate([np.asarray(lanes[k]['centerline'])[:,:2]for k in sorted(unknown)])
    local=(mutable_points-center)@rotation.T
    frame_method='Original route-aligned source frame'
    if np.abs(local).max()>=cfg.fov/2-.01:
        # Rotate the conditioning frame to fit the model field of view.
        trials=[]
        for candidate in angle+np.deg2rad(np.arange(180)):
            matrix=np.array([[np.cos(candidate),-np.sin(candidate)],[np.sin(candidate),np.cos(candidate)]])
            points=mutable_points@matrix.T;lo=points.min(0);hi=points.max(0);span=hi-lo
            trials.append((float(span.max()),float(np.prod(span)),candidate,(lo+hi)/2,matrix))
        _,_,_,mid,rotation=min(trials,key=lambda row:row[:2])
        center=mid@rotation
        frame_method='Rigid full-mutable-extent fit within the unchanged official model field; no coordinate scaling'
    field=box(-cfg.fov/2+.01,-cfg.fov/2+.01,cfg.fov/2-.01,cfg.fov/2-.01)
    context=[];points=[]
    for key,lane in sorted(lanes.items()):
        xy=(np.asarray(lane['centerline'])[:,:2]-center)@rotation.T;line=LineString(xy)
        if key in unknown:
            if not field.covers(line):raise ValueError('Generated path exceeds model field: '+key)
            piece=line
        else:
            clipped=line.intersection(field)
            minimum_length=1e-8 if key in required else .1
            pieces=[p for p in getattr(clipped,'geoms',[clipped]) if p.geom_type=='LineString' and p.length>minimum_length]
            if not pieces:continue
            piece=max(pieces,key=lambda p:p.length)
        points.append(resample_polyline(np.asarray(piece.coords),num_points=20))
        context.append(dict(id=key,source=lane['source'],width_m=lane['width_m'],generated=key in unknown))
    budget=int(cfg.max_num_lanes)
    if len(required)>budget:raise ValueError('Required paths and boundary context exceed the official model lane budget')
    available={r['id'] for r in context};links={key:set() for key in available}
    for key in available:
        for target in [*lanes[key]['successors'],lanes[key].get('left_neighbor'),lanes[key].get('right_neighbor')]:
            if target in links:links[key].add(target);links[target].add(key)
    hops={key:0 for key in required};queue=deque(sorted(required))
    while queue:
        key=queue.popleft()
        for target in sorted(links.get(key,())):
            if target not in hops:hops[target]=hops[key]+1;queue.append(target)
    repair=unary_union([LineString(np.asarray(lanes[key]['centerline'])[:,:2]) for key in unknown])
    order=sorted(range(len(context)),key=lambda i:(context[i]['id'] not in required,
        hops.get(context[i]['id'],1000000),
        LineString(np.asarray(lanes[context[i]['id']]['centerline'])[:,:2]).distance(repair),context[i]['id']))
    chosen=set(order[:budget]);omitted=[r['id'] for i,r in enumerate(context) if i not in chosen]
    points=[p for i,p in enumerate(points) if i in chosen];context=[r for i,r in enumerate(context) if i in chosen]
    ids={r['id']:i for i,r in enumerate(context)};relations=np.zeros((len(context),len(context)),dtype=np.int64)
    for key,i in ids.items():
        for target in lanes[key]['successors']:
            if target in ids:relations[i,ids[target]],relations[ids[target],i]=2,1
        for neighbor,value in (('left_neighbor',3),('right_neighbor',4)):
            if lanes[key].get(neighbor) in ids:relations[i,ids[lanes[key][neighbor]]]=value
    np.fill_diagonal(relations,5)
    for boundary in condition['boundaries']:
        if boundary['fixed'] and boundary['neighbor_id'] not in ids:raise ValueError('Fixed port missing from model field')
    return dict(context=context,points=np.asarray(points),relations=relations,center=center,rotation=rotation,
        conditioning_frame=dict(method=frame_method,field_m=float(cfg.fov)),
        conditioning_selection=dict(maximum_lanes=budget,required_ids=sorted(required),omitted_context_ids=omitted,
            method='Agent-selected budget policy: retain every mutable path and fixed boundary; prioritize graph-connected context by hop count, then spatial distance. Omitted conditioning roads remain unchanged in the full map.')),[ids[k] for k in condition['generated_path_ids']]


def sample(condition, condition_path, output, model, cfg, seed=0, start_t=9):
    import torch
    import pytorch_lightning as pl
    import time
    pl.seed_everything(seed, workers=True)
    torch.cuda.reset_peak_memory_stats()
    started=time.monotonic()
    prepared,indices=model_input(condition,cfg.dataset)
    output.mkdir(parents=True,exist_ok=True)
    with torch.no_grad(),model.ema.average_parameters():
        data=refinement_data(model,prepared,indices);mask=data['lane'].mask
        agents,lanes=data['agent'].latents.clone(),data['lane'].latents.clone();xa=agents[:,None].clone()
        t=torch.full((len(lanes),),start_t,device=model.device,dtype=torch.long)
        xl=model.diff_model.q_sample(lanes[:,None],t,torch.randn_like(lanes[:,None]));xl[mask]=lanes[mask,None]
        for step in reversed(range(start_t+1)):
            ta=torch.full((len(agents),),step,device=model.device,dtype=torch.long)
            tl=torch.full((len(lanes),),step,device=model.device,dtype=torch.long)
            xa,xl=model.diff_model.p_sample(xa,xl,data,ta,tl)
            xa=torch.clip(xa,-cfg.model.diffusion_clip,cfg.model.diffusion_clip)
            xl=torch.clip(xl,-cfg.model.diffusion_clip,cfg.model.diffusion_clip)
            xa[:]=agents[:,None];xl[mask]=lanes[mask,None]
        raw=decode(model,data,xa[:,0],xl[:,0],output,'raw_group')
        known_change=float((xl[mask,0]-lanes[mask]).abs().max())
    records=[]
    lookup={l['id']:l for l in condition['lanes']}
    for i,index in enumerate(np.asarray(raw['lane_ids'],dtype=int)):
        row=prepared['context'][index]
        if not row['generated']:continue
        xy=np.asarray(raw['road_points'][i])@prepared['rotation']+prepared['center']
        source=np.asarray(lookup[row['id']]['centerline'])[:,:2]
        records.append(dict(id=row['id'],centerline_xy_enu_m=xy.tolist(),
            raw_endpoint_displacement_from_source_m=np.linalg.norm(xy[[0,-1]]-source[[0,-1]],axis=1).tolist(),
            simple_xy=bool(LineString(xy).is_simple)))
    save(output/'raw_paths.json',dict(paths=records,condition=str(condition_path.resolve()),
        status='Unprojected generated paths; shared ports, fixed boundaries and steering remain to be checked.'))
    save(output/'sampling.json',dict(model='Official Scenario Dreamer Waymo large and autoencoder',seed=seed,
        method='Multi-path SDEdit conditional sampling with frozen known-road latents',start_t=start_t,reverse_steps=start_t+1,
        from_pure_noise=False,context_lanes=int(mask.sum()),generated_paths=len(records),known_latent_max_change=known_change,
        source_topology_used_as_condition=True,decoded_topology_applied=False,context=prepared['context'],
        input_center_enu_m=prepared['center'].tolist(),enu_to_model_rotation=prepared['rotation'].tolist(),
        conditioning_selection=prepared['conditioning_selection'],conditioning_frame=prepared['conditioning_frame'],
        synthetic_agent='One fixed encoder placeholder at 15 km/h, 4.7 x 1.85 m; not an observed AV2 traffic state or generated actor.',
        elapsed_seconds=time.monotonic()-started,peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
    torch.save(dict(agents=xa[:,0].cpu(),lanes=xl[:,0].cpu(),ids=data['lane'].ids.cpu(),mask=mask.cpu()),output/'latents.pt')
    print(dict(context=int(mask.sum()),generated=len(records),known_latent_max_change=known_change,
        endpoint_displacements=[r['raw_endpoint_displacement_from_source_m'] for r in records]),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    choice=p.add_mutually_exclusive_group(required=True)
    choice.add_argument('--condition',type=Path);choice.add_argument('--batch-root',type=Path)
    p.add_argument('--source-dir',type=Path,required=True);p.add_argument('--output',type=Path)
    p.add_argument('--reuse-sampling-from',type=Path,help='Reuse a prior group sample only when its actual model input arrays are identical')
    p.add_argument('--seed',type=int,default=0);p.add_argument('--start-t',type=int,default=9)
    a=p.parse_args()
    if a.condition and not a.output:p.error('--condition requires --output')
    sys.path.insert(0,str(a.source_dir.resolve()))
    import torch
    from models.scenario_dreamer_ldm import ScenarioDreamerLDM
    from utils.train_helpers import set_latent_stats
    cfg,ae_cfg=configuration(a.source_dir.resolve());cfg=set_latent_stats(cfg)
    torch.set_float32_matmul_precision('medium')
    model=ScenarioDreamerLDM.load_from_checkpoint(str(Path(os.environ['MAP_SD_CHECKPOINT']) if os.environ.get('MAP_SD_CHECKPOINT') else Path(cfg.eval.save_dir)/cfg.eval.run_name/'last.ckpt'),
        cfg=cfg,cfg_ae=ae_cfg,map_location='cpu').to('cuda').eval().requires_grad_(False)
    if a.condition:
        sample(json.loads(a.condition.read_text()),a.condition,a.output,model,cfg,a.seed,a.start_t)
        return
    records=json.loads((a.batch_root/'manifest.json').read_text())['records'];results=[]
    for row in records:
        if row['status']!='prepared':continue
        path=Path(row['condition']);output=path.parent/'generation'
        if (output/'sampling.json').exists() and (output/'latents.pt').exists():
            results.append(dict(region=row['region'],status='reused_existing_sampling'));continue
        try:
            condition=json.loads(path.read_text());reused=False
            previous=a.reuse_sampling_from/row['region'] if a.reuse_sampling_from else None
            if previous and (previous/'generation/latents.pt').exists():
                old,old_indices=model_input(json.loads((previous/'condition.json').read_text()),cfg.dataset)
                new,new_indices=model_input(condition,cfg.dataset)
                prior=json.loads((previous/'generation/sampling.json').read_text())
                reused=(old_indices==new_indices and prior['seed']==a.seed and prior['start_t']==a.start_t and
                    [r['id'] for r in old['context']]==[r['id'] for r in new['context']] and
                    all(np.array_equal(old[k],new[k]) for k in ('points','relations','center','rotation')))
                if reused:
                    shutil.copytree(previous/'generation',output,dirs_exist_ok=True)
                    save(output/'reuse.json',dict(source=str(previous/'generation'),condition=str(path),
                        reason='Actual model input arrays, identities, generated mask, seed and denoising steps unchanged.'))
            if not reused:sample(condition,path,output,model,cfg,a.seed,a.start_t)
            result=dict(region=row['region'],status='reused_identical_input' if reused else 'sampled')
        except ValueError as error:
            result=dict(region=row['region'],status='input_review_required',reason=str(error))
        results.append(result);print(result,flush=True)
        save(a.batch_root/'sampling_batch.json',dict(records=results))
    save(a.batch_root/'sampling_batch.json',dict(records=results))


if __name__=='__main__':main()
