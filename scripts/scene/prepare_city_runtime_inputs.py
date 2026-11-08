"""Export map-owned, independently configurable source-time Area runtime inputs."""
import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('states','handoff','actor-layers','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    read=lambda p:json.loads(p.read_text())
    review=read(a.states/'review.json');routes=read(a.handoff/'candidate_routes.json')
    areas=read(a.handoff/'area_index.json');info=read(a.handoff/'map_info.json')
    signals=read(a.handoff/'signal_programs.json');records=[]
    for area in areas:
        key=area['area_id'];anchor=read(a.actor_layers/key/'runtime_context.json')
        choices=[r for r in review['area_states'] if r['area_id']==key and r['log_id']==anchor['source_log']]
        if len(choices)!=1:raise ValueError('Expected one declared source epoch for '+key)
        selected=choices[0];folder=a.output/key;dest=folder/'handoff';dest.mkdir(parents=True,exist_ok=True)
        source_anchor_delta=(selected['timestamp_ns']-anchor['source_timestamp_ns'])/1e9
        states=read(Path(selected['source_file']))
        sample=next(r for r in states['samples'] if r['timestamp_ns']==selected['timestamp_ns'])
        config=dict(area_id=key,map=info['carla_map_name'],route=key,parked_context=False,
            initial_distribution_source=dict(file=str(Path(selected['source_file']).resolve()),
                source_timestamp_ns=selected['timestamp_ns'],source_log=selected['log_id'],
                source_anchor_difference_s=source_anchor_delta,
                selection='Nearest prepared joint annotation sample from this Area own declared source log; no outcome-based reselection.'),
            actor_modes=dict(observed_replay='Original source cuboids and past-only velocities; native fit is separate.',
                controller='Movable source participants are replaceable; valid native candidates and actual spawn/support checks required.',
                parked_context='Removable optional source-time stationary background; never baked into the map.'),
            source_actor_layers=str((a.actor_layers/key).resolve()),
            native_context_status='pending_support_and_spawn',
            scope='Map runtime handoff preparation only; actual driving is checked separately.')
        for name,data in [('map_info',info),('candidate_routes',{key:routes[key]}),
                          ('area_index',[area]),('signal_programs',signals)]:
            (dest/(name+'.json')).write_text(json.dumps(data,separators=(',',':'))+'\n')
        (folder/'scenario.json').write_text(json.dumps(config,indent=2)+'\n')
        (folder/'source_joint_state.json').write_text(json.dumps(sample,separators=(',',':'))+'\n')
        row=dict(area_id=key,source_log=selected['log_id'],timestamp_ns=selected['timestamp_ns'],
            source_anchor_difference_s=source_anchor_delta,
            traffic_count=len(sample['actors']),parked_count=len(sample['parked_context']),
            ambiguous_traffic=sum(r['source_lane_review']['assignment_reason']!='unique' for r in sample['actors']),
            traffic_without_native_candidates=sum(not r['native_lane_candidates'] for r in sample['actors']),
            folder=str(folder),native_prepared=False)
        records.append(row);print(row,flush=True)
    if len(records)!=len(routes):raise ValueError('Area inputs and all declared routes differ')
    (a.output/'review.json').write_text(json.dumps(dict(areas=records,configured_areas=len(areas),
        prepared_areas=len(records),scope=__doc__,native_runtime_passed=False),indent=2)+'\n')

if __name__=='__main__':main()
