"""Inspect and update task-linked traffic information and construction conditions."""
import argparse
import json
from pathlib import Path
from map_generation.information_graph import add_node, add_edge, compile_conditions, enrich, review, task_packet, attach_runtime_coverage


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['inspect','decide','review','enrich','compile','runtime','revise'])
    p.add_argument('--graph',type=Path,required=True)
    p.add_argument('--scene',type=Path)
    p.add_argument('--run',type=Path)
    p.add_argument('--revision',type=Path,help='Actual externally reviewed nodes, relations and decisions')
    p.add_argument('--id',action='append',default=[])
    p.add_argument('--hops',type=int,default=2)
    p.add_argument('--task',default='source_faithful_simulation')
    p.add_argument('--choice',choices=['retain','simplify','enhance','seek_evidence'])
    p.add_argument('--reason');p.add_argument('--agent',default='construction_agent')
    p.add_argument('--properties',nargs='*',default=[])
    p.add_argument('--realization',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args();data=json.loads(a.graph.read_text());nodes={n['id']:n for n in data['nodes']}
    task=a.task if a.task.startswith('task:') else 'task:'+a.task
    if a.action=='enrich':
        data=enrich(data,json.loads(a.scene.read_text()),a.scene.resolve())
        result={'nodes':len(data['nodes']),'edges':len(data['edges']),'source_scene':data['source_scene']}
    elif a.action=='inspect':
        ids=a.id or [n['id'] for n in data['nodes'] if n.get('realization') in ('pending','needs_review')]
        result=task_packet(data,ids,a.hops)
    elif a.action=='compile':
        scene=json.loads((a.scene or Path(data['source_scene'])).read_text())
        result=compile_conditions(data,scene,task)
    elif a.action=='decide':
        if not a.choice or not a.reason or not a.id:
            raise ValueError('Supply actual agent choice, reason and source IDs')
        if task not in nodes:data['nodes'].append({'id':task,'kind':'simulation_task'})
        for key in a.id:
            nodes[key]
            decision={'source_id':key,'choice':a.choice,'reason':a.reason,'agent':a.agent,'task':task,
                'required_properties':a.properties,
                'evidence_ids':[e['target'] for e in data['edges'] if e['source']==key and e['relation']=='supported_by']}
            data['decisions'].append(decision)
            add_edge(data,task,'requires_review' if a.choice=='seek_evidence' else 'depends_on',key,
                     decision_index=len(data['decisions'])-1)
        result=data['decisions'][-len(a.id):]
    elif a.action=='revise':
        revision=json.loads(a.revision.read_text())
        if a.scene:
            data['source_scene']=str(a.scene.resolve())
        for node in revision.get('nodes',[]):add_node(data,node)
        for edge in revision.get('edges',[]):add_edge(data,**edge)
        for decision in revision.get('decisions',[]):
            data['decisions'].append(decision)
            add_edge(data,decision['task'],'depends_on',decision['source_id'],
                     decision_index=len(data['decisions'])-1)
        result={'revision':str(a.revision),'nodes_added_or_updated':len(revision.get('nodes',[])),
                'decisions':revision.get('decisions',[])}
    elif a.action=='runtime':
        result=attach_runtime_coverage(data,json.loads(a.run.read_text()),a.run.resolve())
    else:
        result=review(data,json.loads(a.realization.read_text()),a.realization.resolve())
    if a.action in ('enrich','decide','review','runtime','revise'):
        a.graph.write_text(json.dumps(data,indent=2)+'\n')
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(result,indent=2)+'\n');print(a.output)
    else:print(json.dumps(result,indent=2))


if __name__=='__main__':main()
