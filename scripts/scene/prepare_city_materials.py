"""Register reviewed city appearance references and reuse material conditioning."""
import argparse,json,subprocess,sys
from pathlib import Path
from map_generation.information_graph import add_node,add_edge,compile_conditions


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('scene','graph','selection','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();scene=json.loads(a.scene.read_text());graph=json.loads(a.graph.read_text());selection=json.loads(a.selection.read_text())
    task='task:city_material_appearance';add_node(graph,dict(id=task,kind='simulation_task'))
    for family in selection['families']:
        key='material:'+family['name'];reference='material_reference:'+family['name']
        add_node(graph,dict(id=reference,kind='source_image',content=dict(image=family['source_image'],crop_xyxy=family['crop_xyxy'],area_id=family['area_id'])))
        add_node(graph,dict(id=key,kind='material_family',content=family['description'],appearance=dict(family=family['name'],source_image=family['source_image']),
            task_status={task:dict(complete=False,missing=['native_material_review'])}))
        add_edge(graph,key,'supported_by',reference,status='reviewed_AV2_material_crop')
        add_edge(graph,task,'depends_on',key)
        graph['decisions'].append(dict(source_id=key,task=task,choice='enhance',agent=selection['agent'],
            reason=selection['reason'],required_properties=family['required_properties'],evidence_ids=[reference]))
    a.output.mkdir(parents=True,exist_ok=True)
    conditions=compile_conditions(graph,scene,task)
    for name,value in [('graph.json',graph),('conditions.json',conditions)]:
        (a.output/name).write_text(json.dumps(value,separators=(',',':'))+'\n')
    subprocess.run([sys.executable,'-m','scripts.assets.prepare_frontage_materials','--selection',str(a.selection),
        '--output',str(a.output/'materials')],check=True)


if __name__=='__main__':main()
