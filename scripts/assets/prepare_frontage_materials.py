"""Prepare reviewed AV2 crops as material-only Cosmos transfer conditions."""
import argparse,json
from pathlib import Path
import cv2,numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--history',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True,help='Reviewed source crops, descriptions and material colors')
    p.add_argument('--straight-boards',action='store_true',help='Native-style review feedback: constrain timber to continuous vertical boards')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    frame=json.loads(a.history.read_text())['frames'][-1] if a.history else {}
    # Use reviewed source crops as material-appearance references.
    selection=json.loads(a.selection.read_text());specs=selection['families']
    material_context=selection.get('material_context','the observed streetscape')
    cases=[]
    for index,spec in enumerate(specs):
        name,crop,description,color=[spec[k] for k in ['name','crop_xyxy','description','linear_color_choice']]
        source_image=spec.get('source_image',frame.get('source_image'))
        if source_image is None:p.error('Each material needs source_image when --history is absent')
        source=cv2.imread(source_image)
        out=a.output/name;out.mkdir(exist_ok=True)
        x0,y0,x1,y1=crop;raw=source[y0:y1,x0:x1]
        cv2.imwrite(str(out/'av2_reference.png'),raw)
        # Resize the crop while retaining its low-frequency color and texture.
        control=cv2.GaussianBlur(cv2.resize(raw,(1024,768),interpolation=cv2.INTER_CUBIC),(0,0),5)
        cv2.imwrite(str(out/'control.png'),control)
        case={'prompt':{'description':description+' Fill the full frame with the same material. Neutral uniform studio illumination, no cast shadows, physically plausible microtexture for '+spec.get('material_context',material_context)+'.'},
              'negative_prompt':{'description':'windows, doors, building silhouette, street, sky, trees, people, vehicles, signs, lettering, logos, perspective, deep shadows, bright lighting gradient, facade grid'},
              'control_images':{'blur':[str((out/'control.png').resolve())]},
              'width':1024,'height':768,'steps':50,'seed':selection.get('seed',0)+index,'control_guidance':1.5,
              'source':{'image':source_image,'crop_xyxy':crop,'selection':str(a.selection),'area_id':spec.get('area_id'),'material_context':spec.get('material_context',material_context),'kind':'AV2 reviewed material reference; low-frequency condition only'},
              'material':{'family':name,'period_m':spec.get('period_m',2.0),'linear_color_choice':color,
                          'scope':'Inferred reusable base color, not measured albedo. Local architectural geometry is constructed independently from OSM.'}}
        layout=spec.get('layout')
        if layout and layout['kind']=='horizontal_brick':
            # Combine the selected material layout with source color and grain.
            edges=np.zeros((768,1024,3),np.uint8)
            bw,bh=layout['brick_width_px'],layout['brick_height_px']
            for row,y in enumerate(range(0,768,bh)):
                cv2.line(edges,(0,y),(1023,y),(255,255,255),2)
                for x in range(-bw,1024,bw):
                    xx=x+(bw//2 if row%2 else 0)
                    cv2.line(edges,(xx,y),(xx,min(y+bh,767)),(255,255,255),2)
            cv2.imwrite(str(out/'brick_edges.png'),edges)
            control[:]=np.median(raw.reshape(-1,3),axis=0).astype(np.uint8)
            cv2.imwrite(str(out/'control.png'),control)
            case['control_images']['edge']=[str((out/'brick_edges.png').resolve())]
            case['feedback']=layout['reason']
        if name=='timber_siding' and a.straight_boards:
            edges=np.zeros((768,1024,3),np.uint8)
            for x in range(48,1024,96):cv2.line(edges,(x,0),(x,767),(255,255,255),2)
            cv2.imwrite(str(out/'board_edges.png'),edges)
            case['control_images']['edge']=[str((out/'board_edges.png').resolve())]
            case['prompt']['description']='Continuous narrow vertical wooden boards, each board running from the very top to the very bottom of the frame, muted brown painted wood grain, soft uniform illumination, flat orthographic material sample.'
            case['negative_prompt']['description']+=', horizontal seams, checkerboard pattern, square panels, parquet, bricks, woven blocks'
            case['feedback']='Prior timber proposal invented alternating square panels. Add a vertical-board layout condition; this is an inferred material layout, not surveyed architecture.'
        (out/'case.json').write_text(json.dumps(case,indent=2)+'\n');cases.append(str(out/'case.json'))
    (a.output/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')


if __name__=='__main__':main()
