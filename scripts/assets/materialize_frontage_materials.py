"""Convert reviewed Cosmos wall proposals to reusable native base-color tiles."""
import argparse,json
from pathlib import Path
import cv2,numpy as np


def overlap_periodic(tile, fraction=.125):
    """Crossfade overlapping opposite edge bands without reflecting detail."""
    for axis in (0, 1):
        data=np.moveaxis(tile,axis,0);n=len(data);overlap=max(2,int(round(n*fraction)))
        weight=(.5-.5*np.cos(np.linspace(0,np.pi,overlap))).reshape((-1,)+(1,)*(data.ndim-1))
        blend=data[-overlap:]*(1-weight)+data[:overlap]*weight
        tile=np.moveaxis(np.concatenate([blend,data[overlap:-overlap]],axis=0),0,axis)
    return tile


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();materials={}
    for path in sorted(a.root.glob('*/case.json')):
        spec=json.loads(path.read_text());generated=path.parent/'generated/generated_000.png'
        im=cv2.imread(str(generated))
        crop=spec['material'].get('generated_crop_xyxy')
        if crop:
            x0,y0,x1,y1=crop;im=im[y0:y1,x0:x1]
        h,w=im.shape[:2];side=min(h,w)
        raw=im[(h-side)//2:(h+side)//2,(w-side)//2:(w+side)//2].astype(np.float32)/255
        linear=np.where(raw<=.04045,raw/12.92,((raw+.055)/1.055)**2.4)
        luminance=linear@np.array([.0722,.7152,.2126],np.float32)
        broad=cv2.GaussianBlur(luminance,(0,0),max(side/12,1))
        detail=np.clip(luminance/np.maximum(broad,1e-5),.65,1.4)
        color=np.array(spec['material']['linear_color_choice'][::-1])
        tile=np.clip(detail[...,None]*color,0,1)
        tile=np.where(tile<=.0031308,tile*12.92,1.055*tile**(1/2.4)-.055)
        periodic=spec['material'].get('periodic_method','mirror')
        if periodic=='overlap':tile=overlap_periodic(tile)
        elif periodic=='mirror':
            tile=np.concatenate([tile,tile[:,::-1]],axis=1);tile=np.concatenate([tile,tile[::-1]],axis=0)
        elif periodic=='repeat':pass  # Agent-selected whole repeating source courses.
        else:raise ValueError('Unknown periodic_method: '+periodic)
        tile=np.rint(tile*255).astype(np.uint8)
        target=path.parent/'basecolor.png';cv2.imwrite(str(target),tile)
        materials[path.parent.name]={**spec['material'],'texture':str(target.resolve()),'generated':str(generated.resolve()),
            'source':spec['source'],'periodic_method':periodic,'processing':'Central square, low-frequency illumination normalization, explicit linear color choice, '+periodic+' periodic tile. Not measured albedo.'}
    (a.root/'materials.json').write_text(json.dumps({'materials':materials},indent=2)+'\n')


if __name__=='__main__':main()
