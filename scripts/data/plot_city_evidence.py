"""Prepare source camera contact sheets with exact file/timestamp provenance."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("config","data-root","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--cameras",nargs="+",default=["ring_front_left","ring_front_center","ring_front_right"])
    args=parser.parse_args()
    config=json.loads(args.config.read_text());args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for area in config["areas"]:
        source=area["source_observations"][0]
        folder=args.data_root/"av2/sensor"/source["split"]/source["log_id"]
        selected=[]
        for camera in args.cameras:
            images=list((folder/"sensors/cameras"/camera).glob("*.jpg"))
            chosen=min(images,key=lambda path:abs(int(path.stem)-source["nearest_timestamp_ns"]))
            selected.append({"camera":camera,"file":str(chosen),"timestamp_ns":int(chosen.stem),
                             "area_anchor_difference_s":(int(chosen.stem)-source["nearest_timestamp_ns"])/1e9})
        rows.append({"area_id":area["id"],"candidate_id":area["candidate_id"],"log_id":source["log_id"],"split":source["split"],"views":selected})
    for start in range(0,len(rows),5):
        group=rows[start:start+5];width,height=384,320
        canvas=Image.new("RGB",(width*len(args.cameras),height*len(group)),"white");draw=ImageDraw.Draw(canvas)
        for y,row in enumerate(group):
            for x,view in enumerate(row["views"]):
                with Image.open(view["file"]) as source:
                    picture=source.convert("RGB");picture.thumbnail((width-8,height-38))
                    canvas.paste(picture,(x*width+(width-picture.width)//2,y*height+35+(height-38-picture.height)//2))
                draw.text((x*width+5,y*height+4),row["area_id"]+" | "+view["camera"],fill="black")
                draw.text((x*width+5,y*height+19),row["log_id"][:8]+f" | dt={view['area_anchor_difference_s']:+.2f}s",fill="black")
        canvas.save(args.output/f"areas_{start+1:02d}_{start+len(group):02d}.jpg",quality=94)
    (args.output/"views.json").write_text(json.dumps({"areas":rows,
        "scope":"Sparse AV2 source views for construction review; original camera aspect and source paths retained. Views can differ by up to the acquisition sampling interval and are not a synchronized policy input."},indent=2)+"\n")
    print({"areas":len(rows),"views":sum(len(row["views"]) for row in rows)})


if __name__=="__main__":main()
