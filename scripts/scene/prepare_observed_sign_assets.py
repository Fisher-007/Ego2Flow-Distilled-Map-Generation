"""Rectify graph-selected sign images into textures and panel meshes."""
import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np


def prepare(conditions, output, max_side=256):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result = copy.deepcopy(conditions)
    records = []
    for row in result["native_objects"]:
        construction = row.get("native_construction", {})
        if row["choice"] not in ("retain", "enhance") or construction.get("kind") != "source_sign_face":
            continue
        spec = construction["settings"]
        observation = spec.get("texture_observation")
        if observation is None:
            continue
        corners = np.asarray(observation["corners_uv"], dtype=np.float32)
        image = cv2.imread(observation["image"])
        if image is None:
            raise FileNotFoundError(observation["image"])
        ratio = spec["width_m"] / spec["height_m"]
        width, height = (max_side, round(max_side / ratio)) if ratio >= 1 else (round(max_side * ratio), max_side)
        target = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
        transform = cv2.getPerspectiveTransform(corners, target)
        texture = cv2.warpPerspective(image, transform, (width, height), flags=cv2.INTER_CUBIC)
        name = spec["id"].replace(":", "_")
        texture_path = output / (name + ".png")
        cv2.imwrite(str(texture_path), texture)
        # Centimetre unit panel: X follows the measured left-to-right tangent,
        # Z is vertical, and UVs preserve the source face's four-corner order.
        mesh_path = output / (name + ".obj")
        outline = spec.get("face_outline_uv")
        if outline is None:
            mesh_path.write_text("o observed_sign_face\nv -50 0 -50\nv 50 0 -50\nv 50 0 50\nv -50 0 50\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nf 1/1 3/3 2/2\nf 1/1 4/4 3/3\n")
        else:
            # Clip the face to the measured outline within its bounding frame.
            lines = ["o observed_sign_face"]
            for u, v in outline:
                lines.append("v %.9g 0 %.9g" % (100*(u-.5), 100*(.5-v)))
            lines.extend("vt %.9g %.9g" % (u, 1-v) for u, v in outline)
            lines.extend("f 1/1 %d/%d %d/%d" % (i, i, i+1, i+1)
                         for i in range(2, len(outline)))
            mesh_path.write_text("\n".join(lines) + "\n")
        # Use the measured face outline for the collision body.
        body_outline = outline or [[0, 0], [1, 0], [1, 1], [0, 1]]
        count = len(body_outline)
        back_path = output / (name + "_back.obj")
        body = ["o observed_sign_back"]
        for depth in (-1., 1.):
            body.extend("v %.9g %.9g %.9g" % (100*(u-.5), depth, 100*(.5-v))
                        for u, v in body_outline)
        for i in range(1, count-1):
            body.append("f 1 %d %d" % (i+2, i+1))
            body.append("f %d %d %d" % (count+1, count+i+1, count+i+2))
        for i in range(count):
            j = (i+1) % count
            body.extend(("f %d %d %d" % (i+1, j+1, count+j+1),
                         "f %d %d %d" % (i+1, count+j+1, count+i+1)))
        back_path.write_text("\n".join(body) + "\n")
        record = dict(source_id=spec["id"], source_observation=observation,
                      texture=str(texture_path.resolve()), mesh=str(mesh_path.resolve()),
                      texture_size_px=[width, height], homography=transform.tolist(),
                      face_outline_uv=outline, backing_mesh=str(back_path.resolve()),
                      backing_thickness_m=.02,
                      source_edge_lengths_px=[float(np.linalg.norm(corners[(i+1)%4]-corners[i])) for i in range(4)],
                      scope="Perspective rectification of the original observed sign. Upsampling adds no source detail; no generated text or policy authority.")
        spec["source_face_asset"] = record
        records.append(record)
    (output / "assets.json").write_text(json.dumps(dict(objects=records), indent=2) + "\n")
    (output / "conditions.json").write_text(json.dumps(result, indent=2) + "\n")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-side", type=int, default=256)
    args = parser.parse_args()
    records = prepare(json.loads(args.conditions.read_text()), args.output, args.max_side)
    print(dict(observed_faces=len(records), assets=str(args.output / "assets.json")), flush=True)


if __name__ == "__main__":
    main()
