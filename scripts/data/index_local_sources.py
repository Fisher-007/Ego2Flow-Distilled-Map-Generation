"""Index existing AV2 vector maps and poses without network access."""
import argparse
import json
from pathlib import Path

from scripts.data.av2_source import CITIES


def index(root, city, splits, log_ids=()):
    root = Path(root)
    rows = []
    for split in splits:
        for vector in sorted((root / 'sensor' / split).glob('*/map/log_map_archive_*.json')):
            identity = vector.parents[1].name
            if log_ids and identity not in log_ids:
                continue
            parts = vector.name.split('____')
            if len(parts) < 2 or parts[1].split('_')[0] != city:
                continue
            pose = vector.parents[1] / 'city_SE3_egovehicle.feather'
            if not pose.is_file():
                raise FileNotFoundError(pose)
            lanes = list(json.loads(vector.read_text())['lane_segments'].values())
            points = [p for lane in lanes for side in ('left_lane_boundary', 'right_lane_boundary') for p in lane[side]]
            bounds = [min(p[k] for p in points) for k in ('x', 'y')] + [max(p[k] for p in points) for k in ('x', 'y')]
            rows.append(dict(city_code=city, city=CITIES[city], split=split, log_id=identity,
                             vector_map=str(vector.relative_to(root)), bbox_city_xy=bounds,
                             source=str(vector.parents[1].resolve()), lane_count=len(lanes),
                             intersection_lane_count=sum(lane['is_intersection'] for lane in lanes),
                             center_city_xy=[(bounds[0]+bounds[2])/2, (bounds[1]+bounds[3])/2]))
    if not rows:
        raise ValueError('No matching local vector maps with poses were found')
    missing = set(log_ids) - {row['log_id'] for row in rows}
    if missing:
        raise ValueError('Requested source logs were not found: ' + ', '.join(sorted(missing)))
    return dict(city=city, splits=list(splits), candidates=rows, indexed_logs=len(rows),
                failures=[], scope='Local source inventory; region selection and source-image review follow.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True, help='AV2 root, containing sensor/')
    parser.add_argument('--city', choices=sorted(CITIES), required=True)
    parser.add_argument('--split', nargs='+', choices=['train', 'val', 'test'], default=['train', 'val'])
    parser.add_argument('--log-id', action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = index(args.data_root, args.city, args.split, args.log_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'indexed_logs': result['indexed_logs'], 'catalog': str(args.output)}))


if __name__ == '__main__':
    main()
