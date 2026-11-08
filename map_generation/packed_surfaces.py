"""Construct physical surfaces from a packed native network.

Shared cross sections retain source heights and interval identities through
longitudinal concatenation and return-road splits.
"""
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from map_generation.road_builder import (
    _fit_job, bridge_geometry, close_surface_connections,
)


def strip_mesh(left, right, **metadata):
    left, right = np.asarray(left), np.asarray(right)
    vertices = np.stack((left, right), axis=1).reshape(-1, 3)
    faces = []
    for j in range(0, len(vertices) - 2, 2):
        for face in ([j, j + 1, j + 3], [j, j + 3, j + 2]):
            tri = vertices[face]
            area = float(np.cross(tri[1] - tri[0], tri[2] - tri[0])[2])
            if abs(area) <= 1e-10:
                continue
            faces.append(face if area > 0 else face[::-1])
    return dict(metadata, vertices=vertices.tolist(), faces=faces,
                left_boundary=left.tolist(), right_boundary=right.tolist())


def grouped_meshes(mapping):
    """Split shared cross sections at each real source and native road span."""
    meshes = []
    for section in mapping['grouped_sections']:
        grid = np.asarray([s['s_m'] for s in section['samples']])
        origins = np.asarray([s['reference_xyz'][:2] for s in section['samples']])
        normals = np.asarray([s['normal_xy'] for s in section['samples']])
        offsets = np.asarray([s['boundary_offsets_m'] for s in section['samples']])
        heights = np.asarray([s['boundary_z_m'] for s in section['samples']])
        boundaries = np.concatenate((origins[:, None, :] + normals[:, None, :] * offsets[:, :, None],
                                     heights[:, :, None]), axis=2)
        parts = section.get('native_sections', [dict(
            road_id=section['source_lanes'][0]['road_id'], start_m=0., end_m=section['length_m'])])
        by_road = {p['road_id']: p for p in parts}
        for i, (entry, source) in enumerate(zip(section['source_lanes'], section['source_lane_records'])):
            for chain_part in source['source_chain']:
                key = chain_part['source_lane_id']
                for span in mapping['lane_to_road_intervals'][key]:
                    if span['junction_id'] is not None or span['road_id'] not in by_road:
                        continue
                    offset = by_road[span['road_id']]['start_m']
                    low, high = offset + span['road_s_start_m'], offset + span['road_s_end_m']
                    stations = np.unique(np.r_[low, grid[(grid > low) & (grid < high)], high])
                    edges = [np.column_stack([np.interp(stations, grid, boundaries[:, b, d])
                                               for d in range(3)]) for b in (i, i + 1)]
                    meshes.append(strip_mesh(*edges, id='group_surface_' + str(len(meshes)),
                        kind='native_group_surface', source_lane_id=key,
                        source_interval_m=[span['source_s_start_m'], span['source_s_end_m']],
                        xodr_road_id=span['road_id'], xodr_lane_id=entry['lane_id'], junction_id=None,
                        native_interval_m=[span['road_s_start_m'], span['road_s_end_m']],
                        boundary_source=source['source']))
    return meshes


def native_successors(path):
    """Read declared lane links for all classes, including bicycle surfaces.

    CARLA GenerateTopology iterates Driving lanes only. Physical road-cap
    closure also needs the retained Biking, Sidewalk and Restricted lanes.
    """
    context = ET.iterparse(path, events=('start', 'end'))
    _, root = next(context)
    edges = set()
    for event, element in context:
        if event != 'end' or element.tag not in ('road', 'junction'):
            continue
        if element.tag == 'road':
            rid = int(element.get('id'))
            sections = element.findall('lanes/laneSection')
            for side, section, direction in [('right', sections[-1], 'successor'),
                                              ('left', sections[0], 'predecessor')]:
                link = element.find('link/' + direction)
                if link is None or link.get('elementType') != 'road':
                    continue
                for lane in section.findall(side + '/lane'):
                    target = lane.find('link/' + direction)
                    if target is not None:
                        edges.add(((rid, int(lane.get('id'))),
                                   (int(link.get('elementId')), int(target.get('id')))))
        else:
            for connection in element.findall('connection'):
                for link in connection.findall('laneLink'):
                    edges.add(((int(connection.get('incomingRoad')), int(link.get('from'))),
                               (int(connection.get('connectingRoad')), int(link.get('to')))))
        root.remove(element)
        element.clear()
    return edges


# Linux workers inherit the read-only parsed map and fitted source curves.
# Each task samples one unique source interval; duplicate movements reuse it.
_SURFACE_NATIVE = None
_SURFACE_GEOMETRIES = None


def sample_piece(job):
    rid, key, start, end, cursor, length, junction, sample_step = job
    geometry = _SURFACE_GEOMETRIES[key]
    grid = np.unique(np.r_[np.linspace(0, end - start,
        max(3, math.ceil((end - start) / sample_step) + 1)),
        geometry.knots[(geometry.knots > start) & (geometry.knots < end)] - start])
    grid = grid[np.r_[True, np.diff(grid) > 1e-6]]
    left, right = [], []; max_error = 0.
    for local in grid:
        margin = min(1e-4, length * 1e-3)
        # Keep the native float32 query strictly inside long road endpoints.
        station = min(float(np.float32(np.clip(cursor + local, margin, length - margin))),
                    float(np.nextafter(np.float32(length),np.float32(0.))))
        waypoint = _SURFACE_NATIVE.get_waypoint_xodr(rid, -1, station)
        if waypoint is None:
            raise ValueError(f'Native surface sample missing: {rid} at {station}')
        p = waypoint.transform.location
        xyz = np.array([p.x, -p.y, p.z])
        forward = waypoint.transform.get_forward_vector()
        normal = np.array([forward.y, forward.x, 0.]); normal /= np.linalg.norm(normal)
        left.append(xyz + normal * waypoint.lane_width / 2)
        right.append(xyz - normal * waypoint.lane_width / 2)
        expected = geometry.xyz(np.clip(start + station - cursor, 0, geometry.length))
        max_error = max(max_error, float(np.linalg.norm(xyz - expected)))
    mesh=strip_mesh(left,right,source_lane_id=key,source_interval_m=[start,end],
        xodr_road_id=rid,xodr_lane_id=-1,junction_id=junction,
        native_interval_m=[cursor,cursor+end-start])
    return mesh,max_error,len(grid)


def uncached_geometry_keys(mapping, reuse):
    """A new native interval needs its curve even when source Z is unchanged."""
    required=set()
    for meta in mapping['roads'].values():
        if meta['kind']=='lane_group':continue
        pieces=([(meta['scene_lane_id'],*meta['source_interval_m'])]
                if meta['kind']=='lane' else meta['source_intervals'])
        for key,start,end in pieces:
            if (key,round(start,9),round(end,9)) not in reuse:required.add(key)
    sources={key for key in required if not key.startswith('bridge:')}
    for key in required:
        if key.startswith('bridge:'):sources.update(key[7:].split('->'))
    return required,sources


def build_surfaces(scene, roads, output, sample_step=.5, workers=1, previous_samples=None, height_revision=None, compiler_scene=None):
    import carla

    roads, output = Path(roads), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    mapping = json.loads((roads / 'road_lane_ids.json').read_text())
    native = carla.Map(scene['map_id'], (roads / 'roads.xodr').read_text())
    lanes = {l['id']: l for l in scene['fusion_map']['lanes']}
    alias_ids=[]
    if compiler_scene:
        reference=json.loads(Path(compiler_scene).read_text())
        aliases={l['id']:l for l in reference['fusion_map']['lanes']}
        required=set()
        for record in mapping['roads'].values():
            if record['kind']=='lane':required.add(record['scene_lane_id'])
            elif record['kind']=='movement':required.update(record['source_lane_path'])
        alias_ids=sorted(required-lanes.keys())
        for key in alias_ids:
            if aliases[key]['source']!='av2':
                raise ValueError('Absent non-observed compiler source needs explicit current geometry: '+key)
            lanes[key]=aliases[key]
        del reference,aliases
    previous=None;changed=None;reuse={}
    if previous_samples:
        revision=json.loads(Path(height_revision).read_text())
        if not revision['native_graph_XY_widths_and_permissions_unchanged']:
            raise ValueError('Surface reuse requires unchanged native geometry and topology')
        changed=set(revision['changed_source_lane_ids'])
        previous=json.loads(Path(previous_samples).read_text())
        if previous['sample_step']!=sample_step:raise ValueError('Surface cache sampling step differs')
        for mesh in previous['meshes'][previous['grouped_count']:]:
            key=mesh['source_lane_id']
            affected=key in changed or (key.startswith('bridge:')and any(k in changed for k in key[7:].split('->')))
            if not affected:reuse[(key,*[round(v,9)for v in mesh['source_interval_m']])]=mesh
        needed=set(changed)
        for meta in mapping['roads'].values():
            if meta['kind']!='movement':continue
            for a,b in zip(meta['source_lane_path'],meta['source_lane_path'][1:]):
                if a in changed or b in changed:needed.update([a,b])
    else:needed=set(lanes)
    required,uncached_sources=uncached_geometry_keys(mapping,reuse)
    needed.update(uncached_sources)
    geometries = {}
    jobs = ((key, lanes[key], 1.) for key in sorted(needed))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for key, geometry, _ in pool.map(_fit_job, jobs, chunksize=16):
            geometries[key] = geometry
    for meta in mapping['roads'].values():
        if meta['kind'] != 'movement':
            continue
        for a, b in zip(meta['source_lane_path'], meta['source_lane_path'][1:]):
            key = f'bridge:{a}->{b}'
            if changed is not None and a not in changed and b not in changed and key not in required:continue
            if key not in geometries:
                geometry = bridge_geometry(geometries[a], geometries[b])
                if geometry is not None:
                    geometries[key] = geometry
    meshes = grouped_meshes(mapping)
    paths = {}
    for index, mesh in enumerate(meshes):
        key = mesh['xodr_road_id'], mesh['xodr_lane_id']
        paths.setdefault(key, []).append(index)
    for indices in paths.values():
        indices.sort(key=lambda i: meshes[i]['native_interval_m'][0])
    grouped_count = len(meshes)
    seen = {}
    max_error = 0.
    samples = 0
    sample_jobs=[];reused_count=0
    for rid, meta in mapping['roads'].items():
        if meta['kind'] == 'lane_group': continue
        rid=int(rid)
        pieces=([(meta['scene_lane_id'],*meta['source_interval_m'])]
            if meta['kind']=='lane' else meta['source_intervals'])
        length=sum(b-a for _,a,b in pieces);cursor=0.;paths[rid,-1]=[]
        for key,start,end in pieces:
            identity=key,round(start,9),round(end,9)
            if identity not in seen:
                seen[identity]=grouped_count+len(sample_jobs)
                sample_jobs.append((rid,key,start,end,cursor,length,meta['junction_id'],sample_step))
            paths[rid,-1].append(seen[identity]);cursor+=end-start
    global _SURFACE_NATIVE,_SURFACE_GEOMETRIES
    _SURFACE_NATIVE=native;_SURFACE_GEOMETRIES=geometries
    pending=[];cached={}
    for i,job in enumerate(sample_jobs):
        _,key,start,end,*_=job;identity=key,round(start,9),round(end,9)
        if identity in reuse:
            mesh=reuse[identity];cached[i]=mesh;reused_count+=1
        else:pending.append(job)
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('fork'))as pool:
        computed=iter(pool.map(sample_piece,pending,chunksize=16))
        for i in range(len(sample_jobs)):
            if i in cached:
                mesh=dict(cached.pop(i));error=previous['max_error'];count=len(mesh['left_boundary'])
                rid,key,start,end,cursor,length,junction,_=sample_jobs[i]
                mesh.update(xodr_road_id=rid,xodr_lane_id=-1,junction_id=junction,
                            native_interval_m=[cursor,cursor+end-start])
            else:mesh,error,count=next(computed)
            mesh['id']='native_surface_'+str(len(meshes));meshes.append(mesh)
            max_error=max(max_error,error);samples+=count
            if (i+1)%5000==0:print(dict(stage='packed_physical_samples',completed=i+1,total=len(sample_jobs)),flush=True)
    _SURFACE_NATIVE=None;_SURFACE_GEOMETRIES=None
    successors = {key: set() for key in paths}
    for first, second in native_successors(roads / 'roads.xodr'):
        successors[first].add(second)
    sampled=dict(meshes=meshes,paths=[[list(k),v]for k,v in paths.items()],
        successors=[[list(k),[list(x)for x in v]]for k,v in successors.items()],
        sample_step=sample_step,grouped_count=grouped_count,max_error=max_error,samples=samples,
        reused_intervals=reused_count,newly_sampled_intervals=len(pending),
        source_roads=str(roads.resolve()),compiler_geometry_scene=str(compiler_scene)if compiler_scene else None,
        compiler_geometry_aliases=alias_ids,compiler_alias_scope='Sampling geometry only; no published source graph or native topology changes.')
    (output/'sampled_surfaces.json').write_text(json.dumps(sampled,separators=(',',':'))+'\n')
    return close_sampled_surfaces(sampled,output)


def close_sampled_surfaces(sampled,output):
    output=Path(output);meshes=list(sampled['meshes'])
    paths={tuple(k):v for k,v in sampled['paths']}
    successors={tuple(k):{tuple(x)for x in v}for k,v in sampled['successors']}
    sample_step=sampled['sample_step'];grouped_count=sampled['grouped_count']
    max_error=sampled['max_error'];samples=sampled['samples']
    closure = close_surface_connections(meshes, paths, successors)
    report = dict(group_surface_count=grouped_count, mesh_count=len(meshes),
                  individual_lane_samples=samples, max_individual_native_source_error_m=max_error,
                  closure_count=closure['patch_count'],
                  zero_area_cap_records=sum(not m['faces']for m in meshes),
                  scope='Native bodies and movement surfaces with source lineage. AV2 polygon restoration, contact joining, terrain and physical driving follow.')
    result = dict(frame='SceneIR/Blender ENU metres', sampling_step_m=sample_step,
                  meshes=meshes, connection_closure=closure, packed_surface_report=report,
                  surface_scope='Shared measured or inferred multi-lane boundaries retain crossfall; other native lanes use horizontal cross sections.')
    (output / 'road_meshes.json').write_text(json.dumps(result, separators=(',', ':')) + '\n')
    (output / 'review.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
