"""Preserve source-bound signal controls when native roads are rebuilt or split."""
from copy import deepcopy
import math
import xml.etree.ElementTree as ET

import numpy as np


def capture_signals(root, mapping):
    roads = {int(r.get('id')): deepcopy(r) for r in root.findall('road')
             if r.find('signals') is not None and len(r.find('signals'))}
    spans = {rid: [] for rid in roads}
    for key, records in mapping['lane_to_road_intervals'].items():
        for record in records:
            if record['road_id'] in spans:
                spans[record['road_id']].append((key, deepcopy(record)))
    return dict(roads=roads, spans=spans,
                controllers=[ET.tostring(e) for e in root.findall('controller')])


def _station(record, value, source_to_road=False):
    road = record.get('road_stations_m', [record['road_s_start_m'], record['road_s_end_m']])
    source = record.get('source_stations_m', [record['source_s_start_m'], record['source_s_end_m']])
    return float(np.interp(value, source if source_to_road else road,
                           road if source_to_road else source))


def _polynomial(elements, station, attribute):
    selected = [e for e in elements if float(e.get(attribute)) <= station + 1e-8]
    if not selected:
        return 0.
    element = selected[-1]
    delta = station - float(element.get(attribute))
    return sum(float(element.get(k, '0')) * delta ** i for i, k in enumerate('abcd'))


def _lane_center_t(road, lane_id, station):
    section = [e for e in road.findall('lanes/laneSection')
               if float(e.get('s')) <= station + 1e-8][-1]
    local = station - float(section.get('s'))
    offset = _polynomial(road.findall('lanes/laneOffset'), station, 's')
    side = 'right' if lane_id < 0 else 'left'
    direction = -1 if lane_id < 0 else 1
    lanes = {int(e.get('id')): e for e in section.findall(side + '/lane')}
    for index in range(1, abs(lane_id) + 1):
        width = _polynomial(lanes[direction * index].findall('width'), local, 'sOffset')
        offset += direction * width * (.5 if index == abs(lane_id) else 1.)
    return offset


def _geometry(road):
    if road is None:
        return None
    return (road.get('length'), tuple(ET.tostring(road.find(k)) if road.find(k) is not None else b''
            for k in ('planView', 'elevationProfile', 'lateralProfile', 'lanes')))


def restore_signals(root, mapping, captured, source_xodr):
    """Transfer signal validities through source stations and split branches.

    Additional movements use references to the same signal definition.
    """
    roads = {int(e.get('id')): e for e in root.findall('road')}
    changed = {rid for rid, road in captured['roads'].items()
               if _geometry(road) != _geometry(roads.get(rid))}
    native = None
    restored = []
    # Recreate original signal containers. The geometry compiler creates no
    # new signal identities; controllers and phase definitions stay untouched.
    for road in roads.values():
        element = road.find('signals')
        if element is not None:
            road.remove(element)
    for rid, old_road in captured['roads'].items():
        if rid not in changed:
            roads[rid].append(deepcopy(old_road.find('signals')))
            continue
        for element in old_road.find('signals'):
            old_s = float(element.get('s'))
            validities = element.findall('validity')
            old_lanes = {lane for v in validities for lane in
                         range(int(v.get('fromLane')), int(v.get('toLane')) + 1)}
            if not old_lanes:
                old_lanes = {int(e.get('id')) for e in old_road.findall('lanes/laneSection/*/lane')
                             if int(e.get('id')) != 0}
            destinations = {}
            for key, old in captured['spans'][rid]:
                if old['lane_id'] not in old_lanes or not (
                        old['road_s_start_m'] - 1e-6 <= old_s <= old['road_s_end_m'] + 1e-6):
                    continue
                source_s = _station(old, old_s)
                for target in mapping['lane_to_road_intervals'][key]:
                    if not target['source_s_start_m'] - 1e-6 <= source_s <= target['source_s_end_m'] + 1e-6:
                        continue
                    new_s = _station(target, source_s, True)
                    new_rid, new_lane = target['road_id'], target['lane_id']
                    identity = (new_rid, round(new_s, 7))
                    entry = destinations.setdefault(identity, dict(road_id=new_rid, s_m=new_s,
                        lanes=set(), sources=[], old_lane_id=old['lane_id']))
                    entry['lanes'].add(new_lane)
                    entry['sources'].append(dict(source_lane_id=key, source_s_m=source_s))
            if not destinations:
                raise ValueError('Signal has no retained source station: ' + str((rid, element.attrib)))
            pose = element.find('positionInertial')
            if element.tag == 'signal' and pose is None:
                if native is None:
                    import carla
                    native = carla.Map('signal_transfer_source', source_xodr.read_text())
                landmark = native.get_all_landmarks_from_id(element.get('id'))[0]
                transform = landmark.transform
                pose = ET.Element('positionInertial', dict(
                    x=str(transform.location.x), y=str(-transform.location.y), z=str(transform.location.z),
                    hdg=str(-math.radians(transform.rotation.yaw)),
                    pitch=str(math.radians(transform.rotation.pitch)),
                    roll=str(math.radians(transform.rotation.roll))))
            for index, destination in enumerate(sorted(destinations.values(), key=lambda d: (d['road_id'], d['s_m']))):
                target = roads[destination['road_id']]
                new = deepcopy(element)
                if element.tag == 'signal' and index:
                    new = ET.Element('signalReference', {k: element.get(k) for k in ('id', 'orientation')})
                for validity in new.findall('validity'):
                    new.remove(validity)
                new.set('s', str(destination['s_m']))
                # Retain the offset from the same source lane center. Display
                # geometry is independently fixed by positionInertial.
                old_t = _lane_center_t(old_road, destination['old_lane_id'], old_s)
                target_t = _lane_center_t(target, min(destination['lanes']), destination['s_m'])
                new.set('t', str(float(element.get('t', '0')) - old_t + target_t))
                for lane in sorted(destination['lanes']):
                    ET.SubElement(new, 'validity', fromLane=str(lane), toLane=str(lane))
                if new.tag == 'signal' and new.find('positionInertial') is None:
                    new.append(deepcopy(pose))
                container = target.find('signals')
                if container is None:
                    container = ET.SubElement(target, 'signals')
                container.append(new)
                restored.append(dict(signal_id=new.get('id'), kind=new.tag, old_road_id=rid,
                    old_s_m=old_s, road_id=destination['road_id'], s_m=destination['s_m'],
                    lane_ids=sorted(destination['lanes']), sources=destination['sources']))
    before_ids = sorted(e.get('id') for road in captured['roads'].values()
                        for e in road.findall('signals/signal'))
    after_ids = sorted(e.get('id') for e in root.findall('road/signals/signal'))
    if before_ids != after_ids:
        raise ValueError('Signal definitions changed during source remapping')
    if captured['controllers'] != [ET.tostring(e) for e in root.findall('controller')]:
        raise ValueError('Signal controllers changed during source remapping')
    references = root.findall('road/signals/signalReference')
    missing = sorted({e.get('id') for e in references} - set(after_ids))
    if missing:
        raise ValueError('Unresolved signal references: ' + str(missing))
    return dict(definitions_before=len(before_ids), definitions_after=len(after_ids),
        references_after=len(references), changed_signal_owning_roads=sorted(changed),
        remapped_records=restored, controllers_exact=True, missing_definitions=[],
        display_policy='Existing inertial transforms retained; changed road-relative definitions retain their actual original CARLA landmark transform.',
        scope='Source-station control encoding and display preservation; native control activation remains a runtime check.')


def refresh_binding_manifest(manifest, transfer, native, root):
    """Update runtime bindings from recorded source-station transfers."""
    result = deepcopy(manifest)
    changed = []
    records = transfer['remapped_records']
    roads = {int(e.get('id')): e for e in root.findall('road')}
    for binding in result['bindings']:
        approaches = []
        for old in binding['approaches']:
            matches = [record for record in records
                       if record['signal_id'] == binding['signal_id']
                       and record['old_road_id'] == old['road_id']
                       and abs(record['old_s_m'] - old['s_m']) < 1e-6
                       and old.get('external_source_lane_id', old['lane_id'])
                       in {s['source_lane_id'] for s in record['sources']}]
            if not matches:
                approaches.append(old)
                continue
            new_rows = []
            for record in matches:
                for lane_id in record['lane_ids']:
                    waypoint = native.get_waypoint_xodr(record['road_id'], lane_id, record['s_m'])
                    if waypoint is None:
                        raise ValueError('Transferred signal station is not a native waypoint: ' + str(record))
                    if (str(waypoint.lane_type) == 'Driving') != old['native_trigger_supported']:
                        raise ValueError('Signal transfer changed the source approach vehicle role')
                    row = deepcopy(old)
                    row['previous_native_interval'] = {key: old[key] for key in (
                        'road_id', 'xodr_lane_id', 'section_id', 's_m', 'width_m',
                        'reference_t_m', 'position_local_xyz', 'heading_rad')}
                    transform = waypoint.transform
                    row.update(road_id=record['road_id'], xodr_lane_id=lane_id,
                               section_id=waypoint.section_id, s_m=record['s_m'],
                               width_m=waypoint.lane_width,
                               reference_t_m=_lane_center_t(roads[record['road_id']], lane_id, record['s_m']),
                               position_local_xyz=[transform.location.x, -transform.location.y, transform.location.z],
                               heading_rad=-math.radians(transform.rotation.yaw),
                               interval_binding_method='Recorded source station on the retained native interval',
                               stop_location_source='Original source stop station transferred with the compiled road',
                               source_station_transfer=deepcopy(record))
                    new_rows.append(row)
            approaches.extend(new_rows)
            changed.append(dict(signal_id=binding['signal_id'], source_lane_id=old['lane_id'],
                                old_road_id=old['road_id'], old_s_m=old['s_m'],
                                current_intervals=[dict(road_id=a['road_id'], lane_id=a['xodr_lane_id'],
                                                        s_m=a['s_m']) for a in new_rows]))
        binding['approaches'] = approaches
    result['native_trigger_approach_count'] = sum(
        a['native_trigger_supported'] for b in result['bindings'] for a in b['approaches'])
    ids = {b['signal_id'] for b in result['bindings']}
    result['signal_reference_count'] = sum(
        e.get('id') in ids for e in root.findall('road/signals/signalReference'))
    result['previous_validation'] = result.get('validation')
    result['validation'] = dict(status='source_station_dependencies_refreshed_pending_loaded_world_check',
                                changed_original_approaches=len(changed))
    report = dict(changed_approaches=changed, source_programs_exact=result['source_programs'] == manifest['source_programs'],
                  identities_and_phases_exact=all(
                      {k: b[k] for k in ('signal_id', 'controller_id', 'junction_id', 'source_signal_id', 'phase_states')} ==
                      {k: old[k] for k in ('signal_id', 'controller_id', 'junction_id', 'source_signal_id', 'phase_states')}
                      for b, old in zip(result['bindings'], manifest['bindings'])),
                  driving_approaches_before=sum(a['native_trigger_supported'] for b in manifest['bindings'] for a in b['approaches']),
                  driving_approaches_after=result['native_trigger_approach_count'],
                  scope='Refresh only recorded transferred stop intervals; source phases and permissions retained. Loaded actor bindings and vehicle behavior remain actual runtime checks.')
    return result, report


def main():
    import argparse
    import json
    from pathlib import Path
    import carla

    parser = argparse.ArgumentParser(description='Refresh runtime signal bindings after recorded native source-station transfer.')
    for name in ('bindings', 'transfer-review', 'xodr', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.bindings.read_text())
    transfer = json.loads(args.transfer_review.read_text())
    native = carla.Map('transferred_signal_binding_review', args.xodr.read_text())
    result, report = refresh_binding_manifest(manifest, transfer, native, ET.parse(args.xodr).getroot())
    result['native_interval_dependencies'] = dict(bindings=str(args.bindings), transfer_review=str(args.transfer_review), xodr=str(args.xodr))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'signal_bindings.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.output / 'binding_refresh_review.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
