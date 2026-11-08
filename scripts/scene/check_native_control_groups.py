"""Check persisted native signal identities, group binding and sustained states."""
import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--map', required=True)
    p.add_argument('--bindings', type=Path, required=True)
    p.add_argument('--conditions', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--hold-frames', type=int, default=40)
    a = p.parse_args()
    if a.hold_frames < 2:
        raise ValueError('Sustained state measurement requires at least two frames')
    import carla
    client = carla.Client('127.0.0.1', a.port)
    client.set_timeout(600)
    world = client.get_world()
    map_name = world.get_map().name.removeprefix('/Game/').removesuffix('.umap')
    if map_name != a.map.removeprefix('/Game/'):
        raise ValueError('Loaded a different map: ' + map_name)
    expected = json.loads(a.bindings.read_text())['bindings']
    conditions = json.loads(a.conditions.read_text())
    lights = list(world.get_actors().filter('traffic.traffic_light*'))
    counts = Counter(x.get_opendrive_id() for x in lights)
    expected_ids = {x['signal_id'] for x in expected}
    missing = sorted(expected_ids - set(counts))
    extra = sorted(set(counts) - expected_ids)
    duplicates = {key: count for key, count in counts.items() if count != 1}
    actors = {x.get_opendrive_id(): x for x in lights}
    errors, groups = [], []
    if missing or extra or duplicates:
        errors.append(dict(kind='native_signal_identity', missing=missing, extra=extra, duplicates=duplicates))
    selected = [x['native_construction']['settings'] for x in conditions['native_objects']
                if x.get('native_construction', {}).get('kind') == 'observed_signal_heads']
    previous = [(actor, actor.get_state(), actor.is_frozen()) for actor in lights]
    frozen_values = {frozen for _, _, frozen in previous}
    if len(frozen_values) != 1:
        raise RuntimeError('CARLA global freeze cannot restore a mixed per-group frozen state')
    original_frozen = next(iter(frozen_values))
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = .05
    world.apply_settings(settings)
    def waypoint(w):
        return dict(road_id=w.road_id, lane_id=w.lane_id, section_id=w.section_id, s=w.s)
    def name(state):
        return str(state).rsplit('.', 1)[-1]
    restoration = {}
    try:
        world.freeze_all_traffic_lights(True)
        world.tick()
        for source in selected:
            ids = sorted(x['signal_id'] for x in source['native_group_members'])
            record = dict(observed_signal_id=source['signal_id'], expected_group_ids=ids,
                          members=[], states=[])
            groups.append(record)
            if any(counts[sid] != 1 for sid in ids):
                errors.append(dict(kind='observed_group_identity', ids=ids))
                continue
            for sid in ids:
                actor = actors[sid]
                actual_ids = sorted(x.get_opendrive_id() for x in actor.get_group_traffic_lights())
                row = dict(signal_id=sid, group_ids=actual_ids,
                           stop_waypoints=[waypoint(w) for w in actor.get_stop_waypoints()],
                           affected_waypoints=[waypoint(w) for w in actor.get_affected_lane_waypoints()])
                record['members'].append(row)
                if actual_ids != ids or not row['stop_waypoints'] or not row['affected_waypoints']:
                    errors.append(dict(kind='observed_group_binding', signal_id=sid))
            for color in ['Red', 'Yellow', 'Green']:
                requested = {sid: color if sid == source['signal_id'] else 'Red' for sid in ids}
                for sid, value in requested.items():
                    actors[sid].set_state(getattr(carla.TrafficLightState, value))
                rows = []
                for _ in range(a.hold_frames):
                    frame = world.tick()
                    observed = {sid: name(actors[sid].get_state()) for sid in ids}
                    frozen = {sid: actors[sid].is_frozen() for sid in ids}
                    rows.append(dict(frame=frame, observed=observed, frozen=frozen,
                                     passed=observed == requested and all(frozen.values())))
                passed = all(row['passed'] for row in rows)
                record['states'].append(dict(requested=requested, frames=rows, passed=passed))
                if not passed:
                    errors.append(dict(kind='sustained_group_state', signal_id=source['signal_id'],
                                       requested=color, failing_frames=sum(not row['passed'] for row in rows)))
    finally:
        for actor, state, _ in previous:
            actor.set_state(state)
        world.freeze_all_traffic_lights(original_frozen)
        world.tick()
        world.tick()
        restoration = dict(requested_frozen=original_frozen,
                           restored_count=sum(actor.is_frozen() == frozen for actor, _, frozen in previous),
                           expected_count=len(previous))
        world.apply_settings(original_settings)
    if restoration['restored_count'] != restoration['expected_count']:
        errors.append(dict(kind='freeze_restoration', **restoration))
    report = dict(map=map_name, bindings=str(a.bindings), conditions=str(a.conditions),
                  expected_signals=len(expected_ids), actual_signals=len(lights), groups=groups,
                  hold_frames_per_state=a.hold_frames, fixed_delta_seconds=.05,
                  freeze_restoration=restoration, errors=errors, passed=not errors,
                  scope='Current-map identity and source group/approach binding; three requested states held across actual synchronous frames with every group member frozen. Prior states/global freezing/world settings restored. Visible emission and vehicle behavior require separate checks.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'groups'}, indent=2))
    if errors:
        raise RuntimeError('Native source control check failed; inspect ' + str(a.output))


if __name__ == '__main__':
    main()
