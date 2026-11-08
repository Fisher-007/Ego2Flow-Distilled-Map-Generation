"""Extract following and signal-response intervals from vehicle trajectory logs.

Vehicle lengths are measured from the spawned actors in meters.
"""

import argparse
import json
import math
from pathlib import Path


def _distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _light(row, light_id):
    return row["lights"].get(str(light_id), "").rsplit(".", 1)[-1]


def _runs(rows, keys):
    """Consecutive recorded frames with the same nonempty observation key."""
    start = None
    for i, key in enumerate(keys):
        same = i > 0 and key == keys[i - 1] and rows[i]["frame"] == rows[i - 1]["frame"] + 1
        if start is not None and not same:
            yield start, i - 1, keys[start]
            start = None
        if key is not None and start is None:
            start = i
    if start is not None:
        yield start, len(rows) - 1, keys[start]


def red_green_events(rows):
    events = []
    names = sorted({name for row in rows for name in row["vehicles"]})
    for name in names:
        keys = []
        for row in rows:
            vehicle = row["vehicles"].get(name)
            light_id = vehicle.get("agent_red_light_id") if vehicle else None
            keys.append(light_id if light_id is not None and _light(row, light_id) == "Red"
                        and vehicle["speed_kmh"] < 0.5 else None)
        for start, end, light_id in _runs(rows, keys):
            duration = rows[end]["sim_time_s"] - rows[start]["sim_time_s"]
            if duration < 1.0 - 1e-9:
                continue
            position = rows[end]["vehicles"][name]["position_enu_m"]
            green_start = None
            for i in range(end + 1, len(rows)):
                row, previous = rows[i], rows[i - 1]
                vehicle = row["vehicles"].get(name)
                if vehicle is None or row["frame"] != previous["frame"] + 1:
                    break
                state = _light(row, light_id)
                # Reset stationary time when the stop interval ends.
                if vehicle.get("agent_red_light_id") not in (None, light_id):
                    break
                if state != "Green":
                    if state != "Yellow" or green_start is not None or vehicle["speed_kmh"] >= 0.5:
                        break
                    continue
                if green_start is None:
                    green_start = i
                displacement = _distance(position, vehicle["position_enu_m"])
                if vehicle["speed_kmh"] > 3.6 and displacement > 1.0:
                    events.append({"vehicle": name, "light_id": light_id,
                                   "red_stop_frames": [rows[start]["frame"], rows[end]["frame"]],
                                   "continuous_red_stop_seconds": duration,
                                   "red_stop_max_speed_kmh": max(r["vehicles"][name]["speed_kmh"] for r in rows[start:end + 1]),
                                   "green_start_frame": rows[green_start]["frame"],
                                   "resumed_frame": row["frame"], "resumed_speed_kmh": vehicle["speed_kmh"],
                                   "displacement_after_stop_m": displacement})
                    break
    return events


def _motion_directions(rows):
    previous, last, directions = {}, {}, []
    for row in rows:
        for name, vehicle in row["vehicles"].items():
            position = vehicle["position_enu_m"]
            if name in previous:
                distance = _distance(position, previous[name])
                if vehicle["speed_kmh"] > 3.6 and distance > 0.01:
                    last[name] = [(position[k] - previous[name][k]) / distance for k in (0, 1)]
            previous[name] = position
        directions.append(dict(last))
    return directions


def following_events(rows, vehicle_lengths_m):
    half_lengths = (vehicle_lengths_m["ego"] + vehicle_lengths_m["leader"]) / 2
    directions, events = _motion_directions(rows), []
    holds = [True if row["leader_forced_stop"] else None for row in rows]
    for first, last, _ in _runs(rows, holds):
        before = [r["vehicles"]["ego"]["speed_kmh"] for r in rows[:first]
                  if r["sim_time_s"] >= rows[first]["sim_time_s"] - 2.0
                  and r["vehicles"]["ego"].get("agent_red_light_id") is None]
        if max(before, default=0) <= 5:
            continue
        keys, gaps, alignments = [None] * len(rows), {}, {}
        for i in range(first, last + 1):
            vehicles = rows[i]["vehicles"]
            if not all(name in vehicles and name in directions[i] for name in ("ego", "leader")):
                continue
            ego, leader = vehicles["ego"], vehicles["leader"]
            e, l = directions[i]["ego"], directions[i]["leader"]
            delta = [leader["position_enu_m"][k] - ego["position_enu_m"][k] for k in (0, 1)]
            alignment = sum(a * b for a, b in zip(e, l))
            gap = sum(a * b for a, b in zip(delta, e)) - half_lengths
            lateral = abs(delta[0] * e[1] - delta[1] * e[0])
            if (alignment > 0.9 and lateral < 1.5 and 0 <= gap <= 15
                    and leader["speed_kmh"] < 0.5 and leader["control"]["brake"] >= 0.5
                    and ego.get("agent_red_light_id") is None):
                keys[i], gaps[i], alignments[i] = True, gap, alignment
        for start, end, _ in _runs(rows, keys):
            duration = rows[end]["sim_time_s"] - rows[start]["sim_time_s"]
            if duration < 1.0 - 1e-9:
                continue
            # Measure the vehicle response across alternating throttle and brake commands.
            braking = [i for i in range(start, end + 1)
                       if rows[i]["vehicles"]["ego"]["control"]["brake"] >= 0.2]
            moving_brakes = [i for i in braking if rows[i]["vehicles"]["ego"]["speed_kmh"] > 5]
            if not moving_brakes:
                continue
            brake_start = moving_brakes[0]
            near_zero = [i for i in range(brake_start, end + 1)
                         if rows[i]["vehicles"]["ego"]["speed_kmh"] < 0.5]
            if not near_zero:
                continue
            response_start = near_zero[0]
            stopped_keys = [True if response_start <= i <= end and row["vehicles"]["ego"]["speed_kmh"] < 0.5 else None
                            for i, row in enumerate(rows)]
            stops = list(_runs(rows, stopped_keys))
            longest = max(stops, key=lambda run: rows[run[1]]["sim_time_s"] - rows[run[0]]["sim_time_s"])
            longest_seconds = rows[longest[1]]["sim_time_s"] - rows[longest[0]]["sim_time_s"]
            steady_stop = longest_seconds >= 1.0 - 1e-9
            if not steady_stop and len(stops) < 2:
                continue
            response = rows[response_start:end + 1]
            ego_positions = [r["vehicles"]["ego"]["position_enu_m"] for r in response]
            brake_keys = [True if brake_start <= i <= end and row["vehicles"]["ego"]["control"]["brake"] >= 0.2 else None
                          for i, row in enumerate(rows)]
            brake_runs = list(_runs(rows, brake_keys))
            stopped = {name: rows[end]["vehicles"][name]["position_enu_m"] for name in ("ego", "leader")}
            for i in range(last + 1, len(rows)):
                row = rows[i]
                if row["sim_time_s"] - rows[last]["sim_time_s"] > 20:
                    break
                if not all(name in row["vehicles"] for name in stopped):
                    break
                ego, leader = row["vehicles"]["ego"], row["vehicles"]["leader"]
                movement = {name: _distance(stopped[name], row["vehicles"][name]["position_enu_m"]) for name in stopped}
                if (ego.get("agent_red_light_id") is None and ego["speed_kmh"] > 5
                        and leader["speed_kmh"] > 3.6 and min(movement.values()) > 1
                        and ego["control"]["throttle"] > 0 and leader["control"]["throttle"] > 0):
                    events.append({"forced_hold_frames": [rows[first]["frame"], rows[last]["frame"]],
                                   "response_type": "stationary_stop" if steady_stop else "stop_and_creep",
                                   "leader_stationary_frames": [rows[start]["frame"], rows[end]["frame"]],
                                   "leader_continuous_stationary_seconds": duration,
                                   "ego_braking_start_frame": rows[brake_start]["frame"],
                                   "ego_braking_start_speed_kmh": rows[brake_start]["vehicles"]["ego"]["speed_kmh"],
                                   "response_frames": [rows[response_start]["frame"], rows[end]["frame"]],
                                   "ego_response_speed_range_kmh": [min(r["vehicles"]["ego"]["speed_kmh"] for r in response), max(r["vehicles"]["ego"]["speed_kmh"] for r in response)],
                                   "ego_response_net_displacement_m": _distance(ego_positions[0], ego_positions[-1]),
                                   "ego_response_traveled_distance_m": sum(_distance(a, b) for a, b in zip(ego_positions, ego_positions[1:])),
                                   "ego_near_zero_intervals": len(stops),
                                   "ego_longest_continuous_near_zero_frames": [rows[longest[0]]["frame"], rows[longest[1]]["frame"]],
                                   "ego_longest_continuous_near_zero_seconds": longest_seconds,
                                   "ego_braking_frame_count": sum(i >= brake_start for i in braking),
                                   "ego_braking_interval_count": len(brake_runs),
                                   "ego_braking_interval_examples": [[rows[a]["frame"], rows[b]["frame"]] for a, b, _ in brake_runs[:5]],
                                   "ego_pre_hold_max_kmh": max(before),
                                   "net_gap_range_m": [min(gaps[j] for j in range(response_start, end + 1)), max(gaps[j] for j in range(response_start, end + 1))],
                                   "minimum_heading_alignment": min(alignments[j] for j in range(response_start, end + 1)),
                                   "resumed_frame": row["frame"], "ego_resumed_speed_kmh": ego["speed_kmh"],
                                   "leader_resumed_speed_kmh": leader["speed_kmh"], "displacement_after_stop_m": movement})
                    break
            if events and events[-1]["forced_hold_frames"][0] == rows[first]["frame"]:
                break
    return events


def far_motion_intervals(rows):
    intervals = []
    names = sorted({name for row in rows for name in row["vehicles"]} - {"ego"})
    for name in names:
        keys, distances = [None] * len(rows), {}
        for i in range(1, len(rows)):
            a, b = rows[i - 1]["vehicles"].get(name), rows[i]["vehicles"].get(name)
            if a is None or b is None:
                continue
            distance = _distance(a["position_enu_m"], b["position_enu_m"])
            if (rows[i]["frame"] == rows[i - 1]["frame"] + 1
                    and min(a["distance_from_ego_m"], b["distance_from_ego_m"]) > 50 and distance > 0.01):
                keys[i], distances[i] = True, distance
        for first, last, _ in _runs(rows, keys):
            start = first - 1
            duration = rows[last]["sim_time_s"] - rows[start]["sim_time_s"]
            displacement = _distance(rows[start]["vehicles"][name]["position_enu_m"], rows[last]["vehicles"][name]["position_enu_m"])
            if duration >= 1.0 - 1e-9 and displacement > 1.0:
                intervals.append({"vehicle": name, "frames": [rows[start]["frame"], rows[last]["frame"]],
                                  "seconds": duration, "consecutive_motion_steps": last - first + 1,
                                  "net_displacement_m": displacement, "traveled_distance_m": sum(distances[i] for i in range(first, last + 1)),
                                  "minimum_distance_from_ego_m": min(r["vehicles"][name]["distance_from_ego_m"] for r in rows[start:last + 1])})
    return intervals


def evaluate_traffic_evidence(rows, vehicle_lengths_m):
    red_green = red_green_events(rows)
    following = following_events(rows, vehicle_lengths_m)
    far_motion = far_motion_intervals(rows)
    return {"checks": {"red_stop_then_green_motion_observed": bool(red_green),
                       "leader_stop_response": bool(following), "vehicles_beyond_50m_update": bool(far_motion)},
            "vehicle_lengths_m": vehicle_lengths_m, "red_green_events": red_green,
            "following_events": following, "far_motion_intervals": far_motion,
            "scope": "Observed continuous trajectory intervals; not formal signal compliance or a controlled causal attribution."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--vehicle-lengths", type=Path, required=True, help="JSON object mapping role names to measured full vehicle lengths in meters")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.trajectory.read_text().splitlines() if line.strip()]
    result = evaluate_traffic_evidence(rows, json.loads(args.vehicle_lengths.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
