"""Match BasicAgent signal responses to source-bound OpenDRIVE routes.

The adapter uses affected lane references, speed-dependent stopping distance,
and emergency braking. The optional yellow_stop setting adds a yellow-phase
stopping decision for background traffic.
"""
import math
import types
from shapely.geometry import LineString, Point


def yellow_stopping_distance(speed_mps, deceleration_mps2, reaction_time_s, buffer_m):
    """Estimate stopping distance from current speed."""
    return speed_mps * reaction_time_s + speed_mps**2 / (2 * deceleration_mps2) + buffer_m


def upstream_signal_waypoint(native_map, waypoint, distance=3.):
    """Match CARLA's trigger shift, constrained to this actual lane section.

    CARLA uses decreasing s on negative lanes and increasing s on positive
    lanes. Native queries find the section boundary without parsing a city-size
    OpenDRIVE string separately for each traffic vehicle.
    """
    direction = -1 if waypoint.lane_id < 0 else 1

    def at(shift):
        station = waypoint.s + direction * shift
        if station < 0:
            return None
        candidate = native_map.get_waypoint_xodr(waypoint.road_id, waypoint.lane_id, station)
        return candidate if candidate is not None and candidate.section_id == waypoint.section_id else None

    target = at(distance)
    if target is not None:
        return target
    low, high = 0., distance
    for _ in range(30):
        middle = (low + high) / 2
        if at(middle) is None:
            high = middle
        else:
            low = middle
    return at(max(0., low - 1e-5))


def install_route_signal_detection(agent, plan, yellow_stop=None, affected_waypoints=None, route_lane_keys=None):
    import carla
    from agents.navigation.basic_agent import TrafficLightDetectionResult

    route = LineString([(wp.transform.location.x, wp.transform.location.y) for wp, _ in plan])
    pairs = {(wp.road_id, wp.section_id, wp.lane_id) for wp, _ in plan}
    # Use the complete source route to recover sections skipped by controller sampling.
    pairs.update(tuple(map(int, key)) for key in (route_lane_keys or []))
    lane_types = {wp.lane_type for wp, _ in plan}
    candidates = []
    for light in agent._lights_list:
        if affected_waypoints is None:
            waypoints=light.get_affected_lane_waypoints()
        else:
            if light.id not in affected_waypoints:
                affected_waypoints[light.id]=light.get_affected_lane_waypoints()
            waypoints=affected_waypoints[light.id]
        for wp in waypoints:
            if wp.lane_type not in lane_types or (wp.road_id, wp.section_id, wp.lane_id) not in pairs:
                continue
            point = Point(wp.transform.location.x, wp.transform.location.y)
            if route.distance(point) > .25:
                continue
            station = route.project(point)
            # TrafficLightComponent::InitializeSign clips the upstream shift
            # to the affected lane's section; preserve its actual lane ID.
            trigger = upstream_signal_waypoint(agent._map, wp)
            trigger_station = route.project(Point(trigger.transform.location.x, trigger.transform.location.y))
            candidates.append({'actor': light, 'signal_id': light.get_opendrive_id(),
                'road_id': wp.road_id, 'lane_id': wp.lane_id, 'section_id': wp.section_id, 'reference_s_m': wp.s, 'lane_type': str(wp.lane_type),
                'reference_route_station_m': station, 'trigger_route_station_m': trigger_station})
    candidates.sort(key=lambda c: c['reference_route_station_m'])
    agent.route_signal_diagnostic = None
    yellow_decisions = {}

    def detect(self, lights_list=None, max_distance=None):
        if self._ignore_traffic_lights:
            return TrafficLightDetectionResult(False, None)
        threshold = max_distance or self._base_tlight_threshold
        position = self._vehicle.get_location()
        station = route.project(Point(position.x, position.y))
        forward_extent = self._vehicle.bounding_box.extent.x
        self.route_signal_diagnostic = None
        # Process signal references in travel order, activating the next after passing the first.
        ahead = [c for c in candidates if c['reference_route_station_m'] >= station + forward_extent]
        active = [c for c in ahead if c['reference_route_station_m'] - ahead[0]['reference_route_station_m'] <= 1e-4] if ahead else []
        # Keep the yellow-phase decision stable through speed changes.
        if yellow_stop is not None:
            for candidate in candidates:
                if candidate['actor'].get_state() != carla.TrafficLightState.Yellow:
                    yellow_decisions.pop(candidate['actor'].id, None)
        # Retain the red latch and accepted yellow stop through the stop phase.
        cached = self._last_traffic_light
        cached_red = cached is not None and cached.get_state() == carla.TrafficLightState.Red
        cached_yellow_stop = (cached is not None and cached.get_state() == carla.TrafficLightState.Yellow
                              and yellow_decisions.get(cached.id, {}).get('stop', False))
        if cached_red or cached_yellow_stop:
            for candidate in active:
                if candidate['actor'].id == cached.id and candidate['reference_route_station_m'] >= station + forward_extent:
                    self.route_signal_diagnostic = {**{k: v for k, v in candidate.items() if k != 'actor'},
                        'latched_red': cached_red, 'latched_yellow_stop': cached_yellow_stop,
                        'yellow_decision': yellow_decisions.get(cached.id), 'trigger_distance_m': candidate['trigger_route_station_m'] - station,
                        'max_distance_m': threshold, 'vehicle_route_station_m': station}
                    return TrafficLightDetectionResult(True, cached)
        for candidate in active:
            light = candidate['actor']
            distance = candidate['trigger_route_station_m'] - station
            # Release references passed by the vehicle front.
            if candidate['reference_route_station_m'] < station + forward_extent:
                continue
            if distance > threshold:
                break
            state = light.get_state()
            diagnostic = {**{k: v for k, v in candidate.items() if k != 'actor'},
                'trigger_distance_m': distance, 'max_distance_m': threshold,
                'vehicle_route_station_m': station}
            self.route_signal_diagnostic = {**diagnostic, 'state': str(state), 'selection': 'nearest_unpassed_route_reference'}
            if state == carla.TrafficLightState.Yellow and yellow_stop is not None:
                if light.id not in yellow_decisions:
                    velocity = self._vehicle.get_velocity()
                    speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                    required = yellow_stopping_distance(speed, **yellow_stop)
                    remaining = candidate['reference_route_station_m'] - station - forward_extent
                    yellow_decisions[light.id] = {'stop': bool(remaining >= required),
                        'speed_mps': speed, 'front_remaining_m': remaining,
                        'required_stopping_distance_m': required, **yellow_stop}
                diagnostic['yellow_decision'] = yellow_decisions[light.id]
                self.route_signal_diagnostic = diagnostic
                if not yellow_decisions[light.id]['stop']:
                    continue
            elif state != carla.TrafficLightState.Red:
                continue
            self._last_traffic_light = light
            self.route_signal_diagnostic = diagnostic
            return TrafficLightDetectionResult(True, light)
        self._last_traffic_light = None
        return TrafficLightDetectionResult(False, None)

    agent._affected_by_traffic_light = types.MethodType(detect, agent)
    return [{k: v for k, v in c.items() if k != 'actor'} for c in candidates]
