"""CARLA GlobalRoutePlanner with native lane identities at graph endpoints.

Extends agents/navigation/global_route_planner.py (MIT, CVC). The upstream
search, lane-change links, and turn classifier operate on road/section/lane
keys and edges sampled to their native endpoints.
"""
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.local_planner import RoadOption


def lane_key(waypoint):
    return waypoint.road_id, waypoint.section_id, waypoint.lane_id


class NativeRoutePlanner(GlobalRoutePlanner):
    def _build_topology(self):
        self._topology = []
        self.native_key_positions = {}
        self._native_waypoints = {}
        self.removed_cross_lane_samples = 0
        for entry, exit in self._wmap.get_topology():
            segment = {'entry': entry, 'exit': exit}
            for side in ('entry', 'exit'):
                wp = segment[side]
                key = lane_key(wp)
                segment[side + 'xyz'] = key
                p = wp.transform.location
                self.native_key_positions[key] = (p.x, p.y, p.z)
            # Use native topology endpoints to delimit each sampled road edge.
            samples = entry.next_until_lane_end(self._sampling_resolution)
            path = [wp for wp in samples if lane_key(wp) == lane_key(entry)]
            if lane_key(entry) == lane_key(exit):
                low, high = sorted((entry.s, exit.s))
                path = [wp for wp in path if low < wp.s < high]
            segment['path'] = path
            self.removed_cross_lane_samples += len(samples) - len(path)
            self._topology.append(segment)
            for waypoint in [entry] + path + [exit]:
                key = waypoint.road_id, waypoint.lane_id, waypoint.s
                self._native_waypoints[key] = waypoint

    def waypoint_at(self, road_id, lane_id, station):
        """Resolve a saved native station without moving a lane-end sample.

        CARLA GetWaypointXODR takes float s; rounding a double lane-end
        station upward can return None. Reuse the identical native object
        already sampled by this planner, with exact road/lane/station keys.
        """
        waypoint = self._wmap.get_waypoint_xodr(road_id, lane_id, station)
        if waypoint is None:
            waypoint = self._native_waypoints.get((road_id, lane_id, station))
        return waypoint

    def _build_graph(self):
        super()._build_graph()
        for key, node in self._id_map.items():
            self._graph.nodes[node]['vertex'] = self.native_key_positions[key]

    def _localize(self, location):
        wp = location if hasattr(location, 'road_id') else self._wmap.get_waypoint(location)
        return self._road_id_to_edge[wp.road_id][wp.section_id][wp.lane_id]

    def trace_waypoints(self, origin, destination):
        """Upstream trace with explicit native endpoints kept through search.

        CARLA's nearest-XY query can select another overlapping source branch.
        The caller supplies a source-backed native road/lane/station instead.
        """
        route = self._path_search(origin, destination)
        current_waypoint = origin
        destination_location = destination.transform.location
        trace = []
        self._previous_decision = RoadOption.VOID
        self._intersection_end_node = -1
        for i in range(len(route) - 1):
            option = self._turn_decision(i, route)
            edge = self._graph.edges[route[i], route[i + 1]]
            if edge['type'] not in (RoadOption.LANEFOLLOW, RoadOption.VOID):
                trace.append((current_waypoint, option))
                exit_wp = edge['exit_waypoint']
                n1, n2 = self._localize(exit_wp)
                following = self._graph.edges[n1, n2]
                if following['path']:
                    index = self._find_closest_in_list(current_waypoint, following['path'])
                    current_waypoint = following['path'][min(len(following['path']) - 1, index + 5)]
                else:
                    current_waypoint = following['exit_waypoint']
                trace.append((current_waypoint, option))
            else:
                path = [edge['entry_waypoint']] + edge['path'] + [edge['exit_waypoint']]
                index = self._find_closest_in_list(current_waypoint, path)
                for waypoint in path[index:]:
                    current_waypoint = waypoint
                    trace.append((waypoint, option))
                    if len(route) - i <= 2:
                        if waypoint.transform.location.distance(destination_location) < 2 * self._sampling_resolution:
                            break
                        if lane_key(waypoint) == lane_key(destination):
                            destination_index = self._find_closest_in_list(destination, path)
                            if index > destination_index:
                                break
        return trace
