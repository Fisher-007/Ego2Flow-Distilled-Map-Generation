"""Build OpenDRIVE roads and physical meshes from the fused SceneIR graph.

scenariogeneration writes geometry and profiles; SciPy provides interpolation
and arc-length operations. Junction movements follow fitted source intervals.
"""

import json
import math
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
from numpy.polynomial import Polynomial
from scipy.integrate import quad
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator, PPoly
from scipy.optimize import brentq
from shapely.geometry import LineString, Point, Polygon
from shapely.strtree import STRtree
from scenariogeneration import xodr


def stable_elevation_rows(rows, end, tolerance=1e-5):
    """Avoid CARLA cancellation when local cubics are expanded at large road s.

    Well-conditioned cubics remain exact. Affected intervals retain endpoints
    and use linear chords with analytically bounded height error.
    """
    rows=list(rows);origin_shift=0.
    if rows and 0<rows[0][0]<1e-9:
        start,a,b,c,d=map(float,rows[0]);delta=-start
        rows[0]=(0.,a+delta*(b+delta*(c+delta*d)),b+delta*(2*c+3*d*delta),c+3*d*delta,d)
        origin_shift=start
    output=[];replaced=0;maximum_error=0.
    for i,row in enumerate(rows):
        start,a,b,c,d=map(float,row);finish=rows[i+1][0] if i+1<len(rows) else end
        scale=abs(a)+abs(b*start)+abs(c*start**2)+abs(d*start**3)
        if 32*np.finfo(float).eps*scale<=tolerance:
            output.append(tuple(row));continue
        replaced+=1
        def emit(low,high):
            nonlocal maximum_error
            h=high-low
            aa=a+low*(b+low*(c+low*d));bb=b+low*(2*c+3*low*d);cc=c+3*d*low
            slope=bb+h*(cc+h*d)
            difference=Polynomial([0.,bb-slope,cc,d])
            candidates=[0.,h]+[float(r.real) for r in difference.deriv().roots()
                                if abs(r.imag)<1e-12 and 0<r.real<h]
            error=max(abs(float(difference(t))) for t in candidates)
            if error>tolerance:
                mid=(low+high)/2;emit(low,mid);emit(mid,high)
            else:
                output.append((start+low,aa,slope,0.,0.));maximum_error=max(maximum_error,error)
        emit(0.,finish-start)
    return output,dict(replaced_cubic_intervals=replaced,result_intervals=len(output),
                       maximum_approximation_error_m=maximum_error,tolerance_m=tolerance,origin_station_shift_m=origin_shift)


@dataclass
class Curve:
    coefficients: np.ndarray  # global xy, ascending powers of normalized p

    def __post_init__(self):
        self.x = Polynomial(self.coefficients[:, 0])
        self.y = Polynomial(self.coefficients[:, 1])
        self.dx, self.dy = self.x.deriv(), self.y.deriv()
        self.straight_speed = (float(np.linalg.norm(self.coefficients[1]))
                               if not np.any(self.coefficients[2:]) else None)
        self.length = self.arc(1.0)

    def arc(self, p):
        if self.straight_speed is not None:return p*self.straight_speed
        c1, c2, c3 = self.coefficients[1], 2*self.coefficients[2], 3*self.coefficients[3]
        ax, ay, bx, by, cx, cy = *c1, *c2, *c3
        return quad(lambda t: math.hypot(ax+t*(bx+t*cx), ay+t*(by+t*cy)),
                    0, p, epsabs=1e-9)[0]

    def parameter(self, s):
        if s <= 1e-10:
            return 0.0
        if s >= self.length - 1e-10:
            return 1.0
        if self.straight_speed is not None:return s/self.straight_speed
        return brentq(lambda p: self.arc(p) - s, 0, 1, xtol=1e-12)

    def xy(self, p):
        return np.array([self.x(p), self.y(p)])

    def tangent(self, p):
        vector = np.array([self.dx(p), self.dy(p)])
        return vector / np.linalg.norm(vector)

    def clipped(self, start, end):
        p0, p1 = self.parameter(start), self.parameter(end)
        c0,c1,c2,c3=self.coefficients
        delta=p1-p0
        xy=c0+p0*(c1+p0*(c2+p0*c3))
        first=(c1+p0*(2*c2+3*p0*c3))*delta
        heading=math.atan2(first[1],first[0])
        local=np.array([np.zeros(2),first,(c2+3*p0*c3)*delta**2,c3*delta**3])
        rotation=np.array([[math.cos(heading),-math.sin(heading)],
                           [math.sin(heading),math.cos(heading)]])
        coefficients=local@rotation
        geometry=xodr.ParamPoly3(*coefficients[:,0],*coefficients[:,1],prange="normalized")
        return geometry, xy, heading


@dataclass
class LaneGeometry:
    curves: list
    elevations: object
    widths: tuple

    def __post_init__(self):
        self.knots = np.r_[0.0, np.cumsum([curve.length for curve in self.curves])]
        self.length = float(self.knots[-1])

    def xy(self, s):
        i = min(len(self.curves) - 1, max(0, int(np.searchsorted(self.knots, s, side="right") - 1)))
        curve = self.curves[i]
        return curve.xy(curve.parameter(s - self.knots[i]))

    def xyz(self, s):
        return np.r_[self.xy(s), float(self.elevations(s))]

    def width(self, s):
        return self.widths[0] + (self.widths[1] - self.widths[0]) * s / self.length

    def add_plan(self, plan, start, end):
        def add_part(curve, lo, hi):
            p0, p1 = curve.parameter(lo), curve.parameter(hi)
            parameters = np.linspace(p0, p1, 6)
            points = np.column_stack([curve.x(parameters), curve.y(parameters)])
            middle = (parameters[:-1] + parameters[1:]) / 2
            exact = np.column_stack([curve.x(middle), curve.y(middle)])
            interpolation_error = np.max(np.linalg.norm(exact - (points[:-1] + points[1:]) / 2, axis=1))
            length_error = hi - lo - np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
            # CARLA samples each paramPoly3 with at least five straight chords.
            # Split the same polynomial where that approximation loses precision.
            if max(interpolation_error, length_error) > 0.0015 and hi - lo > 0.01:
                mid = (lo + hi) / 2
                add_part(curve, lo, mid)
                add_part(curve, mid, hi)
            else:
                geometry, xy, heading = curve.clipped(lo, hi)
                plan.add_fixed_geometry(geometry, *xy, heading)
        for i, curve in enumerate(self.curves):
            lo, hi = max(start, self.knots[i]), min(end, self.knots[i + 1])
            if hi > lo + 1e-10:
                add_part(curve, lo - self.knots[i], hi - self.knots[i])

    def elevation_rows(self, start, end, offset):
        rows = []
        for i in range(len(self.elevations.x) - 1):
            lo, hi = max(start, self.elevations.x[i]), min(end, self.elevations.x[i + 1])
            if hi > lo + 1e-10:
                poly = Polynomial(self.elevations.c[:, i][::-1])
                clipped = poly(Polynomial([lo - self.elevations.x[i], 1]))
                coefficients = np.pad(clipped.coef, (0, 4 - len(clipped.coef)))
                rows.append((offset + lo - start, *coefficients))
        return rows


def source_fit_stations(chord, fit_step):
    """Keep every source knot; omit numerical duplicates among added grid knots."""
    extra = np.arange(0, chord[-1], fit_step)
    right = np.searchsorted(chord, extra)
    distance = np.minimum(np.abs(extra-chord[np.minimum(right,len(chord)-1)]),
                          np.abs(extra-chord[np.maximum(right-1,0)]))
    return np.sort(np.r_[chord, extra[distance > 1e-7]])


def fit_lane(lane, fit_step):
    original = np.asarray(lane["centerline"], dtype=float)
    keep = np.r_[True, np.linalg.norm(np.diff(original[:, :2], axis=0), axis=1) > 1e-7]
    points = original[keep]
    if len(points) < 2 or not np.all(np.isfinite(points)):
        raise ValueError(f"Lane {lane['id']} lacks a finite, nonzero centreline")
    chord = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1))]
    # Include every source knot. Extra chord samples keep corner smoothing local.
    samples = source_fit_stations(chord, fit_step)
    xyz = np.column_stack([np.interp(samples, chord, points[:, i]) for i in range(3)])
    if "boundary_shape" in lane or "parametric_xy_segments" in lane:
        from map_generation.boundary_shape import BoundaryShape
        if 'parametric_xy_segments' in lane:
            curves=[Curve(np.asarray(c)) for c in lane['parametric_xy_segments']]
        else:
            shape = BoundaryShape.from_dict(lane["boundary_shape"])
            curves = [Curve(c) for c in shape.polynomial_segments()]
        arc = np.r_[0., np.cumsum([c.length for c in curves])]
        # Map source heights by normalized arc length; widths use the endpoint policy.
        height_stations = chord/chord[-1]*arc[-1]
        if 'parametric_elevation_rows' in lane:
            rows=np.asarray(lane['parametric_elevation_rows'],dtype=float)
            elevation=PPoly(rows[:,[4,3,2,1]].T,np.r_[rows[:,0],arc[-1]])
        else:
            elevation = PchipInterpolator(height_stations, points[:, 2])
        geometry = LaneGeometry(curves, elevation, tuple(lane.get("end_widths_m", [lane["width_m"]]*2)))
        discrepancy = max(LineString(points[:, :2]).distance(Point(c.xy(p)))
                          for c in curves for p in (.25, .5, .75))
        midpoints = (height_stations[:-1]+height_stations[1:])/2
        height_error = float(np.max(np.abs(elevation(midpoints)-np.interp(midpoints,height_stations,points[:,2]))))
        return geometry, {"source_points": len(original), "fit_segments": len(curves),
            "length_m": geometry.length, "removed_duplicate_xy_points": int((~keep).sum()),
            "max_xy_deviation_from_source_polyline_m": discrepancy,
            "max_height_deviation_from_source_polyline_m": height_error,
            "method": "Exact supplied XY and elevation polynomial segments." if "parametric_elevation_rows" in lane else "Exact fixed-boundary cubic segments; source height profile by normalized arc length.",
            "height_comparison_scope": "PCHIP versus linear height profile at input interval midpoints; not terrain or CARLA fidelity."}
    spline = PchipInterpolator(samples, xyz[:, :2], axis=0)
    curves = [Curve(spline.c[:, i][::-1] * np.diff(samples)[i] ** np.arange(4)[:, None])
              for i in range(len(samples) - 1)]
    arc = np.r_[0.0, np.cumsum([c.length for c in curves])]
    elevation = PchipInterpolator(arc, xyz[:, 2])
    widths = tuple(lane.get("end_widths_m", [lane["width_m"], lane["width_m"]]))
    geometry = LaneGeometry(curves, elevation, widths)
    source_line = LineString(points[:, :2])
    xy_error, z_error = [], []
    for i, curve in enumerate(curves):
        for p in (0.25, 0.5, 0.75):
            xy_error.append(source_line.distance(Point(curve.xy(p))))
            z_error.append(abs(float(elevation(arc[i] + curve.arc(p))) - float(xyz[i, 2] * (1 - p) + xyz[i + 1, 2] * p)))
    return geometry, {"source_points": len(original), "fit_segments": len(curves),
                      "length_m": geometry.length, "removed_duplicate_xy_points": int((~keep).sum()),
                      "max_xy_deviation_from_source_polyline_m": max(xy_error),
                      "max_height_deviation_from_source_polyline_m": max(z_error)}


def chord_station_to_arc(lane, geometry, station, fit_step=1.0):
    """Locate a source-polyline cut on the exact curve used by the compiler."""
    points=np.asarray(lane['centerline'],dtype=float)
    points=points[np.r_[True,np.linalg.norm(np.diff(points[:,:2],axis=0),axis=1)>1e-7]]
    chord=np.r_[0.,np.linalg.norm(np.diff(points[:,:2],axis=0),axis=1).cumsum()]
    if 'boundary_shape' in lane or 'parametric_xy_segments' in lane:
        return float(station/chord[-1]*geometry.length)
    samples=source_fit_stations(chord, fit_step)
    i=min(len(geometry.curves)-1,max(0,int(np.searchsorted(samples,station,side='right')-1)))
    parameter=float(np.clip((station-samples[i])/(samples[i+1]-samples[i]),0.,1.))
    return float(geometry.knots[i]+geometry.curves[i].arc(parameter))


def arc_station_to_chord(lane, geometry, station, fit_step=1.0):
    """Inverse source station mapping for compiler-only subdivisions."""
    points=np.asarray(lane['centerline'],dtype=float)
    points=points[np.r_[True,np.linalg.norm(np.diff(points[:,:2],axis=0),axis=1)>1e-7]]
    chord=np.r_[0.,np.linalg.norm(np.diff(points[:,:2],axis=0),axis=1).cumsum()]
    if 'boundary_shape' in lane or 'parametric_xy_segments' in lane:
        return float(station/geometry.length*chord[-1])
    samples=source_fit_stations(chord, fit_step)
    i=min(len(geometry.curves)-1,max(0,int(np.searchsorted(geometry.knots,station,side='right')-1)))
    parameter=geometry.curves[i].parameter(station-geometry.knots[i])
    return float(samples[i]+parameter*(samples[i+1]-samples[i]))


def bridge_geometry(a, b):
    start, end = a.xyz(a.length), b.xyz(0)
    distance = float(np.linalg.norm(end[:2] - start[:2]))
    if distance < 1e-5:
        return None
    spline = CubicHermiteSpline([0, 1], np.vstack([start[:2], end[:2]]),
                               np.vstack([a.curves[-1].tangent(1), b.curves[0].tangent(0)]) * distance)
    curve = Curve(spline.c[:, 0][::-1])
    elevations = PchipInterpolator([0, curve.length], [start[2], end[2]])
    return LaneGeometry([curve], elevations, (a.width(a.length), b.width(0)))


def lane_type(classes):
    if "passenger" in classes:
        return xodr.LaneType.driving
    if set(classes) <= {"bicycle", "pedestrian"} and "bicycle" in classes:
        return xodr.LaneType.biking
    if set(classes) == {"pedestrian"}:
        return xodr.LaneType.sidewalk
    return xodr.LaneType.restricted


def physical_junction_groups(scene, lanes, edges, geometries):
    """Group real internal lanes plus their shared graph ports, then enumerate movements."""
    polygons = {j["id"]: Polygon(np.asarray(j["boundary"])[:, :2])
                for j in scene["coarse_map"]["junctions"] if len(j["boundary"]) >= 3}
    polygon_ids = list(polygons)
    polygon_index = STRtree(list(polygons.values()))
    outgoing_edges = {}
    for a, b in edges:
        outgoing_edges.setdefault(a, []).append((a, b))
    sumo_members = {lane_id: j["id"] for j in scene["coarse_map"]["junctions"] for lane_id in j["internal_lane_ids"]}
    ports = nx.Graph()
    for a, b in edges:
        ports.add_edge((a, "end"), (b, "start"))
    assignments = {}
    # Retain source-junction coverage cuts as declared open road ends.
    open_boundaries = scene["fusion_map"].get("open_internal_boundaries", {})
    predecessors = {key: [] for key in lanes}
    for a, b in edges:
        predecessors[b].append(a)
    for key, evidence in open_boundaries.items():
        lane = lanes[key]
        if not lane.get("internal"):
            raise ValueError("Open junction boundary must be an internal source lane: " + key)
        if lane["source"] == "av2":
            source = next(row for row in scene["lanes"] if row["id"] == key)
            side = evidence.get("open_side")
            observed_field = {"start": "predecessors", "end": "successors"}.get(side)
            observations = source.get("source_observations", [])
            if (evidence.get("kind") != "observed_source_terminal" or observed_field is None
                    or not observations or any(row["observed_links"][observed_field] for row in observations)
                    or (predecessors[key] if side == "start" else lane["successors"])):
                raise ValueError("AV2 open boundary lacks an observed source terminal: " + key)
        if predecessors[key] and lane["successors"]:
            raise ValueError("Reviewed open boundary now has both graph neighbours: " + key)
    for key, lane in lanes.items():
        if key in open_boundaries or not (lane.get("internal") or lane.get("is_intersection")):
            continue
        ports.add_edge((key, "start"), (key, "end"))
        source_node = sumo_members.get(lane.get("source_lane_id"))
        if source_node is None and lane["source"] == "av2":
            line = LineString(np.asarray(lane["centerline"])[:, :2])
            candidates = [polygon_ids[i] for i in polygon_index.query(line.buffer(3))]
            nearby = [(line.intersection(polygons[jid].buffer(2)).length, -line.distance(polygons[jid]), jid)
                      for jid in candidates if line.distance(polygons[jid]) < 3]
            if nearby:
                source_node = max(nearby)[2]
        if source_node is not None:
            assignments[key] = source_node
            ports.add_edge((key, "start"), ("@sumo:" + source_node, "anchor"))
    # AV2 internal lanes without a SUMO anchor may still cross inside one intersection.
    unanchored = [key for key, lane in lanes.items() if lane.get("is_intersection") and key not in assignments and key not in open_boundaries]
    for i, a in enumerate(unanchored):
        for b in unanchored[i + 1:]:
            if LineString(np.asarray(lanes[a]["centerline"])[:, :2]).distance(LineString(np.asarray(lanes[b]["centerline"])[:, :2])) < 1:
                ports.add_edge((a, "start"), (b, "start"))
    # Use source subcurves and aligned ports for accesses that split parallel lanes.
    for binding in scene.get('native_compilation', {}).get('junction_port_groups', []):
        first, second = map(tuple, binding['ports'])
        if first not in ports or second not in ports:
            raise ValueError('Compiled junction port lacks a source graph endpoint')
        ports.add_edge(first, second)
    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    groups = []
    for component in sorted([sorted(c) for c in nx.connected_components(ports)]):
        member_ports = {p for p in component if p[0] in lanes}
        member_lanes = {key for key, _ in member_ports}
        body = sorted(key for key in member_lanes if (key, "start") in member_ports and (key, "end") in member_ports)
        selected_edges = sorted(edge for key, end in member_ports if end == "end"
                                for edge in outgoing_edges.get(key, []))
        has_gap = any(np.linalg.norm(geometries[a].xyz(geometries[a].length) - geometries[b].xyz(0)) > 0.02 for a, b in selected_edges)
        if not body and len(member_ports) == 2 and not has_gap:
            continue
        incoming = sorted({a for a, b in selected_edges if a not in body})
        outgoing = sorted({b for a, b in selected_edges if b not in body})
        subgraph = graph.subgraph(set(body) | set(incoming) | set(outgoing)).copy()
        # A movement stops at its first external outgoing lane.
        subgraph.remove_edges_from([(a, b) for a, b in subgraph.edges if a in outgoing or b in incoming])
        paths = [path for a in incoming for b in outgoing for path in nx.all_simple_paths(subgraph, a, b)]
        covered = {key for path in paths for key in path[1:-1]}
        groups.append({"source_lane_ids": body, "incoming_lane_ids": incoming, "outgoing_lane_ids": outgoing,
                       "source_sumo_junction_ids": sorted({assignments[k] for k in body if k in assignments}),
                       "paths": sorted(paths), "uncovered_internal_lane_ids": sorted(set(body) - covered),
                       "absorbed_unmarked_lane_ids": [key for key in body if not (lanes[key].get("internal") or lanes[key].get("is_intersection"))]})
    return groups


def resolve_source_route(route_ids, lanes, source_paths=None):
    """Expand retained source-parent references after an exact lane subdivision."""
    children = {}
    for key, lane in lanes.items():
        parent = lane.get('source_parent_lane_id')
        if parent is not None:
            children.setdefault(parent, {})[key]=lane['source_parent_interval_m'][0]
        conflict_parent=lane.get('source_conflict_parent_lane_id')
        if conflict_parent is not None:
            children.setdefault(conflict_parent, {})[key]=lane['source_conflict_parent_interval_m'][0]
        native_parent=lane.get('native_source_interval')
        if native_parent is not None:
            children.setdefault(native_parent['source_lane_id'], {})[key]=native_parent['start_m']
    requested = set(route_ids)
    supplied = []
    ownership = []
    for key in route_ids:
        path = (source_paths or {}).get(key, [key])
        if set(path) - {key} & requested:
            path = [key]
        if path != [key]:
            ownership.append(dict(source_lane=key, child_lane_ids=path, kind='reviewed_source_ownership'))
        supplied.extend(path)
    requested = set(supplied)
    resolved, expansions = [], ownership
    for key in supplied:
        positions = children.get(key, {})
        parts = list(positions)
        # An explicitly supplied sequence of child intervals is already a route.
        if len(parts) > 1 and not (set(parts) - {key}) & requested:
            parts = sorted(parts, key=positions.get)
            if any(b not in lanes[a]['successors'] for a, b in zip(parts, parts[1:])):
                raise ValueError('Source route subdivision is disconnected: ' + key)
            expansions.append(dict(source_lane=key, child_lane_ids=parts))
            resolved.extend(parts)
        else:
            resolved.append(key)
    return resolved, expansions


def native_source_route_checks(route_ids, lanes, mapping, native_edges, source_paths=None):
    resolved, expansions = resolve_source_route(route_ids, lanes, source_paths)
    ids = mapping['key_to_road_id']
    movements = {}
    for rid, metadata in mapping['roads'].items():
        if metadata['kind'] == 'movement':
            path = metadata['source_lane_path']
            for pair in zip(path, path[1:]):
                movements.setdefault(pair, []).append(int(rid))
    links = []
    for a, b in zip(resolved, resolved[1:]):
        copies = movements.get((a, b), [])
        if copies:
            ok = all((ids[mapping['roads'][str(rid)]['from_lane_id']], rid) in native_edges
                     and (rid, ids[mapping['roads'][str(rid)]['to_lane_id']]) in native_edges
                     for rid in copies)
        else:
            ok = a in ids and b in ids and (ids[a], ids[b]) in native_edges
        links.append(dict(from_lane=a, to_lane=b, carla_successor_reachable=ok,
            movement_road_ids=copies,
            passenger_allowed=all('passenger' in lanes[k]['allowed_classes'] for k in (a,b))))
        links[-1]['from'] = links[-1].pop('from_lane')
        links[-1]['to'] = links[-1].pop('to_lane')
    return links, dict(requested_lane_ids=route_ids, resolved_lane_ids=resolved,
                       source_parent_expansions=expansions)


def make_road(rid, pieces, geometries, classes, speed, junction=-1):
    plan, lanes = xodr.PlanView(), xodr.Lanes()
    lane, offset, elevation_rows = None, 0.0, []
    for key, start, end in pieces:
        geometry = geometries[key]
        slope = (geometry.widths[1] - geometry.widths[0]) / geometry.length
        width = geometry.width(start)
        if lane is None:
            lane = xodr.Lane(lane_type(classes), a=width, b=slope)
            section = xodr.LaneSection(0, xodr.Lane())
            section.add_right_lane(lane)
            lanes.add_lanesection(section)
        else:
            lane.add_lane_width(a=width, b=slope, soffset=offset)
        lanes.add_laneoffset(xodr.LaneOffset(offset, width / 2, slope / 2))
        geometry.add_plan(plan, start, end)
        elevation_rows.extend(geometry.elevation_rows(start, end, offset))
        offset += end - start
    road = xodr.Road(rid, plan, lanes, road_type=junction)
    road.add_type(xodr.RoadType.town, speed=float(speed), speed_unit="m/s")
    elevation_rows,_=stable_elevation_rows(elevation_rows,offset)
    for row in elevation_rows:
        road.add_elevation(*row)
    return road


def _fit_job(job):
    key, lane, fit_step = job
    geometry, report = fit_lane(lane, fit_step)
    return key, geometry, report


def build_network(scene, output, trim=0.5, fit_step=1.0, workers=1, reuse_xodr=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    lanes = {lane["id"]: lane for lane in scene["fusion_map"]["lanes"]}
    geometries, fits = {}, {}
    jobs = ((key, lanes[key], fit_step) for key in sorted(lanes))
    if workers == 1:
        for key, geometry, report in map(_fit_job, jobs):
            geometries[key], fits[key] = geometry, report
    else:
        # Workers inherit the launcher's taskset CPU allocation.
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for count, (key, geometry, report) in enumerate(pool.map(_fit_job, jobs, chunksize=16), 1):
                geometries[key], fits[key] = geometry, report
                if count % 5000 == 0:
                    print({"stage":"fit_roads", "completed":count, "total":len(lanes)}, flush=True)
    edges = sorted({(key, successor) for key, lane in lanes.items() for successor in lane["successors"] if successor in lanes})
    external = sorted({(key, successor) for key, lane in lanes.items() for successor in lane["successors"] if successor not in lanes})
    if reuse_xodr:
        mapping = json.loads((output / "road_lane_ids.json").read_text())
        if set(mapping["lane_to_road_intervals"]) != set(lanes):
            raise ValueError("Existing compilation has a different set of source lanes")
        ids = mapping["key_to_road_id"]
        intervals, expected, covered, bridges = {}, {}, set(), {}
        roads = {int(k): v for k, v in mapping["roads"].items()}
        expected = {rid: [] for rid in roads}
        for rid, row in roads.items():
            if row["kind"] == "lane":
                intervals[rid] = [(row["scene_lane_id"], *row["source_interval_m"])]
                continue
            intervals[rid] = row["source_intervals"]
            a, b = row["from_lane_id"], row["to_lane_id"]
            expected[ids[a]].append(rid)
            expected[rid] = [ids[b]]
            for source, target in zip(row["source_lane_path"], row["source_lane_path"][1:]):
                covered.add((source, target))
                key = f"bridge:{source}->{target}"
                if key in bridges:
                    continue
                bridge = bridge_geometry(geometries[source], geometries[target])
                bridges[key] = None
                if bridge is not None:
                    geometries[key] = bridge
                    bridges[key] = dict(from_lane=source, to_lane=target,
                        source_gap_3d_m=float(np.linalg.norm(geometries[source].xyz(geometries[source].length)-geometries[target].xyz(0))),
                        bridge_length_m=bridge.length)
        direct = sorted(set(edges)-covered)
        for a, b in direct:
            expected[ids[a]].append(ids[b])
        covered.update(direct)
        return dict(path=output/"roads.xodr", roads=roads, intervals=intervals,
            road_lengths={rid: sum(end-start for _, start, end in pieces) for rid, pieces in intervals.items()},
            geometries=geometries, expected=expected, mapping=mapping, fits=fits,
            bridges=[v for v in bridges.values() if v], external_successors=external,
            junction_count=len(mapping["physical_junctions"]), source_edges=edges,
            lanes=lanes, covered_source_edges=covered, direct_source_edges=direct)
    groups = physical_junction_groups(scene, lanes, edges, geometries)
    if any(group["uncovered_internal_lane_ids"] for group in groups):
        (output / "junction_review.json").write_text(json.dumps({
            "uncovered_groups": [g for g in groups if g["uncovered_internal_lane_ids"]],
            "source_scene": scene["map_id"],
            "scope": "Observed compilation failure; inspect source ports before changing junction ownership."
        }, indent=2) + "\n")
        raise ValueError("Physical junction has internal lanes without an entry-to-exit path")
    bodies = {key for group in groups for key in group["source_lane_ids"]}
    cuts = {key: [0.0, geometry.length] for key, geometry in geometries.items() if key not in bodies}
    junction_for_end, movement_groups = {}, {}
    for index, group in enumerate(groups):
        jid = 1000000000 + index
        group["id"] = jid
        for endpoint, keys in (("end", group["incoming_lane_ids"]), ("start", group["outgoing_lane_ids"])):
            for key in keys:
                amount = min(trim, geometries[key].length * 0.2)
                cuts[key][0 if endpoint == "start" else 1] += amount if endpoint == "start" else -amount
                junction_for_end[key, endpoint] = jid
        for path in group["paths"]:
            movement_groups["movement:" + ">".join(path)] = (jid, path)
    keys = sorted(cuts) + sorted(movement_groups)
    mapping_path = output / "road_lane_ids.json"
    previous = json.loads(mapping_path.read_text()) if mapping_path.exists() else {}
    allocated = dict(previous.get("allocated_road_ids", previous.get("key_to_road_id", {})))
    next_id = max(allocated.values(), default=0) + 1
    for key in keys:
        if key not in allocated:
            allocated[key] = next_id
            next_id += 1
    ids = {key: allocated[key] for key in keys}
    roads, intervals, metadata, expected = {}, {}, {}, {}
    for count, key in enumerate(sorted(cuts),1):
        rid = ids[key]
        intervals[rid] = [(key, *cuts[key])]
        roads[rid] = make_road(rid, intervals[rid], geometries, lanes[key]["allowed_classes"], lanes[key]["speed_mps"])
        metadata[str(rid)] = {"kind": "lane", "scene_lane_id": key, "source": lanes[key]["source"],
                              "source_interval_m": cuts[key], "allowed_classes": lanes[key]["allowed_classes"],
                              "source_internal": bool(lanes[key].get("internal", False)),
                              "source_is_intersection": bool(lanes[key].get("is_intersection", False)),
                              "open_boundary_evidence": scene["fusion_map"].get("open_internal_boundaries", {}).get(key),
                              "speed_mps": lanes[key]["speed_mps"], "junction_id": None}
        expected[rid] = []
        if count % 5000 == 0:
            print({"stage":"write_external_roads","completed":count,"total":len(cuts)},flush=True)
    network = xodr.OpenDrive(str(scene["map_id"]) + "_roads")
    junctions = {g["id"]: xodr.Junction(f"physical_{g['id']}", g["id"]) for g in groups}
    for (key, endpoint), jid in sorted(junction_for_end.items()):
        getattr(roads[ids[key]], "add_successor" if endpoint == "end" else "add_predecessor")(xodr.ElementType.junction, jid)
    covered_edges, bridges, direct_edges = set(), {}, []
    for count, (key, (jid, path)) in enumerate(sorted(movement_groups.items()),1):
        rid, a, b = ids[key], path[0], path[-1]
        pieces = []
        for index, source in enumerate(path):
            start = cuts[source][1] if index == 0 else 0.0
            end = cuts[source][0] if index == len(path) - 1 else geometries[source].length
            pieces.append((source, start, end))
            if index == len(path) - 1:
                continue
            target = path[index + 1]
            covered_edges.add((source, target))
            bridge_key = f"bridge:{source}->{target}"
            if bridge_key not in bridges:
                bridge = bridge_geometry(geometries[source], geometries[target])
                if bridge is not None:
                    geometries[bridge_key] = bridge
                    gap = float(np.linalg.norm(geometries[source].xyz(geometries[source].length) - geometries[target].xyz(0)))
                    bridges[bridge_key] = {"from_lane": source, "to_lane": target, "source_gap_3d_m": gap, "bridge_length_m": bridge.length}
                else:
                    bridges[bridge_key] = None
            if bridges[bridge_key] is not None:
                pieces.append((bridge_key, 0.0, geometries[bridge_key].length))
        classes = sorted(set.intersection(*(set(lanes[source]["allowed_classes"]) for source in path)))
        speed = min(lanes[source]["speed_mps"] for source in path)
        road = make_road(rid, pieces, geometries, classes, speed, jid)
        road.add_predecessor(xodr.ElementType.road, ids[a], xodr.ContactPoint.end)
        road.add_successor(xodr.ElementType.road, ids[b], xodr.ContactPoint.start)
        road.lanes.lanesections[0].rightlanes[0].add_link("predecessor", -1).add_link("successor", -1)
        junctions[jid].add_connection(xodr.Connection(ids[a], rid, xodr.ContactPoint.start).add_lanelink(-1, -1))
        roads[rid], intervals[rid] = road, pieces
        metadata[str(rid)] = {"kind": "movement", "from_lane_id": a, "to_lane_id": b,
                              "source_lane_path": path, "source_intervals": pieces, "allowed_classes": classes,
                              "speed_mps": speed, "junction_id": jid}
        expected[ids[a]].append(rid)
        expected[rid] = [ids[b]]
        if count % 5000 == 0:
            print({"stage":"write_movement_roads","completed":count,"total":len(movement_groups)},flush=True)
    for a, b in edges:
        if (a, b) in covered_edges:
            continue
        if a not in cuts or b not in cuts or (a, "end") in junction_for_end or (b, "start") in junction_for_end:
            raise ValueError(f"Source successor edge not covered by a movement: {a} -> {b}")
        roads[ids[a]].add_successor(xodr.ElementType.road, ids[b], xodr.ContactPoint.start)
        roads[ids[b]].add_predecessor(xodr.ElementType.road, ids[a], xodr.ContactPoint.end)
        roads[ids[a]].lanes.lanesections[0].rightlanes[0].add_link("successor", -1)
        roads[ids[b]].lanes.lanesections[0].rightlanes[0].add_link("predecessor", -1)
        expected[ids[a]].append(ids[b])
        covered_edges.add((a, b))
        direct_edges.append((a, b))
    for rid in sorted(roads):
        network.add_road(roads[rid])
    for jid in sorted(junctions):
        network.add_junction(junctions[jid])
    path = output / "roads.xodr"
    network.write_xml(str(path))
    lane_intervals = {key: [] for key in lanes}
    for rid, pieces in intervals.items():
        offset = 0.0
        for key, start, end in pieces:
            if key in lane_intervals:
                lane_intervals[key].append({"road_id": rid, "lane_id": -1, "road_s_start_m": offset,
                                           "road_s_end_m": offset + end - start, "source_s_start_m": start,
                                           "source_s_end_m": end, "junction_id": metadata[str(rid)]["junction_id"]})
            offset += end - start
    physical = [{key: value for key, value in g.items() if key != "paths"} for g in groups]
    movement_ids={g['id']:[] for g in physical}
    for rid in sorted(roads):
        jid=metadata[str(rid)]['junction_id']
        if jid is not None:movement_ids[jid].append(rid)
    for group in physical:
        group["movement_road_ids"] = movement_ids[group['id']]
    mapping = {"key_to_road_id": ids, "allocated_road_ids": allocated,
               "road_lane_ids": {str(ids[key]): key for key in sorted(cuts)}, "roads": metadata,
               "lane_to_road_intervals": lane_intervals, "physical_junctions": physical,
               "native_permission_scope": "CARLA waypoint links do not enforce vehicle classes. Preserve allowed_classes for class-aware routing; non-passenger lanes use restricted/biking types."}
    mapping_path.write_text(json.dumps(mapping, indent=2) + "\n")
    return {"path": path, "roads": roads, "intervals": intervals, "geometries": geometries, "expected": expected,
            "mapping": mapping, "fits": fits, "bridges": [v for v in bridges.values() if v], "external_successors": external,
            "junction_count": len(junctions), "source_edges": edges, "lanes": lanes,
            "covered_source_edges": covered_edges, "direct_source_edges": direct_edges}


def close_surface_connections(meshes, road_mesh_paths, successors):
    """Fill cap gaps only along the compiled road graph, without moving lanes."""
    import shapely

    connections = {pair for path in road_mesh_paths.values() for pair in zip(path, path[1:])}
    connections.update((road_mesh_paths[a][-1], road_mesh_paths[b][0])
                       for a, targets in successors.items() for b in targets)
    triangles = np.concatenate([np.asarray(m["vertices"])[np.asarray(m["faces"],dtype=int).reshape(-1,3)] for m in meshes])
    polygons = shapely.polygons(triangles[:, :, :2])
    tree = STRtree(polygons)
    # Union only faces intersecting the current cap.
    patch_bins, patches = {}, []
    def cells(geometry):
        low = np.floor(np.asarray(geometry.bounds[:2]) / 10).astype(int)
        high = np.floor(np.asarray(geometry.bounds[2:]) / 10).astype(int)
        return [(x, y) for x in range(low[0], high[0] + 1)
                for y in range(low[1], high[1] + 1)]
    records = []
    for count, (first, second) in enumerate(sorted(connections), 1):
        if count % 5000 == 0:
            print(dict(stage="close_surface_connections", completed=count,
                       total=len(connections), patches=len(records)), flush=True)
        a, b = meshes[first], meshes[second]
        caps = np.asarray([a["left_boundary"][-1], a["right_boundary"][-1],
                           b["left_boundary"][0], b["right_boundary"][0]])
        # Rotated cross sections can cross at their centers. Their convex hull
        # fills both wedges; an ordered quad would become a self-crossing face.
        hull = shapely.convex_hull(shapely.multipoints(caps[:, :2]))
        if hull.area == 0:
            continue
        nearby = polygons[tree.query(hull, predicate="intersects")]
        previous = {i for cell in cells(hull) for i in patch_bins.get(cell, [])}
        footprint = shapely.union_all([*nearby, *(patches[i] for i in sorted(previous))])
        missing = shapely.difference(hull, footprint)
        if missing.area == 0:
            continue
        vertices, faces = [], []
        for base in shapely.get_parts(shapely.constrained_delaunay_triangles(hull)):
            xy = np.asarray(base.exterior.coords)[:3]
            nearest = np.linalg.norm(xy[:, None] - caps[None, :, :2], axis=2).argmin(axis=1)
            xyz = np.column_stack((xy, caps[nearest, 2]))
            normal = np.cross(xyz[1] - xyz[0], xyz[2] - xyz[0])
            for patch in shapely.get_parts(shapely.constrained_delaunay_triangles(base.intersection(missing))):
                points = np.asarray(patch.exterior.coords)[:3]
                z = xyz[0, 2] - ((points - xyz[0, :2]) * normal[:2]).sum(axis=1) / normal[2]
                tri = np.column_stack((points, z))
                area = float(np.cross(tri[1] - tri[0], tri[2] - tri[0])[2])
                if abs(area) <= 1e-12:
                    continue
                if area < 0:
                    tri = tri[::-1]
                faces.append(list(range(len(vertices), len(vertices) + 3)))
                vertices.extend(tri.tolist())
        if not faces:
            continue
        patch_id = f"surface_closure_{len(records)}"
        meshes.append({"id": patch_id, "kind": "connection_closure", "source_lane_id": None,
                       "source_interval_m": None, "xodr_road_id": a["xodr_road_id"], "xodr_lane_id": -1,
                       "junction_id": a["junction_id"], "from_mesh_id": a["id"], "to_mesh_id": b["id"],
                       "vertices": vertices, "faces": faces})
        patch_triangles = np.asarray(vertices)[faces]
        coverage = shapely.union_all(shapely.polygons(patch_triangles[:, :, :2]))
        residual = float(missing.difference(coverage).area)
        residual_width=0.
        if residual>1e-8:
            remaining=missing.difference(coverage)
            parts=shapely.get_parts(remaining);parts=parts[shapely.area(parts)>0]
            circles=shapely.maximum_inscribed_circle(parts,tolerance=1e-7)
            residual_width=float(2*np.max(shapely.length(circles)))
            # Require numerical slivers to fit within the 0.1 mm endpoint margin.
            if residual_width>1e-4:
                raise ValueError(f"Connection closure left uncovered area at {a['id']} -> {b['id']}: area={residual}, width={residual_width}")
        index = len(patches)
        patches.append(coverage)
        for cell in cells(coverage):
            patch_bins.setdefault(cell, []).append(index)
        records.append({"id": patch_id, "from_mesh_id": a["id"], "to_mesh_id": b["id"],
                        "from_source": a["source_lane_id"], "to_source": b["source_lane_id"],
                        "area_m2": float(coverage.area), "triangles": len(faces),
                        "center_gap_m": float(np.linalg.norm(caps[:2].mean(axis=0) - caps[2:].mean(axis=0))),
                        "cap_span_m": float(np.linalg.norm(caps[:, None] - caps[None, :], axis=2).max()),
                        "height_difference_m": float(np.ptp(caps[:, 2])), "uncovered_area_m2": residual,
                        "uncovered_inscribed_diameter_m":residual_width})
    return {"method": "Convex hull of adjacent sampled end/start caps, clipped to uncovered XY area and triangulated at interpolated cap heights. Only compiled piece order and road successor links are joined; original vertices and XODR remain unchanged.",
            "connection_count": len(connections), "patch_count": len(records),
            "area_m2": sum(r["area_m2"] for r in records),
            "max_center_gap_m": max((r["center_gap_m"] for r in records), default=0),
            "max_cap_span_m": max((r["cap_span_m"] for r in records), default=0),
            "max_height_difference_m": max((r["height_difference_m"] for r in records), default=0),
            "patches": records}


def sample_and_check(scene, built, output, sample_step=0.5, carla_root=None, reuse_samples=False, native_planner=False):
    import carla
    output = Path(output)
    parsed = carla.Map(str(scene["map_id"]), built["path"].read_text())
    lengths = (built["road_lengths"] if "road_lengths" in built else
               {rid: road.planview.get_total_length() for rid, road in built["roads"].items()})
    meshes, seen, parse_errors, height_errors, width_errors = [], {}, [], [], []
    road_mesh_paths = {}
    per_lane_errors = {}
    mesh_winding = {"reoriented_triangles": 0, "zero_area_triangles_omitted": 0}
    width_steps = []
    cache_path=output/'surface_samples.pkl'
    if reuse_samples:
        with cache_path.open('rb') as stream:
            cache=pickle.load(stream)
        if cache.get('sampling_version')!=2 or cache['sample_step']!=sample_step or cache['map_id']!=scene['map_id']:
            raise ValueError('Sampling cache belongs to a different map, spacing or sampler version')
        (meshes,parse_errors,height_errors,width_errors,road_mesh_paths,per_lane_errors,
         mesh_winding,width_steps)=cache['values']
    else:
        for count,(rid, pieces) in enumerate(built["intervals"].items(),1):
            if count%5000==0:
                print(dict(stage='sample_native_roads',completed=count,total=len(built['intervals']),
                           samples=len(parse_errors)),flush=True)
            road_mesh_paths[rid] = []
            piece_offsets = np.r_[0.0, np.cumsum([end - start for _, start, end in pieces])]
            for previous, current in zip(pieces, pieces[1:]):
                delta = built["geometries"][current[0]].width(current[1]) - built["geometries"][previous[0]].width(previous[2])
                if abs(delta) > 0.01:
                    width_steps.append({"road_id": rid, "from_source": previous[0], "to_source": current[0], "width_change_m": delta})
            cursor = 0.0
            for key, start, end in pieces:
                geometry = built["geometries"][key]
                vertices, faces, left, right = [], [], [], []
                errors = []
                # Dense road profiles and original knots use the exact same final XODR.
                sample_s = np.unique(np.r_[np.linspace(0, end - start, max(3, math.ceil((end - start) / sample_step) + 1)),
                                           geometry.knots[(geometry.knots > start) & (geometry.knots < end)] - start])
                sample_s = sample_s[np.r_[True, np.diff(sample_s) > 1e-6]]
                # CARLA accepts float32 stations. Keep each source cap on its
                # own side of a piece boundary, where elevation/width may jump.
                margin = min(1e-4, lengths[rid] * 1e-3)
                lower = max(cursor, margin)
                upper = min(cursor + end - start, lengths[rid] - margin)
                low32, high32 = np.float32(lower), np.float32(upper)
                if float(low32) <= cursor:
                    low32 = np.nextafter(low32, np.float32(np.inf))
                if float(high32) >= cursor + end - start:
                    high32 = np.nextafter(high32, np.float32(-np.inf))
                if low32 > high32:
                    raise ValueError(f"Source piece cannot be sampled in native float32 stations: road {rid}, source {key}")
                for local in sample_s:
                    s = float(np.clip(np.float32(cursor + local), low32, high32))
                    waypoint = parsed.get_waypoint_xodr(rid, -1, s)
                    if waypoint is None:
                        raise ValueError(f"CARLA cannot sample road {rid} at {s}/{lengths[rid]}")
                    p = waypoint.transform.location
                    xyz = np.array([p.x, -p.y, p.z])
                    source_s = float(np.clip(start + s - cursor, 0, geometry.length))
                    expected_xyz = geometry.xyz(source_s)
                    error = float(np.linalg.norm(xyz - expected_xyz))
                    errors.append(error)
                    height_errors.append(float(abs(xyz[2] - expected_xyz[2])))
                    # Width records are right-continuous at piece boundaries; float32
                    # s may lie in the following source interval at their shared knot.
                    active = min(len(pieces) - 1, int(np.searchsorted(piece_offsets, s, side="right") - 1))
                    active_key, active_start, _ = pieces[active]
                    expected_width = built["geometries"][active_key].width(active_start + s - piece_offsets[active])
                    width_errors.append(abs(waypoint.lane_width - expected_width))
                    forward = waypoint.transform.get_forward_vector()
                    normal = np.array([forward.y, forward.x, 0.0])
                    normal /= np.linalg.norm(normal)
                    l = (xyz + normal * waypoint.lane_width / 2).tolist()
                    r = (xyz - normal * waypoint.lane_width / 2).tolist()
                    left.append(l); right.append(r)
                    vertices.extend([l, r])
                    if len(left) > 1:
                        i = 2 * len(left) - 4
                        for face in ([i, i + 1, i + 3], [i, i + 3, i + 2]):
                            a, b, c = np.asarray([vertices[j] for j in face])[:, :2]
                            area = float(np.cross(b - a, c - a))
                            if abs(area) < 1e-10:
                                mesh_winding["zero_area_triangles_omitted"] += 1
                                continue
                            if area < 0:
                                face = [face[0], face[2], face[1]]
                                mesh_winding["reoriented_triangles"] += 1
                            faces.append(face)
                parse_errors.extend(errors)
                per_lane_errors[key] = max(per_lane_errors.get(key, 0), max(errors))
                interval_key = (key, round(start, 9), round(end, 9))
                if interval_key not in seen:
                    metadata = built["mapping"]["roads"][str(rid)]
                    meshes.append({"id": f"road_surface_{len(meshes)}", "source_lane_id": key,
                                   "source_interval_m": [start, end], "xodr_road_id": rid, "xodr_lane_id": -1,
                                   "junction_id": metadata["junction_id"], "vertices": vertices, "faces": faces,
                                   "left_boundary": left, "right_boundary": right})
                    seen[interval_key] = len(meshes) - 1
                road_mesh_paths[rid].append(seen[interval_key])
                cursor += end - start
        if len(meshes)>10000:
            with cache_path.open('wb') as stream:
                pickle.dump(dict(sampling_version=2,sample_step=sample_step,map_id=scene['map_id'],values=(meshes,
                    parse_errors,height_errors,width_errors,road_mesh_paths,per_lane_errors,
                    mesh_winding,width_steps)),stream,protocol=5)
            print(dict(stage='native_surface_samples_saved',meshes=len(meshes),samples=len(parse_errors)),flush=True)
    source_piece_joins = []
    for rid, pieces in built["intervals"].items():
        for previous, current in zip(pieces, pieces[1:]):
            a = built["geometries"][previous[0]].xyz(previous[2])
            b = built["geometries"][current[0]].xyz(current[1])
            delta = np.asarray(b) - np.asarray(a)
            if float(np.linalg.norm(delta)) > 1e-6:
                source_piece_joins.append(dict(road_id=rid, from_source=previous[0], to_source=current[0],
                    xyz_delta_m=delta.tolist(), position_gap_m=float(np.linalg.norm(delta)),
                    height_step_m=float(delta[2])))
    closure = close_surface_connections(meshes, road_mesh_paths, built["expected"])
    mesh_path = output / "road_meshes.json"
    mesh_path.write_text(json.dumps({"frame": "SceneIR/Blender ENU metres; CARLA Y reflected back; source z retained",
                                    "sampling_step_m": sample_step, "meshes": meshes, "connection_closure": closure,
                                    "surface_scope": "Horizontal lane cross sections; no fitted banking. Exact shared movement source intervals appear once. Graph-adjacent cap gaps receive explicit closure patches. XY footprint is the union of these triangle faces."}) + "\n")
    observed, issues = {}, []
    for rid, targets in built["expected"].items():
        margin = min(0.001, lengths[rid] * 0.05)
        s = float(np.float32(lengths[rid] - margin))
        waypoint = parsed.get_waypoint_xodr(rid, -1, s)
        step = lengths[rid] - s + min([lengths[t] / 4 for t in targets] or [0.01])
        actual = sorted({w.road_id for w in waypoint.next(step)})
        observed[rid] = actual
        if actual != sorted(targets):
            issues.append({"road_id": rid, "expected_next": sorted(targets), "actual_next": actual})
    ids = built["mapping"]["key_to_road_id"]
    for a, b in built["direct_source_edges"]:
        first, second = built["geometries"][a], built["geometries"][b]
        delta = second.width(0) - first.width(first.length)
        if abs(delta) > 0.01:
            width_steps.append({"road_id": ids[a], "next_road_id": ids[b],
                                "from_source": a, "to_source": b, "width_change_m": delta})
    # Native topology can expose wrong-direction edges that Waypoint.next filters out.
    topology = {(a.road_id, b.road_id) for a, b in parsed.get_topology() if a.road_id != b.road_id}
    expected_topology = {(a, b) for a, targets in built["expected"].items() for b in targets
                         if "passenger" in built["mapping"]["roads"][str(a)]["allowed_classes"]}
    topology_extra = sorted(topology - expected_topology)
    topology_missing = sorted(expected_topology - topology)
    source_edges = set(built["source_edges"])
    missing_source_edges = sorted(source_edges - built["covered_source_edges"])
    extra_source_edges = sorted(built["covered_source_edges"] - source_edges)
    core_ids = scene.get("frame_validation", {}).get("route", {}).get("lane_ids", [])
    core_links, core_resolution = native_source_route_checks(
        core_ids, built['lanes'], built['mapping'],
        {(a,b) for a,targets in observed.items() for b in targets},
        {row['source']: row['path'] for row in scene.get('source_interval_ownership', {}).get('review', {}).get('affected_source_paths', [])})
    fits = built["fits"]
    internal_coverage, junction_checks = [], []
    for key, lane in built["lanes"].items():
        if lane.get("internal") or lane.get("is_intersection"):
            copies = built["mapping"]["lane_to_road_intervals"][key]
            waypoints = [parsed.get_waypoint_xodr(row["road_id"], -1, (row["road_s_start_m"] + row["road_s_end_m"]) / 2) for row in copies]
            internal_coverage.append({"scene_lane_id": key, "road_ids": sorted({row["road_id"] for row in copies}),
                                      "junction_ids": sorted({wp.junction_id for wp in waypoints}),
                                      "native_is_junction": bool(waypoints) and all(wp.is_junction for wp in waypoints),
                                      "reviewed_open_boundary": scene["fusion_map"].get("open_internal_boundaries", {}).get(key),
                                      "native_contains_open_port": any(not wp.is_junction for wp in waypoints)})
    for group in built["mapping"]["physical_junctions"]:
        rid = group["movement_road_ids"][0]
        junction = parsed.get_waypoint_xodr(rid, -1, lengths[rid] / 2).get_junction()
        members = sorted({a.road_id for a, b in junction.get_waypoints(carla.LaneType.Any)})
        junction_checks.append({"junction_id": group["id"], "expected_road_ids": group["movement_road_ids"],
                                "native_road_ids": members, "passed": junction.id == group["id"] and members == group["movement_road_ids"]})
    report = {"passed": not (issues or topology_extra or topology_missing or missing_source_edges or extra_source_edges)
              and max(parse_errors) < 0.03 and max(width_errors) < 0.001
              and all(c["carla_successor_reachable"] for c in core_links)
              and all(row["native_is_junction"] or (row["reviewed_open_boundary"] and row["native_contains_open_port"])
                      for row in internal_coverage) and all(row["passed"] for row in junction_checks),
              "source_lane_count": len(built["lanes"]), "road_count": len(built["roads"]), "junction_count": built["junction_count"],
              "source_successor_edges": len(built["source_edges"]), "xodr_successor_checks": len(observed),
              "routing_issues": issues, "native_topology": {"edge_count": len(topology), "unexpected_edges": topology_extra, "missing_edges": topology_missing},
              "source_edge_coverage": {"covered": len(built["covered_source_edges"]), "missing": missing_source_edges, "extra": extra_source_edges},
              "physical_junction_checks": junction_checks, "core_route": core_links,
              "core_route_resolution": core_resolution, "external_successor_references": built["external_successors"],
              "source_fit": {"method": "SciPy PCHIP xy retaining source knots, or explicit boundary_shape cubic segments; elevation PCHIP on planar arc length; linear endpoint width",
                             "max_xy_deviation_m": max(v["max_xy_deviation_from_source_polyline_m"] for v in fits.values()),
                             "max_height_deviation_m": max(v["max_height_deviation_from_source_polyline_m"] for v in fits.values()), "lanes": fits},
              "carla_parameter_check": {"sample_count": len(parse_errors), "max_position_error_m": max(parse_errors),
                                        "max_height_error_m": max(height_errors), "max_width_error_m": max(width_errors),
                                        "worst_sources": sorted(per_lane_errors.items(), key=lambda x: x[1], reverse=True)[:10]},
              "source_gap_bridges": built["bridges"], "source_width_steps_at_connections": width_steps,
              "source_piece_cap_offsets": {"scope": "Retained source endpoint differences; one-sided native sampling does not alter them. Physical surface and driving review remain required.",
                  "count": len(source_piece_joins), "maximum_gap_m": max((x["position_gap_m"] for x in source_piece_joins), default=0),
                  "joins": source_piece_joins},
              "mesh_count": len(meshes), "mesh_winding": mesh_winding, "connection_closure": closure,
              "source_internal_junction_coverage": {"source_internal_lane_count": len(internal_coverage),
                  "native_is_junction_count": sum(item["native_is_junction"] for item in internal_coverage),
                  "fraction": sum(item["native_is_junction"] for item in internal_coverage) / len(internal_coverage) if internal_coverage else None,
                  "lanes": internal_coverage,
                  "scope": "All source internal lanes are represented inside standard full entry-to-exit connecting roads, grouped by SUMO node and AV2 intersection geometry/connectivity. Native junction membership is checked for every copy."},
              "scope": "Offline CARLA geometry and explicit graph traversal. Vehicle-class legality remains SceneIR metadata; no traffic signals, server driving, banking or GNSS validation."}
    # Retain completed geometry/topology evidence before constructing a planner.
    (output / "road_geometry_check.json").write_text(json.dumps(report, indent=2) + "\n")
    if carla_root and core_ids:
        sys.path.insert(0, str(Path(carla_root) / "PythonAPI/carla"))
        from agents.navigation.global_route_planner import GlobalRoutePlanner
        planner_class = GlobalRoutePlanner
        if native_planner:
            from map_generation.native_route_planner import NativeRoutePlanner
            planner_class = NativeRoutePlanner
        resolved = core_resolution['resolved_lane_ids']
        first_source, last_source = resolved[0], resolved[-1]
        first_rows = built['mapping']['lane_to_road_intervals'][first_source]
        last_rows = built['mapping']['lane_to_road_intervals'][last_source]
        first_station = min(row['source_s_start_m'] for row in first_rows)
        last_station = max(row['source_s_end_m'] for row in last_rows)
        starts = [row for row in first_rows if abs(row['source_s_start_m'] - first_station) < 1e-7]
        ends = [row for row in last_rows if abs(row['source_s_end_m'] - last_station) < 1e-7]
        import networkx as nx
        native_graph = nx.DiGraph()
        native_graph.add_nodes_from(int(rid) for rid, meta in built['mapping']['roads'].items() if 'passenger' in meta['allowed_classes'])
        native_graph.add_edges_from((a,b) for a,targets in observed.items() for b in targets if a in native_graph and b in native_graph)
        options = []
        for a in starts:
            for b in ends:
                try:
                    native_path = nx.shortest_path(native_graph, a['road_id'], b['road_id'])
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
                options.append((len(native_path), a['road_id'], b['road_id'], a, b))
        if not options:
            raise ValueError('No native route joins the complete resolved source probe endpoints')
        _, first, last, first_interval, last_interval = min(options, key=lambda row: row[:3])
        start_s = first_interval['road_s_start_m'] + min(1.0, (first_interval['road_s_end_m']-first_interval['road_s_start_m'])/4)
        end_s = last_interval['road_s_end_m'] - min(1.0, (last_interval['road_s_end_m']-last_interval['road_s_start_m'])/4)
        start = parsed.get_waypoint_xodr(first, -1, start_s)
        end = parsed.get_waypoint_xodr(last, -1, end_s)
        core_resolution['official_planner_endpoints'] = dict(start_source_lane=first_source, end_source_lane=last_source,
            start_native_interval=first_interval, end_native_interval=last_interval,
            start_road_s_m=start_s, end_road_s_m=end_s, compatible_endpoint_pairs=len(options))
        # Save geometry and topology results before reporting planner failures.
        planner = planner_class(parsed, 1.0)
        try:
            trace = planner.trace_route(start.transform.location, end.transform.location)
            planner_error = None
        except nx.NetworkXNoPath as error:
            trace = []
            planner_error = str(error)
        nearest_start = parsed.get_waypoint(start.transform.location)
        nearest_end = parsed.get_waypoint(end.transform.location)
        core_resolution['official_planner_localization'] = dict(
            error=planner_error,
            requested_start_road=first, requested_end_road=last,
            nearest_start_road=nearest_start.road_id, nearest_end_road=nearest_end.road_id,
            requested_start_xyz=[start.transform.location.x,start.transform.location.y,start.transform.location.z],
            requested_end_xyz=[end.transform.location.x,end.transform.location.y,end.transform.location.z],
            nearest_start_s=nearest_start.s,nearest_end_s=nearest_end.s)

        traced_ids = list(dict.fromkeys(w.road_id for w, _ in trace))
        class_allowed = all("passenger" in built["mapping"]["roads"][str(rid)]["allowed_classes"] for rid in traced_ids)
        ok = bool(trace) and trace[0][0].road_id == first and trace[-1][0].road_id == last and class_allowed
        probe = {"passed": ok, "waypoint_count": len(trace), "road_ids": traced_ids,
                 "requested_start_road": first, "requested_end_road": last,
                 "source_metadata_allows_passenger": class_allowed}
        # Check the full movement through its receiving lane when coordinate localization is ambiguous.
        end_metadata = built["mapping"]["roads"][str(last)]
        if not ok and nearest_end.road_id != last and end_metadata["kind"] == "movement":
            receiver = end_metadata["to_lane_id"]
            receiver_id = built["mapping"]["key_to_road_id"][receiver]
            if receiver_id in native_graph and (last, receiver_id) in native_graph.edges:
                exit_s = lengths[receiver_id] / 2
                exit_waypoint = parsed.get_waypoint_xodr(receiver_id, -1, exit_s)
                localized_exit = parsed.get_waypoint(exit_waypoint.transform.location)
                try:
                    exit_trace = planner.trace_route(start.transform.location, exit_waypoint.transform.location)
                    exit_error = None
                except nx.NetworkXNoPath as error:
                    exit_trace, exit_error = [], str(error)
                exit_ids = list(dict.fromkeys(w.road_id for w, _ in exit_trace))
                exit_classes = all("passenger" in built["mapping"]["roads"][str(rid)]["allowed_classes"]
                                   for rid in exit_ids)
                full_movement = last in exit_ids and (last, receiver_id) in set(zip(exit_ids, exit_ids[1:]))
                exit_ok = (bool(exit_trace) and localized_exit.road_id == receiver_id
                           and exit_trace[0][0].road_id == first and exit_trace[-1][0].road_id == receiver_id
                           and full_movement and exit_classes)
                probe = {"passed": exit_ok, "waypoint_count": len(exit_trace), "road_ids": exit_ids,
                         "requested_start_road": first, "requested_end_road": receiver_id,
                         "source_metadata_allows_passenger": exit_classes,
                         "original_location_probe": probe,
                         "junction_exit_probe": {"reason": "Original destination localizes to a different converging movement",
                             "receiver_source_lane": receiver, "receiver_road_id": receiver_id, "receiver_s_m": exit_s,
                             "localized_receiver_road_id": localized_exit.road_id,
                             "complete_requested_movement_traversed": full_movement, "error": exit_error,
                             "scope": "Existing receiving lane continuation; original coordinate localization failure retained."}}
        # Use explicit native endpoints after recording coordinate-based localization.
        if native_planner and not probe["passed"] and (
                nearest_start.road_id != first or nearest_end.road_id != last):
            location_probe = probe
            try:
                source_trace = planner.trace_waypoints(start, end)
                source_error = None
            except nx.NetworkXNoPath as error:
                source_trace, source_error = [], str(error)
            source_ids = list(dict.fromkeys(w.road_id for w, _ in source_trace))
            source_classes = all("passenger" in built["mapping"]["roads"][str(rid)]["allowed_classes"]
                                 for rid in source_ids)
            source_ok = (bool(source_trace) and source_trace[0][0].road_id == first
                         and source_trace[-1][0].road_id == last and source_classes)
            probe = {"passed": source_ok, "waypoint_count": len(source_trace), "road_ids": source_ids,
                     "requested_start_road": first, "requested_end_road": last,
                     "source_metadata_allows_passenger": source_classes,
                     "original_location_probe": location_probe,
                     "source_identity_probe": {"error": source_error,
                         "start_road_s_m": start_s, "end_road_s_m": end_s,
                         "scope": "Original source-backed native endpoints; no road or permission change. "
                                  "Nearest-coordinate localization failure retained; no driving claim."}}
        probe["implementation"] = type(planner).__name__
        report["official_global_route_planner"] = probe
        report["passed"] &= probe["passed"]
    (output / "road_build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def build_roads(scene, output, trim=0.5, fit_step=1.0, sample_step=0.5, carla_root=None, workers=1, reuse_xodr=False, reuse_samples=False, native_planner=False):
    built = build_network(scene, output, trim, fit_step, workers, reuse_xodr)
    return sample_and_check(scene, built, output, sample_step, carla_root, reuse_samples, native_planner)
