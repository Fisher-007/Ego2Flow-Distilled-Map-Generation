"""Realize source-supported vehicle space inside coarse building masses."""
from collections import defaultdict
from copy import deepcopy
import re
import numpy as np
from shapely.geometry import LineString, Polygon, Point, MultiPoint, mapping
from shapely.ops import unary_union
from shapely.strtree import STRtree
from shapely import set_precision, get_coordinates
from .buildings import _polygons


def height_metres(value):
    if value is None:
        return None
    value = str(value).strip()
    imperial = re.fullmatch(r"(\d+(?:\.\d+)?)\s*'\s*(\d+(?:\.\d+)?)?\s*\"?", value)
    if imperial:
        return float(imperial[1]) * .3048 + float(imperial[2] or 0) * .0254
    metric = re.fullmatch(r'(\d+(?:\.\d+)?)\s*m?', value)
    return float(metric[1]) if metric else None


def passage_evidence(lane):
    tags = lane.get('source_tags', {})
    if tags.get('covered') == 'yes' or tags.get('tunnel') in ('building_passage', 'yes') or tags.get('indoor') == 'yes':
        return {k: tags[k] for k in ('covered', 'tunnel', 'indoor', 'maxheight', 'level', 'layer') if k in tags}
    return None


def open_deck_evidence(lane):
    """An explicitly uncovered parking level supplies an open-sky cue."""
    tags=lane.get('source_tags',{})
    if tags.get('covered')!='no' or 'level' not in tags:return None
    try:level=float(tags['level'])
    except ValueError:return None
    if level<0:return None
    return dict(covered='no',level=tags['level'],open_top=True)


def tagged_lower_envelope(building):
    """Height evidence for an elevated building part, with level assumptions."""
    tags = building['tags']
    height = height_metres(tags.get('min_height'))
    source = 'OSM min_height' if height is not None else None
    fractional = False
    if height is None and 'building:min_level' in tags and 'building:levels' in tags:
        lower, total = float(tags['building:min_level']), float(tags['building:levels'])
        if 0 < lower < total:
            height = building['nominal_height_m'] * lower / total
            fractional = not lower.is_integer()
            source = 'OSM building:min_level with the existing nominal storey-height assumption'
    if height is None:
        return None
    ground = building.get('final_support_range_m', building.get('ground_range_m'))
    base = ground[1] if ground else building['roof_z_m']-building['nominal_height_m']
    return dict(bottom_z_m=base + height, height_above_ground_m=height, source=source,
        vertical_datum_z_m=base, vertical_datum_basis='Same upper support datum as the source roof/nominal height; foundation relief is not occupied upper-storey thickness',
        tags={k: tags[k] for k in ('min_height', 'building:min_level', 'building:levels') if k in tags},
        nonstandard_fractional_level=fractional,
        scope=('Fractional min_level is nonstandard; interpreted as a proportional geometry cue alongside a source-tagged passage. Physical clearance remains inferred and must be reviewed.'
               if fractional else 'Source lower-envelope geometry; level-to-metre conversion is inferred, not surveyed.'))



def eligible_vehicle_height(scene, lane_id, default_height_m=2.2):
    """Read the vehicle-height limit recorded for the route."""
    records=[c for c in scene.get('vehicle_height_conditions',[])
             if lane_id in c['lane_ids']]
    if not records:return float(default_height_m),None
    condition=min(records,key=lambda c:c['legal_maxheight_m'])
    return min(float(default_height_m),float(condition['legal_maxheight_m'])),condition['id']


def vehicle_sweep(points, length_m, width_m):
    """Body space along segments and a finite neighbourhood of each turn.

    A coarse polyline corner represents a moving heading transition. Sweep
    its intermediate headings over the adjacent half-body-length reaches,
    clipped to the source segments. This constructs conservative access space;
    actual steering and pavement support remain native driving checks.
    """
    points=np.asarray(points)[:,:2];segments=np.diff(points,axis=0)
    lengths=np.linalg.norm(segments,axis=1);valid=lengths>1e-7
    starts=points[:-1][valid];ends=points[1:][valid];directions=segments[valid]/lengths[valid,None]
    lengths=lengths[valid]
    parts=[];half_length=length_m/2;half_width=width_m/2
    for a,b,d in zip(starts,ends,directions):
        parts.append(LineString([a-half_length*d,b+half_length*d]).buffer(half_width,cap_style='flat'))
    for i in range(1,len(directions)):
        yaw0=np.arctan2(directions[i-1,1],directions[i-1,0])
        turn=(np.arctan2(directions[i,1],directions[i,0])-yaw0+np.pi)%(2*np.pi)-np.pi
        if abs(turn)<1e-7:continue
        c=starts[i]
        centers=[c-directions[i-1]*min(half_length,lengths[i-1]),c,
                 c+directions[i]*min(half_length,lengths[i])]
        for yaw in yaw0+np.linspace(0,turn,max(2,int(np.ceil(abs(turn)/np.deg2rad(5)))+1)):
            d=np.array([np.cos(yaw),np.sin(yaw)]);n=np.array([-d[1],d[0]])
            offsets=[a*half_length*d+b*half_width*n for a,b in [(-1,-1),(1,-1),(1,1),(-1,1)]]
            for a,b in zip(centers[:-1],centers[1:]):
                parts.append(MultiPoint([center+offset for center in (a,b)for offset in offsets]).convex_hull)
    return unary_union(parts)


def realize(scene, default_clearance_m=4.5, lateral_margin_m=.3,
            vehicle_length_m=4.8,vehicle_width_m=2.2):
    """Retain covered/indoor routes and source-tagged open parking decks.

    Connected seams inherit the same local access space. Covered routes keep
    upper masses; parking deck support comes from source road surfaces. Missing
    headroom uses a recorded geometry prior. A maxheight tag is an access rule.
    """
    lanes = {l['id']: l for l in scene['fusion_map']['lanes']}
    canonical = scene.get('building_passage_source_buildings', [])
    parents = {b['id'] for b in canonical}
    buildings = [b for b in scene['buildings'] if b.get('passage_parent_id', b['id']) not in parents] + canonical
    shapes = [Polygon(b['exterior'], b['holes']) for b in buildings]
    tree = STRtree(shapes)
    links = defaultdict(set)
    for lane in lanes.values():
        for target in lane['successors']:
            if target in lanes:
                links[lane['id']].add(target)
                links[target].add(lane['id'])
    tagged = {k: evidence for k,l in lanes.items()
              if (evidence:=passage_evidence(l) or open_deck_evidence(l)) is not None}
    # Source-selected entrances define swept openings at facade terminals.
    seeds=list(tagged.items())
    for condition in scene.get('building_access_conditions', []):
        way_ids=set(map(str,condition.get('osm_way_ids',[])))
        lane_ids=set(condition.get('lane_ids',[]))
        for key,lane in lanes.items():
            way=str(lane.get('edge_id','')).lstrip('-').split('#')[0]
            if key in lane_ids or (not lane.get('internal') and way in way_ids):
                seeds.append((key,dict(condition,explicit_entrance=True)))
    by_building = defaultdict(list)
    for key, evidence in seeds:
        lane = lanes[key]
        source_line = LineString(np.asarray(lane['centerline'])[:, :2])
        explicit=evidence.get('explicit_entrance',False)
        reach=(vehicle_length_m**2+max(vehicle_width_m,lane['width_m'])**2)**.5/2+lateral_margin_m
        source_pavement=source_line.buffer(lane['width_m']/2,cap_style='flat')
        query=source_line.buffer(reach) if explicit else source_pavement
        for i in tree.query(query, predicate='intersects'):
            building=buildings[int(i)]
            if explicit:
                sources=set(building.get('volume_source_features',[]))|{building.get('source_id')}
                if not sources.intersection(evidence['building_source_ids']):continue
            open_top=evidence.get('open_top',False)
            tags=building['tags']
            if open_top and not explicit and not(tags.get('parking')=='multi-storey' or tags.get('building')=='parking'):
                continue
            # Use the full road-width envelope to identify covered facade contacts.
            if not explicit and source_pavement.intersection(shapes[i]).area < .01:
                continue
            selected = {key}
            queue = list(links[key])
            while queue:
                other = queue.pop()
                if other in selected:
                    continue
                road = lanes[other]
                line=LineString(np.asarray(road['centerline'])[:,:2])
                short_approach=(line.length<=vehicle_length_m+2*lateral_margin_m and
                                not passage_evidence(road) and not open_deck_evidence(road))
                if not (road.get('internal') or other.startswith('seam:') or short_approach):
                    continue
                reach=(vehicle_length_m**2+vehicle_width_m**2)**.5/2+lateral_margin_m
                if line.distance(shapes[i])>reach:
                    continue
                selected.add(other)
                queue.extend(links[other] - selected)
            envelopes = []; heights = []
            for other in selected:
                road = lanes[other]; points = np.asarray(road['centerline'])
                envelope = LineString(points[:, :2]).buffer(road['width_m'] / 2 + lateral_margin_m,
                                                            cap_style='square', join_style='round')
                # The centre can occupy the legal lateral portion of this
                # carriageway. Sweep the body over that portion, including
                # the front/rear corners at source heading changes.
                body=vehicle_sweep(points,vehicle_length_m,max(vehicle_width_m,road['width_m']))
                envelope=envelope.union(body.buffer(lateral_margin_m))
                envelopes.append(envelope)
                # Use local road points, including the nearest boundary point.
                inside = np.array([shapes[i].distance(Point(p[:2])) <= road['width_m']
                                   for p in points])
                heights.extend(points[inside, 2].tolist())
                # Interpolate entry and exit heights where sparse segments cross a building.
                path = LineString(points[:, :2])
                station = np.r_[0., np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1).cumsum()]
                contact = path.intersection(shapes[i].buffer(road['width_m']))
                for xy in get_coordinates(contact):
                    heights.append(float(np.interp(path.project(Point(xy)), station, points[:, 2])))
            legal = height_metres(evidence.get('maxheight'))
            clearance = float(evidence.get('inferred_clearance_m', default_clearance_m)) if legal is None else legal + .15
            roof_bottom = max(heights) + clearance
            lower = tagged_lower_envelope(buildings[int(i)])
            nominal_layer_feedback = (evidence.get('source_lower_envelope_policy') ==
                'road_conditioned_nominal_storeys' and lower is not None and
                'min_height' not in buildings[int(i)]['tags'])
            if lower is not None:
                roof_bottom = max(roof_bottom, lower['bottom_z_m']) if nominal_layer_feedback else lower['bottom_z_m']
                clearance = roof_bottom - max(heights)
            if open_top and min(heights)>=buildings[int(i)]['roof_z_m']:
                continue
            by_building[int(i)].append(dict(geometry=unary_union(envelopes).intersection(shapes[i]),
                roof_bottom=roof_bottom, lane_ids=sorted(selected), source_lane_id=key, evidence=evidence,
                open_top=open_top,road_height_range_m=[min(heights),max(heights)],
                legal_maxheight_m=legal, clearance_geometry_m=clearance,
                source_lower_envelope=lower, nominal_layer_feedback=nominal_layer_feedback,
                clearance_basis=lower['source'] if lower is not None else
                                'OSM maxheight plus 0.15 m modelling margin' if legal is not None else
                                'Declared missing-headroom geometry prior; not an observed height'))
    result = []; records = []
    for i, building in enumerate(buildings):
        proposals = by_building.get(i, [])
        if not proposals:
            result.append(building); continue
        covered=[p for p in proposals if not p['open_top']]
        outdoor=[p for p in proposals if p['open_top']]
        bottom = max((p['roof_bottom'] for p in covered),default=building['base_z_m'])
        report = dict(building_id=building['id'], relations=[dict(
            {k:v for k,v in p.items()if k!='geometry'}, clearance_footprint=mapping(p['geometry']),
            clearance_floor_z_m=p['road_height_range_m'][0],
            clearance_ceiling_z_m=(p['road_height_range_m'][1]+default_clearance_m
                                  if p['open_top']else p['roof_bottom']))for p in proposals])
        roof_structure = building['tags'].get('building') == 'roof'
        roof = building['roof_z_m']
        nominal_feedback=[p for p in covered if p.get('nominal_layer_feedback')]
        if nominal_feedback and building['height_source'] != 'OSM height':
            lower=nominal_feedback[0]['source_lower_envelope']
            upper_thickness=building['nominal_height_m']-lower['height_above_ground_m']
            roof=max(roof,bottom+upper_thickness)
            report['nominal_elevated_part_feedback']=dict(
                original_bottom_z_m=lower['bottom_z_m'],
                original_roof_z_m=building['roof_z_m'],conditioned_bottom_z_m=bottom,
                conditioned_roof_z_m=roof,retained_upper_storey_thickness_m=upper_thickness,
                reason='Explicit source tunnel/elevated-part condition resolves inferred ground reference and nominal storey heights; source footprint, levels and measured height tags remain unchanged.',
                scope='Inferred geometry, not surveyed underside/top; native review required.')
        if roof_structure and building['height_source'] != 'OSM height':
            # Use source headroom to position the underside of roof-only canopies.
            roof = max(roof, bottom + .35)
            report['roof_only_construction'] = dict(
                source_tag='building=roof', original_nominal_top_z_m=building['roof_z_m'],
                conditioned_top_z_m=roof, assumed_slab_thickness_m=.35,
                basis='Source roof-only role; passage headroom; inferred slab thickness. Levels-to-metres was not a surveyed roof height.',
                missing='Unobserved support posts are not placed inside source road space.')
        if covered and bottom >= roof - 1e-6:
            if building.get('source_parent_support'):
                # Retain the perimeter walls when a passage occupies the full lower slice.
                report['lower_slice_passage'] = dict(
                    slice_top_z_m=roof, required_passage_ceiling_z_m=bottom,
                    elevated_source_part=building['source_parent_support']['elevated_part_source_id'],
                    retained_upper_geometry='The elevated source part is realized independently; this lower slice supplies only its occupied outside walls.')
            else:
                report.update(status='requires_height_evidence', reason='Proposed clearance reaches the supplied roof')
                result.append(building); records.append(report); continue
        # Use a shared 0.01 mm Boolean grid to remove coincident-edge slivers before extrusion.
        grid = 1e-5
        shell = set_precision(shapes[i], grid)
        opening = set_precision(unary_union([p['geometry'] for p in proposals]), grid).intersection(shell)
        # Extend openings by two grid cells to clear rounding slivers at facade boundaries.
        opening_cut = set_precision(opening.buffer(2 * grid), grid)
        walls = shell.difference(opening_cut)
        open_area=set_precision(unary_union([p['geometry'] for p in outdoor]),grid).intersection(shell)
        covered_area=set_precision(unary_union([p['geometry'] for p in covered]),grid).intersection(shell)
        parking_structure=(building['tags'].get('parking')=='multi-storey' or
                           building['tags'].get('building')=='parking')
        if parking_structure:
            # Represent garages with perimeter shells and source-supported openings.
            # Construct access floors separately from their height evidence.
            shell_thickness=.35
            walls=walls.intersection(shell.difference(shell.buffer(-shell_thickness)))
            if covered:
                covered_area=shell.difference(open_area)
            report['building_representation']=dict(
                kind='parking_perimeter_shell',wall_thickness_m=shell_thickness,
                source_tags={k:building['tags'][k]for k in ('building','amenity','parking')if k in building['tags']},
                interior='Source aisles retain their physical floors; unknown internal partitions are left unspecified.',
                floor_representation='Source road decks supply each floor and ramp; no footprint-based ground-to-deck infill is constructed.',
                inference='Perimeter thickness and unobserved interior layout are construction assumptions. Source footprint, road identities and explicitly open decks are retained.')
        else:
            report['building_representation']=dict(kind='solid_building_with_source_passages')
        parts=([('overhead',shell.difference(open_area),max(building['base_z_m'],bottom),roof)]
               if roof_structure else
               [('walls',walls,building['base_z_m'],roof),
                ('overhead',covered_area.difference(open_area),bottom,roof)])
        # Keep separate deck surfaces beneath uncovered upper aisles.
        for role, shape, base, roof in parts:
            if roof <= base and building.get('source_parent_support'):
                continue
            for j, polygon in enumerate(_polygons(shape)):
                if polygon.area < 1e-8:
                    continue
                part = deepcopy(building)
                part.update(id=f"{building['id']}:passage:{role}:{j}",
                    exterior=list(map(list, polygon.exterior.coords)),
                    holes=[list(map(list, r.coords)) for r in polygon.interiors],
                    base_z_m=base, roof_z_m=roof, height_m=roof - base,
                    passage_role=role, passage_parent_id=building['id'])
                part['assumptions'] = [s for s in part.get('assumptions', []) if 'no elevated passage' not in s]
                part['assumptions'].append('Source-tagged passage retained; geometry headroom and side margin are recorded modelling choices.')
                result.append(part)
        report.update(status='constructed_pending_native_review', opening_area_m2=opening.area,
                      open_sky_area_m2=open_area.area,
                      overhead_base_z_m=bottom, original_roof_z_m=building['roof_z_m'],
                      boolean_grid_m=grid, opening_boolean_margin_m=2*grid, footprint_change_m2=shapes[i].symmetric_difference(shell).area)
        records.append(report)
    output = dict(scene, buildings=result)
    selected_parents = {r['building_id'] for r in records}
    output['building_passage_source_buildings'] = [b for b in buildings if b['id'] in selected_parents]
    report = dict(records=records, lateral_margin_m=lateral_margin_m, default_clearance_m=default_clearance_m,
        vehicle_body_length_m=vehicle_length_m,vehicle_body_width_m=vehicle_width_m,
        vehicle_turn_reach_m=vehicle_length_m/2,
        vehicle_sweep_scope='Body space along source polylines and intermediate headings over each adjacent half-body-length turn reach, including legal lateral centre positions and the declared side margin. Connected internal movements and short untagged approaches within body reach inherit the passage condition. This is a conservative construction envelope; steering feasibility and actual tracking require native driving.',
        tagged_lanes=len(tagged), changed_buildings=sum(r['status']=='constructed_pending_native_review' for r in records),
        scope='Source-tagged covered or indoor routes, plus explicitly uncovered nonnegative-level roads in parking structures. Source footprint and remaining masses retained; physical headroom and deck geometry remain inferred and require native driving review.')
    output['building_passage_review'] = report
    return output, report


class BuildingVolumes:
    """Query building solids within a supplied vehicle-height envelope."""
    def __init__(self, buildings):
        self.buildings = buildings
        self.shapes = [Polygon(b['exterior'], b['holes']) for b in buildings]
        self.tree = STRtree(self.shapes)

    def obstacles(self, points, width, vehicle_height=2.2):
        points = np.asarray(points)
        line = LineString(points[:, :2])
        envelope = line.buffer(width / 2, cap_style='flat')
        ids = self.tree.query(envelope, predicate='intersects')
        ids = [int(i) for i in ids if self.buildings[i]['base_z_m'] < points[:, 2].max() + vehicle_height
               and self.buildings[i]['roof_z_m'] > points[:, 2].min() - .01]
        station=np.r_[0.,np.linalg.norm(np.diff(points[:,:2],axis=0),axis=1).cumsum()]
        relevant=[]
        for i in ids:
            # Evaluate road height only where its width envelope intersects the building.
            contact=line.intersection(self.shapes[i].buffer(width/2))
            coordinates=get_coordinates(contact)
            if not len(coordinates):continue
            at=np.array([line.project(Point(xy))for xy in coordinates])
            local_z=np.interp(at,station,points[:,2]);building=self.buildings[i]
            if building['base_z_m'] < local_z.max()+vehicle_height and building['roof_z_m'] > local_z.min()-.01:
                relevant.append(i)
        return unary_union([self.shapes[i] for i in relevant])
