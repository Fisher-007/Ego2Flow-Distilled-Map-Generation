"""Query native successor links for Driving, Biking, Sidewalk, and Restricted lanes."""
import xml.etree.ElementTree as ET


def topology_with_non_driving(native, path):
    import carla

    root = ET.parse(path).getroot()
    roads = {int(r.get('id')): r for r in root.findall('road')}
    lengths = {rid: float(r.get('length')) for rid, r in roads.items()}
    junctions = {int(j.get('id')): j for j in root.findall('junction')}
    topology = list(native.get_topology())
    records = []
    for rid, road in roads.items():
        sections = road.findall('lanes/laneSection')
        for index, section in enumerate(sections):
            low = float(section.get('s'))
            high = float(sections[index + 1].get('s')) if index + 1 < len(sections) else lengths[rid]
            for lane in section.findall('left/lane') + section.findall('right/lane'):
                lid = int(lane.get('id'))
                middle = native.get_waypoint_xodr(rid, lid, (low + high) / 2.)
                if middle is None:
                    raise ValueError('Native lane has no interior waypoint: ' + str((rid, lid, index)))
                if middle.lane_type == carla.LaneType.Driving:
                    continue
                forward = lid < 0
                endpoint = high if forward else low
                at_road_end = index == len(sections) - 1 if forward else index == 0
                target_lengths = []
                if at_road_end:
                    link = road.find('link/' + ('successor' if forward else 'predecessor'))
                    if link is not None and link.get('elementType') == 'road':
                        target_lengths.append(lengths[int(link.get('elementId'))])
                    elif link is not None and link.get('elementType') == 'junction':
                        junction = junctions[int(link.get('elementId'))]
                        for connection in junction.findall('connection'):
                            if int(connection.get('incomingRoad')) == rid and any(
                                    int(x.get('from')) == lid for x in connection.findall('laneLink')):
                                target_lengths.append(lengths[int(connection.get('connectingRoad'))])
                else:
                    adjacent = index + 1 if forward else index - 1
                    a = float(sections[adjacent].get('s'))
                    b = float(sections[adjacent + 1].get('s')) if adjacent + 1 < len(sections) else lengths[rid]
                    target_lengths.append(b - a)
                # Stay within the source section and its immediate successor,
                # including the actual sub-metre retained source tails.
                epsilon = min([.001, (high - low) / 4.] + [n / 4. for n in target_lengths])
                start = native.get_waypoint_xodr(rid, lid, endpoint + (-epsilon if forward else epsilon))
                successors = start.next(2. * epsilon)
                for target in successors:
                    if (target.road_id, target.lane_id, target.section_id) == (rid, lid, index):
                        raise ValueError('Native endpoint query stayed within its source lane: ' + str((rid, lid, index)))
                    topology.append((start, target))
                records.append(dict(road_id=rid, lane_id=lid, section_id=index,
                                    lane_type=str(middle.lane_type), probe_distance_m=2. * epsilon,
                                    successors=[[t.road_id, t.lane_id, t.section_id] for t in successors]))
    return topology, dict(method='Stock Driving topology plus actual CARLA next queries at every non-Driving native lane endpoint; source lane types unchanged.',
                          queried_non_driving_lanes=len(records), records=records)
