"""Carry reviewed controls onto a height-revised, otherwise identical road map."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from lxml import etree


def children(path):
    parser = etree.iterparse(str(path), events=('start', 'end'), huge_tree=True,
                             remove_blank_text=True)
    _, root = next(parser)
    for event, element in parser:
        if event == 'end' and element.getparent() is root:
            yield element
            element.clear()
            while element.getprevious() is not None:
                del root[0]


def geometry_without_height_or_controls(road):
    result = deepcopy(road)
    for name in ('elevationProfile', 'signals'):
        for node in result.findall(name):
            result.remove(node)
    return etree.tostring(result, method='c14n')


def junction_without_controls(junction):
    result = deepcopy(junction)
    for node in result.findall('controller'):
        result.remove(node)
    return etree.tostring(result, method='c14n')


def control_reference_review(path):
    """Check the controller-to-junction links needed by CARLA's light groups."""
    controllers, junction_refs, signals, controlled_signals = set(), set(), set(), set()
    for element in children(path):
        if element.tag == 'controller':
            controllers.add(element.get('id'))
            controlled_signals.update(node.get('signalId') for node in element.findall('control'))
        elif element.tag == 'junction':
            junction_refs.update(node.get('id') for node in element.findall('controller'))
        elif element.tag == 'road':
            signals.update(node.get('id') for node in element.findall('signals/signal'))
    missing = sorted(controllers - junction_refs)
    unknown = sorted(junction_refs - controllers)
    missing_signals = sorted(controlled_signals - signals)
    return dict(controllers=len(controllers), junction_controller_ids=len(junction_refs),
        controllers_without_junction=missing, undefined_junction_controllers=unknown,
        undefined_controlled_signals=missing_signals, passed=not (missing or unknown or missing_signals))


def merge(current, controlled, output):
    output.mkdir(parents=True, exist_ok=True)
    extras, controls, junctions, signal_count, references = [], {}, {}, 0, 0
    # Control definitions are small; keep only these while streaming the
    # complete source map. Absolute observed sign positions remain intact.
    for element in children(controlled):
        if element.tag == 'controller':
            extras.append(deepcopy(element))
        if element.tag == 'junction':
            junctions[element.get('id')] = (junction_without_controls(element),
                [deepcopy(node) for node in element.findall('controller')])
        if element.tag == 'road' and element.find('signals') is not None:
            node = element.find('signals')
            controls[element.get('id')] = deepcopy(node)
            signal_count += len(node.findall('signal'))
            references += len(node.findall('signalReference'))
    old_roads = (e for e in children(controlled) if e.tag == 'road')
    count = changed = junction_references = 0
    seen_junctions = set()
    with etree.xmlfile(str(output / 'roads.xodr'), encoding='utf-8') as writer:
        writer.write_declaration()
        with writer.element('OpenDRIVE'):
            for road in children(current):
                if road.tag == 'controller':
                    continue
                if road.tag == 'road':
                    previous = next(old_roads)
                    if geometry_without_height_or_controls(road) != geometry_without_height_or_controls(previous):
                        raise ValueError('Control reuse requires unchanged XY, rules and topology: ' + road.get('id'))
                    a, b = road.find('elevationProfile'), previous.find('elevationProfile')
                    changed += etree.tostring(a, method='c14n') != etree.tostring(b, method='c14n')
                    for node in road.findall('signals'):
                        road.remove(node)
                    if road.get('id') in controls:
                        road.append(controls[road.get('id')])
                    count += 1
                elif road.tag == 'junction':
                    previous, links = junctions[road.get('id')]
                    if junction_without_controls(road) != previous:
                        raise ValueError('Control reuse requires unchanged junction connections: ' + road.get('id'))
                    for node in road.findall('controller'):
                        road.remove(node)
                    road.extend(links)
                    seen_junctions.add(road.get('id'))
                    junction_references += len(links)
                writer.write(road)
            if next(old_roads, None) is not None:
                raise ValueError('Control map has additional roads')
            for node in extras:
                writer.write(node)
    if seen_junctions != set(junctions):
        raise ValueError('Control reuse requires the same junctions')
    native_links = control_reference_review(output / 'roads.xodr')
    if not native_links['passed']:
        raise ValueError('Control reuse left invalid native light-group references: ' + str(native_links))
    review = dict(current=str(current), control_source=str(controlled), roads=count,
        changed_elevation_profiles=changed, signals=signal_count, references=references,
        controllers=len(extras), junction_controller_references=junction_references,
        native_control_links=native_links, identical_XY_topology_and_rules=True,
        scope='Retains revised elevation and previously reviewed control definitions. Native control placement and driving are checked on the assembled map.')
    (output / 'control_reuse_review.json').write_text(json.dumps(review, indent=2) + '\n')
    print(json.dumps(review), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('current', 'controlled', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args()
    merge(a.current, a.controlled, a.output)


if __name__ == '__main__':
    main()
