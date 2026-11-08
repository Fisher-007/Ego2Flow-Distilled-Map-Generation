"""Apply reviewed road-use conditions to imported vehicle permissions."""


def apply_use_conditions(lanes, conditions):
    records=[]
    for lane in lanes:
        if lane.get('source') == 'av2' or lane['id'].startswith('av2:'):
            continue
        tags=lane.get('source_tags',{})
        for condition in conditions:
            if not all(tags.get(k)==v for k,v in condition.get('match_tags',{}).items()):continue
            ways=condition.get('source_way_ids')
            if ways is not None:
                # SUMO retains the source way in each signed/split edge ID.
                # Internal connector IDs are handled by permission propagation.
                way=lane.get('edge_id','').lstrip('-').split('#')[0]
                if way not in ways:continue
            if any(tags.get(k) in values for k,values in condition.get('unless_tags',{}).items()):continue
            original=lane['allowed_classes']
            retained=sorted(set(original)&set(condition['allow']))
            if set(retained)==set(original):continue
            lane['allowed_classes']=retained
            record=dict(lane_id=lane['id'],condition_id=condition['id'],source_tags=tags,
                imported_allowed_classes=original,allowed_classes=retained,
                reason=condition['reason'],evidence=condition['evidence'])
            if ways is not None:record['source_way_id']=way
            if 'judgment' in condition:record['judgment']=condition['judgment']
            lane.setdefault('source_use_review',[]).append(record)
            records.append(record)
    return records


def inherit_internal_permissions(lanes, changed_lane_ids=None, include_exits=False):
    """Propagate reviewed entry and, optionally, exit classes to connectors."""
    by_id={lane['id']:lane for lane in lanes};previous={key:[] for key in by_id}
    for key,lane in by_id.items():
        for target in lane['successors']:
            if target in previous:previous[target].append(key)
    changed={}
    pending=set(by_id if changed_lane_ids is None else changed_lane_ids)
    while pending:
        parent=pending.pop()
        neighbors=by_id[parent]['successors'] + (previous[parent] if include_exits else [])
        for key in neighbors:
            if key not in by_id:continue
            lane=by_id[key]
            if lane.get('source')=='av2' or key.startswith('av2:'):continue
            if not (lane.get('internal') or key.startswith('seam:')) or not previous[key]:continue
            entry=set().union(*(set(by_id[p]['allowed_classes']) for p in previous[key]))
            allowed=set(lane['allowed_classes']) & entry
            exits=[n for n in lane['successors'] if n in by_id]
            if include_exits and exits:
                allowed &= set().union(*(set(by_id[n]['allowed_classes']) for n in exits))
            if allowed==set(lane['allowed_classes']):continue
            changed.setdefault(key,dict(lane_id=key,imported_allowed_classes=lane['allowed_classes'],entry_lane_ids=previous[key]))
            if include_exits:changed[key]['exit_lane_ids']=exits
            lane['allowed_classes']=sorted(allowed);changed[key]['allowed_classes']=lane['allowed_classes'];pending.add(key)
    return list(changed.values())
