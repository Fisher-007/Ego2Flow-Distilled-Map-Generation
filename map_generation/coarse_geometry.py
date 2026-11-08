"""Fit shared endpoints on inferred roads to preserve source movement direction."""
import numpy as np
from scipy.sparse import coo_matrix, eye
import osqp


def _direction(points, at_end):
    delta = np.diff(points[:, :2], axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    valid = np.flatnonzero(lengths > 1e-5)
    if not len(valid):
        return None
    index = valid[-1 if at_end else 0]
    return delta[index] / lengths[index]


def repair(coarse, movement_limit=3., minimum_span=.1):
    lanes = coarse['lanes']
    lookup = {lane['id']: i for i, lane in enumerate(lanes)}
    shapes = [np.asarray(lane['centerline'], dtype=float) for lane in lanes]
    ends = np.asarray([p for shape in shapes for p in (shape[0, :2], shape[-1, :2])])
    parent = np.arange(len(ends))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return int(i)

    def union(a, b):
        parent[find(b)] = find(a)

    for i, lane in enumerate(lanes):
        for target in lane['successors']:
            j = lookup[target]
            if np.linalg.norm(ends[2*i+1] - ends[2*j]) < .02:
                union(2*i+1, 2*j)
    roots = [find(i) for i in range(len(ends))]
    groups = {r: i for i, r in enumerate(sorted(set(roots)))}
    vertices = np.asarray([groups[r] for r in roots])
    count = len(groups)
    base = np.column_stack([np.bincount(vertices, weights=ends[:, axis]) /
                            np.bincount(vertices) for axis in (0, 1)])
    constraints = []
    straight = {}
    for connection in coarse['connections']:
        if connection['direction'] != 's':
            continue
        a, b, middle = (lookup[connection[key]] for key in ('from_lane', 'to_lane', 'next_lane'))
        if lanes[a]['internal'] or lanes[b]['internal'] or not lanes[middle]['internal']:
            continue
        da, db = _direction(shapes[a], True), _direction(shapes[b], False)
        if da is None or db is None or da @ db < .8:
            continue
        direction = (da + db) / np.linalg.norm(da + db)
        straight[middle] = direction
    for i, lane in enumerate(lanes):
        if i in straight:
            direction, span = straight[i], minimum_span
        elif not lane['internal']:
            chord = shapes[i][-1, :2] - shapes[i][0, :2]
            length = np.linalg.norm(chord)
            if length < .01:
                continue
            direction, span = chord / length, min(minimum_span, length / 2)
        else:
            continue
        a, b = vertices[2*i:2*i+2]
        original = float((base[b] - base[a]) @ direction)
        constraints.append((i, int(a), int(b), direction, span, original))
    row, col, val, lower = [], [], [], []
    for i, a, b, direction, span, original in constraints:
        r = len(lower)
        for axis in (0, 1):
            row.extend([r, r]); col.extend([2*a+axis, 2*b+axis])
            val.extend([-direction[axis], direction[axis]])
        lower.append(span-original)
    size = 2 * count
    n = len(lower)
    row.extend(range(n, n+size)); col.extend(range(size)); val.extend(np.ones(size))
    matrix = coo_matrix((val, (row, col)), shape=(n+size, size)).tocsc()
    solver = osqp.OSQP()
    solver.setup(P=eye(size, format='csc'), q=np.zeros(size), A=matrix,
                 l=np.r_[lower, np.full(size, -movement_limit)],
                 u=np.r_[np.full(n, np.inf), np.full(size, movement_limit)],
                 eps_abs=1e-5, eps_rel=1e-5, max_iter=20000, verbose=False, polishing=True)
    solution = solver.solve()
    report = dict(method=__doc__, solver_status=solution.info.status,
                  constrained_straight_lanes=len(straight), constraints=n,
                  original_violations=int(sum(original < span-1e-6 for _, _, _, _, span, original in constraints)),
                  movement_limit_per_axis_m=movement_limit, changed_lanes=[], remaining=[])
    if solution.info.status_val != 1:
        certificate=np.asarray(solution.prim_inf_cert[:n])
        selected=np.argsort(abs(certificate))[-40:][::-1]
        report['conflicting_constraints']=[dict(lane_id=lanes[constraints[r][0]]['id'],
            certificate=float(certificate[r]),start_group=constraints[r][1],end_group=constraints[r][2],
            required_span_m=float(constraints[r][4]),original_span_m=float(constraints[r][5]),
            points=shapes[constraints[r][0]].tolist()) for r in selected if abs(certificate[r])>1e-8]
        return report
    corrected = base + solution.x.reshape(-1, 2)
    for i, a, b, direction, span, _ in constraints:
        projection = float((corrected[b] - corrected[a]) @ direction)
        if projection < span-1e-4:
            report['remaining'].append(dict(lane_id=lanes[i]['id'], projection_m=projection))
    for i, lane in enumerate(lanes):
        a, b = vertices[2*i:2*i+2]
        shape = shapes[i].copy()
        station = np.r_[0., np.linalg.norm(np.diff(shape[:, :2], axis=0), axis=1).cumsum()]
        u = station / station[-1] if station[-1] > 1e-6 else np.linspace(0., 1., len(shape))
        delta_a, delta_b = corrected[a]-shape[0, :2], corrected[b]-shape[-1, :2]
        shape[:, :2] += (1-u[:, None])*delta_a + u[:, None]*delta_b
        movement = float(max(np.linalg.norm(delta_a), np.linalg.norm(delta_b)))
        if i in straight and (np.diff(shape[:, :2], axis=0) @ straight[i] < -1e-5).any():
            shape[:, :2] = (1-u[:, None])*corrected[a] + u[:, None]*corrected[b]
        if movement > 1e-4 or not np.allclose(shape, shapes[i], atol=1e-5, rtol=0):
            lane['centerline'] = shape.tolist()
            lane['coarse_geometry_repair'] = dict(method='shared_endpoint_forward_order', max_endpoint_movement_m=movement)
            report['changed_lanes'].append(dict(lane_id=lane['id'], max_endpoint_movement_m=movement))
    report['max_endpoint_movement_m'] = max((v['max_endpoint_movement_m'] for v in report['changed_lanes']), default=0.)
    coarse['geometry_repair'] = {key: value for key, value in report.items() if key != 'changed_lanes'}
    return report
