"""Use timestamped source light observations for static paired views."""


def phase_for_observation(bindings, source_edge, state, additional_states=None):
    matches = [b for b in bindings if any(
        a['source_lane_id'].rsplit('_', 1)[0] == source_edge for a in b['approaches'])]
    programs = {b['source_signal_id'] for b in matches}
    if len(programs) != 1:
        raise ValueError('Observed approach must identify one source control program: '+source_edge)
    program = programs.pop()
    additional_states = additional_states or {}
    program_bindings = {b['signal_id']: b for b in bindings if b['source_signal_id'] == program}
    if set(additional_states) - set(program_bindings):
        raise ValueError('Simultaneous observed signal belongs to another source program')
    phases = [i for i in range(len(matches[0]['phase_states']))
              if all(b['phase_states'][i] == state for b in matches)
              and all(program_bindings[key]['phase_states'][i] == value
                      for key, value in additional_states.items())]
    if not phases:
        raise ValueError('Source program has no phase matching observed state: '+source_edge)
    return program, phases[0], sorted({b['signal_id'] for b in matches} | set(additional_states))


class ObservedSignalSnapshot:
    def __init__(self, world, manifest, evidence):
        self.world = world
        self.bindings = manifest['bindings']
        self.evidence = evidence['areas'] if 'areas' in evidence else [
            row['native_construction']['settings'] for row in evidence['native_objects']
            if row['choice'] in ('retain', 'enhance')
            and row.get('native_construction', {}).get('kind') == 'paired_signal_snapshot']
        self.actors = {a.get_opendrive_id(): a for a in world.get_actors().filter('traffic.traffic_light*')}
        self.previous = []
        self.current_states = {}

    def apply(self, log_id, camera, timestamp_ns):
        import carla
        sources = [row for row in self.evidence
                   if row['log_id'] == log_id and row['camera'] == camera
                   and any(o['timestamp_ns'] == timestamp_ns for o in row['observations'])]
        if not sources:
            return dict(observed_state_applied=False, reason='No reviewed light state at this exact camera timestamp.')
        if len(sources) != 1:
            raise ValueError('Combine simultaneous source observations before rendering')
        source = sources[0]
        explicit = source.get('explicit_observed_signal_states')
        if explicit is not None:
            by_id = {b['signal_id']: b for b in self.bindings}
            if not explicit or set(explicit) - set(by_id):
                raise ValueError('Explicit source snapshot needs known native signal identities')
            programs = {by_id[key]['source_signal_id'] for key in explicit}
            if len(programs) != 1:
                raise ValueError('Combine explicit observations within one source control program')
            if source.get('unobserved_companion_policy') != 'red' or not source.get('snapshot_inference_reason'):
                raise ValueError('Declare the red companion-state inference for an explicit source snapshot')
            if set(explicit.values()) - {'Red', 'Yellow', 'Green', 'Off'}:
                raise ValueError('Explicit source snapshot contains an unsupported lamp state')
            program, phase, observed_ids = programs.pop(), None, sorted(explicit)
            method = 'explicit_observed_identities_with_unobserved_red'
            scope = ('Static snapshot at the reviewed source timestamp. Observed states apply to associated native heads; '
                     'unobserved companion heads use an inferred Red state. Prior actor states are restored on close.')
        else:
            program, phase, observed_ids = phase_for_observation(
                self.bindings, source['controlled_source_edge'], source['observed_state'],
                source.get('additional_observed_signal_states'))
            method = 'compatible_inferred_phase'
            scope = ('Exact reviewed source timestamp. Only the explicitly associated observed heads are checked against source states. '
                     'Other heads use a compatible inferred phase; unresolved visible observations are listed separately. '
                     'Real phase timing is unknown. Static paired view only.')
        selected = [b for b in self.bindings if b['source_signal_id'] == program]
        affected = {a.id: a for b in selected for a in self.actors[b['signal_id']].get_group_traffic_lights()}
        self.previous = [(a, a.get_state(), a.is_frozen()) for a in affected.values()]
        for actor in affected.values():
            actor.freeze(True)
        states = []
        for binding in selected:
            state = explicit.get(binding['signal_id'], 'Red') if explicit is not None else binding['phase_states'][phase]
            self.actors[binding['signal_id']].set_state(getattr(carla.TrafficLightState, state))
            states.append(dict(signal_id=binding['signal_id'], state=state,
                               observed=binding['signal_id'] in observed_ids))
            self.current_states[binding['signal_id']] = state
        return dict(observed_state_applied=True, camera=camera, timestamp_ns=timestamp_ns,
                    source_edge=source.get('controlled_source_edge'), observed_state=source.get('observed_state'),
                    source_program=program, compatible_inferred_phase=phase, states=states,
                    state_selection_method=method, snapshot_inference_reason=source.get('snapshot_inference_reason'),
                    observed_native_signal_states={key: self.current_states[key] for key in observed_ids},
                    unmapped_visible_signal_observations=source.get('unmapped_visible_signal_observations', []),
                    scope=scope)

    def check(self):
        actual = {key: str(self.actors[key].get_state()).rsplit('.', 1)[-1] for key in self.current_states}
        return dict(actual_states=actual, matched=bool(actual) and actual == self.current_states)

    def close(self):
        for actor, state, frozen in self.previous:
            actor.set_state(state)
            actor.freeze(frozen)
        self.previous = []
        self.current_states = {}


class ObservedSignalReplay:
    """Apply timestamped signal observations over declared replay intervals."""
    def __init__(self, world, manifest, conditions, log_id, camera):
        self.world = world
        self.manifest = manifest
        self.intervals = [r for r in conditions['intervals']
                          if r['log_id'] == log_id and r['camera'] == camera]
        for row in self.intervals:
            if row['start_timestamp_ns'] > row['end_timestamp_ns']:
                raise ValueError('Signal replay interval has reversed times')
            if not row['observations'] or not row['inference_reason']:
                raise ValueError('Signal replay interval needs observations and explicit inference')
        self.log_id, self.camera = log_id, camera
        self.active = None
        self.snapshot = None
        self.selection = None

    def apply(self, timestamp_ns):
        rows = [r for r in self.intervals
                if r['start_timestamp_ns'] <= timestamp_ns <= r['end_timestamp_ns']]
        if len(rows) > 1:
            raise ValueError('Combine simultaneous source states in one replay interval')
        row = rows[0] if rows else None
        if row is not self.active:
            self.close()
            if row is not None:
                evidence = dict(row, observations=[dict(timestamp_ns=timestamp_ns)])
                self.snapshot = ObservedSignalSnapshot(self.world, self.manifest,
                                                       {'areas': [evidence]})
                self.selection = self.snapshot.apply(self.log_id, self.camera, timestamp_ns)
                self.active = row
        if row is None:
            return dict(applied=False, reason='No selected source-supported interval at this frame.')
        observed = any(o['timestamp_ns'] == timestamp_ns for o in row['observations'])
        result = dict(self.selection)
        result.update(applied=True, timestamp_ns=timestamp_ns,
                      evidence_kind='reviewed_sample' if observed else 'inferred_interval',
                      observations=row['observations'],
                      inference_reason=row['inference_reason'],
                      interval_id=row['id'],
                      scope='Finite source-supported replay interval. Only listed sample timestamps are directly observed; other frames and compatible companion states are inferred. Historical phase timing remains unknown.')
        return result

    def check(self):
        return self.snapshot.check() if self.snapshot else None

    def close(self):
        if self.snapshot is not None:
            self.snapshot.close()
        self.active = None
        self.snapshot = None
        self.selection = None
