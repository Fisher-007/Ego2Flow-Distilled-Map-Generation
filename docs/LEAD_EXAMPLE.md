# LEAD: dynamic-policy integration

LEAD can serve as the learned driving policy used during map construction. The integration point is `run_policy` in [learned_policy.py](../examples/adapters/learned_policy.py). The template defines the command interface; implement model loading, sensors, and control using the installed LEAD implementation.

## Configuration

1. Install LEAD and its weights in a separate environment.
2. Copy `learned_policy.py` to the adapter directory and implement `run_policy`.
3. Copy [model.example.json](../examples/lead/model.example.json) and set the model, sensor, navigation, and controller paths.
4. Copy the contents of [adapter.example.json](../examples/lead/adapter.example.json) into the build config's `dynamic_policy`, using the adapter's executable, script, working directory, and configuration file.
5. Document the traffic runner's `routes` and `scenario` formats and reference that document in the build config's `instructions`.

The model example specifies a 20 Hz simulation, a learned-policy vehicle, and BasicAgent background traffic. Adapt these settings to the model and traffic runner. `policy.name` in the result matches the configured name, such as `LEAD`.

## Implement `run_policy`

| Step | Implementation |
| --- | --- |
| Connect | Read the runtime manifest, routes, and scenario; connect to `resources.port` and check `map_asset`. |
| Load | Load the model, checkpoint, and preprocessing; create separate navigation and history state for each controlled vehicle. |
| Configure sensors | Create the sensors required by the model and record their calibration, resolution, rate, and frame alignment. |
| Set navigation | Convert source-supported routes to the model's navigation format and align GNSS or local pose inputs with the map. |
| Control | Advance the synchronous world, collect matching sensor frames, run inference, and apply throttle, steering, and braking through the model's controller. |
| Generate traffic | Admit requests throughout the observation window and track waiting, completion, active vehicles, and unfinished removals. |
| Record | Save trajectories, controls, sensor views, collisions, and rule events; identify learned-policy vehicles separately from background traffic. |
| Clean up | Destroy objects created by the adapter and restore changed world settings in `finally`. Leave the framework's simulator running. |

Use the model's declared sensor and navigation inputs. Record any additional intervention in the control log. Return the fields in the [dynamic-policy result schema](../configs/workflows/instructions/dynamic_policy.md#result), including driving failures and coverage limitations.

## Run

After implementing the adapter:

```bash
python build_map.py doctor --config configs/build.local.json
python build_map.py run --config configs/build.local.json
```

The dynamic reviewer prepares a request, invokes the adapter, and evaluates its result. Execution files include `request.json`, `execution.json`, `adapter_result.json`, and the validated `result.json`. The workflow registers the latter as `dynamic_policy_results`.
