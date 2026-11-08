"""Persistent construction workflow with mailbox and HTTP agent providers."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import time
from datetime import datetime, timezone


STAGES = (
    "source_selection", "coarse", "fusion", "information", "structure",
    "appearance", "compile", "static_review", "dynamic_review", "delivery",
)
REVIEW_STAGES = {"static_review", "dynamic_review"}
ROLE_INSTRUCTIONS = {
    "decision": ("construction.md",),
    "static_review": ("review.md",),
    "dynamic_review": ("review.md", "dynamic_policy.md"),
}
REQUIRED_ARTIFACTS = {
    "source_selection": {"city_config", "source_catalog", "area_selection"},
    "coarse": {"coarse_scene"},
    "fusion": {"fused_scene", "source_review"},
    "information": {"information_graph", "generation_conditions"},
    "structure": {"structure_report", "road_scene"},
    "appearance": {"appearance_report", "asset_manifest"},
    "compile": {"native_build", "runtime_manifest"},
    "static_review": {"static_review", "native_views"},
    "dynamic_review": {"dynamic_review", "driving_results", "traffic_results", "paired_results", "dynamic_policy_results"},
    "delivery": {"delivery_manifest", "reproduction_manifest"},
}

def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def action_schema():
    string = {"type": "string"}
    return {
        "type": "object", "additionalProperties": False,
        "properties": {
            "kind": {"type": "string", "enum": ["tool", "advance", "feedback", "blocked"]},
            "reason": string, "tool": string, "arguments_json": string,
            "artifacts": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "properties": {"name": string, "path": string}, "required": ["name", "path"]}},
            "target_stage": string,
            "findings": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "properties": {"id": string, "description": string, "required_change": string,
                               "evidence": {"type": "array", "items": string}},
                "required": ["id", "description", "required_change", "evidence"]}},
            "summary": string,
        },
        "required": ["kind", "reason", "tool", "arguments_json", "artifacts", "target_stage", "findings", "summary"],
    }


class CityWorkflow:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.state_path = self.directory / "state.json"
        self.state = read_json(self.state_path)
        self.config = self.state["config"]
        self.workspace = Path(self.config["workspace"]).resolve()
        self._workers = {}
        # The jobs journal also survives a runner exit between launch and state save.
        self.state["jobs"] = sorted({*self.state["jobs"], *(p.name for p in (self.directory / "jobs").glob("job-*"))})

    @classmethod
    def create(cls, directory, config):
        directory = Path(directory).resolve()
        if (directory / "state.json").exists():
            raise ValueError("A workflow already exists here; use resume.")
        workspace = Path(config["workspace"]).resolve()
        if not workspace.is_dir():
            raise ValueError(f"Missing workspace: {workspace}")
        if not config.get("resources"):
            raise ValueError("Supply the allocated GPU/CPU resource pairs.")
        config = dict(config, workspace=str(workspace))
        write_json(directory / "state.json", {
            "version": 1, "created_at": utc_now(), "config": config,
            "stage": STAGES[0], "status": "running", "revision": 1,
            "sequence": 0, "pending_request": None, "artifacts": {},
            "completed_stages": [], "feedback": [], "recent": [], "jobs": [],
        })
        return cls(directory)

    def save(self):
        write_json(self.state_path, self.state)

    def role(self):
        stage = self.state["stage"]
        return stage if stage in REVIEW_STAGES else "decision"

    def stages(self):
        return STAGES

    def required_artifacts(self, stage):
        return set(REQUIRED_ARTIFACTS[stage])

    def agent_settings(self, role=None):
        """Role overrides are optional; each HTTP call already has fresh messages."""
        settings = dict(self.config["agent"])
        roles = settings.pop("roles", {})
        settings.update(roles.get(role or self.role(), {}))
        return settings

    def configure_agent(self, settings):
        """Replace transport configuration without changing an in-flight request."""
        for role in ("decision", *sorted(REVIEW_STAGES)):
            resolved = {**settings, **settings.get("roles", {}).get(role, {})}
            if resolved.get("provider") not in {"mailbox", "http"} or not resolved.get("model"):
                raise ValueError(f"Agent role {role} needs provider (mailbox/http) and model.")
            if resolved["provider"] == "http" and not resolved.get("base_url"):
                raise ValueError(f"HTTP Agent role {role} needs base_url.")
        self.config["agent"] = settings
        self.event("agent_configuration", {"settings": settings,
            "pending_request_preserved": self.state.get("pending_request")})

    def job_index(self):
        jobs = []
        for identity in self.state["jobs"]:
            folder = self.directory / "jobs" / identity
            result = folder / "result.json"
            if result.exists() and identity not in self.state["jobs"][-8:]:
                continue
            spec = read_json(folder / "spec.json")
            jobs.append({"job_id": identity, "resource": spec["resource"],
                         "service": spec.get("service", False),
                         "purpose": spec.get("purpose", ""), "spec": str(folder / "spec.json"),
                         "result": str(result) if result.exists() else None,
                         "log": str(folder / "output.log"),
                         "status": "finished" if result.exists() else "running"})
        return jobs

    def event(self, kind, content):
        item = {"time": utc_now(), "stage": self.state["stage"],
                "revision": self.state["revision"], "role": self.role(), "kind": kind, **content}
        with (self.directory / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        self.state["recent"].append(item)
        self.state["recent"] = self.state["recent"][-12:]
        self.state.setdefault("role_recent", {}).setdefault(self.role(), []).append(item)
        self.state["role_recent"][self.role()] = self.state["role_recent"][self.role()][-12:]

    def path(self, value):
        path = Path(value)
        return path.resolve() if path.is_absolute() else (self.workspace / path).resolve()

    def reopen(self, instruction):
        """Record post-delivery input for the decision Agent at the current stage."""
        if self.state["status"] != "complete":
            raise ValueError("reopen requires a completed workflow; use respond or note for an active run.")
        snapshot = self.directory / "completions" / f"completion-{self.state['sequence']:05d}.json"
        write_json(snapshot, self.state)
        self.state.setdefault("external_instructions", []).append(instruction)
        self.event("workflow_reopened", {"previous_status": "complete",
                                        "completion_snapshot": str(snapshot), **instruction})
        # Advance the request sequence when reopening a completed run.
        self.state.update(status="running", pending_request=None)

    def request(self):
        from map_generation.agent_provider import AgentRequest
        pending = self.state.get("pending_request")
        if pending:
            stored = read_json(self.directory / "requests" / (pending + ".json"))
            return AgentRequest(**stored)
        number = self.state["sequence"] + 1
        identity = f"{self.config['run_id']}-{number:05d}"
        stage = self.state["stage"]
        role = self.role()
        setting = self.agent_settings(role)
        workflow_root = Path("configs/workflows")
        instruction_files = [
            workflow_root / "prompts/common.md",
            workflow_root / "instructions/method.md",
            *(workflow_root / "instructions" / name for name in ROLE_INSTRUCTIONS[role]),
            workflow_root / "prompts" / f"{role}.md",
        ]
        instructions = "\n\n".join((self.workspace / path).read_text() for path in instruction_files)
        packet = {
            "task": self.config["task"], "run_id": self.config["run_id"],
            "run_directory": str(self.directory),
            "role": role, "stage": stage, "revision": self.state["revision"],
            "role_session": f"{self.config['run_id']}/{role}",
            "environment": self.config,
            "required_artifacts_to_advance": sorted(self.required_artifacts(stage)),
            "enabled_stages": self.stages(),
            "artifacts": self.state["artifacts"], "completed_stages": self.state["completed_stages"],
            "feedback": self.state["feedback"], "recent_events": self.state.get("role_recent", {}).get(role, []),
            "prior_revision_manifests": [str(p) for p in sorted((self.directory / "revisions").glob("*.json"))],
            "external_instructions": self.state.get("external_instructions", []),
            "tool_reference": TOOL_REFERENCE,
            "instruction_files": [str(path) for path in instruction_files],
            "recipe": "configs/workflows/instructions/construction.md",
            "tool_catalog": "configs/workflows/city_tools.json",
            "full_event_log": str(self.directory / "events.jsonl"),
            "jobs": self.job_index(),
            "job_directory": str(self.directory / "jobs"),
        }
        content = [{"type": "text", "text": json.dumps(packet, ensure_ascii=False)}]
        # Image requests originate in the tool loop and are delivered to both providers.
        for item in self.state.pop("next_images", []):
            content.append({"type": "image_file", "path": item, "detail": "high"})
        messages = [{"role": "system", "content": instructions}, {"role": "user", "content": content}]
        params = {"request_id": identity, "model": setting["model"],
                  "messages": messages, "response_schema": action_schema(),
                  "parameters": setting.get("parameters", {}),
                  "response_mode": setting.get("response_mode", "json_schema")}
        write_json(self.directory / "requests" / (identity + ".json"), params)
        self.state.update(sequence=number, pending_request=identity)
        self.save()
        return AgentRequest(**params)

    def apply(self, action, request_id):
        if request_id != self.state["pending_request"]:
            raise ValueError("Response is for a different workflow request.")
        write_json(self.directory / "actions" / (request_id + ".json"), action)
        self.event("agent_action", {"request_id": request_id, "action": action})
        try:
            kind = action["kind"]
            if not action.get("reason"):
                raise ValueError("Supply the evidence-based reason for this action.")
            if kind == "tool":
                result = self.tool(action["tool"], json.loads(action["arguments_json"] or "{}"))
            elif kind == "advance":
                result = self.advance(action)
            elif kind == "feedback":
                result = self.feedback(action)
            elif kind == "blocked":
                self.state["status"] = "blocked"
                result = {"blocking_question": action["summary"], "findings": action["findings"]}
            else:
                raise ValueError(f"Unknown action kind: {kind}")
            self.event("action_result", {"request_id": request_id, "result": result})
        except (ValueError, OSError, KeyError, TypeError) as error:
            # Return tool failures to the agent as feedback.
            self.event("action_error", {"request_id": request_id, "error": str(error),
                                        "error_type": type(error).__name__})
        self.state["pending_request"] = None
        self.save()

    def advance(self, action):
        stage = self.state["stage"]
        supplied = {a["name"]: str(self.path(a["path"])) for a in action["artifacts"]}
        missing = self.required_artifacts(stage) - supplied.keys()
        if missing:
            raise ValueError(f"Stage {stage} still needs artifacts: {sorted(missing)}")
        for name, value in supplied.items():
            self.require_artifact(value)
        if any(not (self.directory / "jobs" / j / "result.json").exists()
               and not read_json(self.directory / "jobs" / j / "spec.json").get("service") for j in self.state["jobs"]):
            raise ValueError("Observe outstanding tool jobs before advancing the stage.")
        if stage in REVIEW_STAGES:
            report = read_json(supplied[stage])
            if report.get("status") != "pass" or not report.get("evidence"):
                raise ValueError("Review needs a pass report with actual evidence; otherwise send feedback.")
            if report.get("revision") != self.state["revision"]:
                raise ValueError("Review must refer to the current construction revision.")
            for evidence in report["evidence"]:
                self.require_artifact(str(self.path(evidence)))
            if stage == "dynamic_review":
                from map_testing.dynamic import validate_workflow_artifact
                validate_workflow_artifact(supplied['dynamic_policy_results'], self.config, self.state['revision'])
            self.state.setdefault("review_requests", {})[stage] = {
                "request_id": self.state["pending_request"], "role": self.role(),
                "revision": self.state["revision"], "report": supplied[stage]}
        if stage == "delivery":
            enabled_reviews = REVIEW_STAGES.intersection(self.stages())
            if not enabled_reviews.issubset(self.state["completed_stages"]):
                raise ValueError("Both native reviews must finish on the current revision.")
        for name, path in supplied.items():
            self.state["artifacts"][name] = {"path": path, "stage": stage, "revision": self.state["revision"]}
        self.state["completed_stages"].append(stage)
        if stage == "delivery":
            self.state["status"] = "complete"
        else:
            stages = self.stages()
            self.state["stage"] = stages[stages.index(stage) + 1]
        return {"completed_stage": stage, "next_stage": self.state["stage"], "status": self.state["status"]}

    @staticmethod
    def require_artifact(value):
        path = Path(value)
        if not path.exists() or (path.is_file() and path.stat().st_size == 0) or (path.is_dir() and next(path.iterdir(), None) is None):
            raise ValueError(f"Artifact does not exist or is empty: {value}")

    def feedback(self, action):
        target = action["target_stage"]
        if target not in STAGES or STAGES.index(target) > STAGES.index(self.state["stage"]):
            raise ValueError("Feedback must return to the affected current or earlier construction stage.")
        if not action["findings"] or any(not f.get("evidence") for f in action["findings"]):
            raise ValueError("Feedback needs the observed problem and its evidence references.")
        self.state["feedback"].append({"from_stage": self.state["stage"], "target_stage": target,
                                       "revision": self.state["revision"], "findings": action["findings"]})
        write_json(self.directory / "revisions" / f"revision-{self.state['revision']:04d}.json",
                   {"revision": self.state["revision"], "artifacts": self.state["artifacts"],
                    "completed_stages": self.state["completed_stages"], "feedback": action["findings"]})
        invalid = set(STAGES[STAGES.index(target):])
        self.state["artifacts"] = {k: v for k, v in self.state["artifacts"].items() if v["stage"] not in invalid}
        self.state["completed_stages"] = [s for s in self.state["completed_stages"] if s not in invalid]
        self.state["review_requests"] = {k: v for k, v in self.state.get("review_requests", {}).items()
                                         if k not in invalid}
        self.state["revision"] += 1
        self.state["stage"] = target
        return {"repair_stage": target, "revision": self.state["revision"],
                "invalidated_stages": sorted(invalid), "previous_files_preserved": True}

    def tool(self, name, args):
        if name == "read_tail":
            path = self.path(args["path"])
            count = max(0, min(int(args.get("bytes", 14000)), 24000))
            with path.open("rb") as stream:
                end = stream.seek(0, os.SEEK_END)
                start = max(0, end - count)
                stream.seek(start)
                data = stream.read(count)
            return {"path": str(path), "start_byte": start, "bytes_read": len(data),
                    "text": data.decode(errors="replace"), "truncated": start > 0}
        if name == "read":
            path = self.path(args["path"])
            start, count = int(args.get("start_line", 1)), min(int(args.get("lines", 180)), 500)
            with path.open(errors="replace") as stream:
                import itertools
                text = "".join(itertools.islice(stream, max(0, start - 1), max(0, start - 1) + count))
            return {"path": str(path), "start_line": start, "text": text[:24000], "truncated": len(text) > 24000}
        if name == "list":
            path = self.path(args.get("path", "."))
            import itertools
            files = list(itertools.islice(path.glob(args.get("pattern", "*")), 201))
            return {"paths": [str(f) for f in files[:200]], "truncated": len(files) > 200}
        if name == "write":
            path = self.path(args["path"])
            if self.state["stage"] in REVIEW_STAGES and not path.is_relative_to(self.directory):
                raise ValueError("Review Agents write findings in the workflow directory; construction changes go through feedback.")
            path.parent.mkdir(parents=True, exist_ok=True)
            edit = self.directory / "edits" / f"{self.state['sequence']:05d}"
            edit.mkdir(parents=True, exist_ok=True)
            if path.exists():
                (edit / "before.txt").write_text(path.read_text())
            (edit / "after.txt").write_text(args["text"])
            write_json(edit / "target.json", {"path": str(path), "stage": self.state["stage"]})
            path.write_text(args["text"])
            return {"written": str(path), "bytes": path.stat().st_size, "edit_record": str(edit)}
        if name == "image":
            path = self.path(args["path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            self.state.setdefault("next_images", []).append(str(path))
            return {"image_attached_to_next_request": str(path)}
        if name == "run":
            return self.start_job(args)
        if name == "poll":
            job = self.directory / "jobs" / args["job_id"]
            if not job.is_dir():
                raise FileNotFoundError(job)
            until = time.monotonic() + min(float(args.get("wait_seconds", 0)), 30)
            result = job / "result.json"
            while not result.exists() and time.monotonic() < until:
                time.sleep(.2)
            log = job / "output.log"
            tail = ""
            if log.exists():
                with log.open("rb") as stream:
                    stream.seek(max(0, log.stat().st_size - 14000))
                    tail = stream.read().decode(errors="replace")
            return {"job_id": args["job_id"], "status": "finished" if result.exists() else "running",
                    "result": read_json(result) if result.exists() else None, "log_tail": tail}
        if name == "stop":
            identity = args["job_id"]
            if identity not in self.state["jobs"]:
                raise ValueError("Only a job owned by this workflow can be stopped.")
            folder = self.directory / "jobs" / identity
            if (folder / "result.json").exists():
                return read_json(folder / "result.json")
            pid = read_json(folder / "worker.json")["pid"]
            try:
                os.killpg(pid, signal.SIGTERM)
                time.sleep(.5)
                # Terminate remaining processes in the service's process group.
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process = self._workers.pop(identity, None)
            if process:
                process.wait(timeout=10)
            result = dict(exit_code=None, stopped_by_agent=True, reason=args["reason"], finished_at=utc_now())
            write_json(folder / "result.json", result)
            return result
        raise ValueError(f"Unknown tool: {name}")

    def start_job(self, args):
        identity = f"job-{self.state['sequence']:05d}"
        folder = self.directory / "jobs" / identity
        if folder.exists():
            return {"job_id": identity, "resumed_existing_execution": True,
                    "spec": str(folder / "spec.json"), "log": str(folder / "output.log")}
        slot = args.get("resource", next(iter(self.config["resources"])))
        resource = self.config["resources"][slot]
        cpu_only = bool(args.get("cpu_only", False))
        if cpu_only and args.get("service"):
            raise ValueError("CPU-only helpers cannot also be service jobs.")
        for existing_identity in self.state["jobs"]:
            folder = self.directory / "jobs" / existing_identity
            existing = read_json(folder / "spec.json")
            if not (folder / "result.json").exists() and existing["resource"] == slot:
                if cpu_only or existing.get("cpu_only"):
                    if cpu_only and existing.get("cpu_only"):
                        raise ValueError(f"Resource {slot} has CPU-only helper {existing_identity}; poll/stop it before another helper.")
                    continue
                if existing.get("service") and not args.get("service"):
                    continue
                raise ValueError(f"Resource {slot} has job {existing_identity}; poll/stop it before another computation or service.")
        argv = args["argv"]
        if not isinstance(argv, list) or not argv or not all(isinstance(v, str) for v in argv):
            raise ValueError("run.argv is a nonempty string array.")
        folder = self.directory / "jobs" / identity
        folder.mkdir(parents=True, exist_ok=False)
        environment = {**self.config.get("environment", {}), **args.get("env", {})}
        # Apply the configured CPU/GPU allocation to the worker.
        environment.update(MAP_CPUSET=resource["cpus"], MAP_GPU=str(resource["gpu"]),
                           CUDA_VISIBLE_DEVICES=str(resource["gpu"]),
                           MAP_RPC_PORT=str(resource["port"]), OPENBLAS_NUM_THREADS="1",
                           OMP_NUM_THREADS=str(resource.get("threads", 2)))
        if cpu_only:
            environment.update(MAP_GPU="", CUDA_VISIBLE_DEVICES="",
                               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
                               BLIS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
        spec = {"argv": ["taskset", "-c", resource["cpus"], *argv],
                "cwd": str(self.path(args.get("cwd", "."))), "env": environment,
                "resource": slot, "service": bool(args.get("service", False)), "cpu_only": cpu_only,
                "purpose": args.get("purpose", ""), "created_at": utc_now()}
        write_json(folder / "spec.json", spec)
        with (folder / "worker.log").open("w") as log:
            process = subprocess.Popen([sys.executable, "-m", "scripts.workflow.run", "worker", "--job", str(folder)],
                                       cwd=self.workspace, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(folder / "worker.json", {"pid": process.pid})
        self._workers[identity] = process
        self.state["jobs"].append(identity)
        return {"job_id": identity, "worker_pid": process.pid, "spec": str(folder / "spec.json"),
                "log": str(folder / "output.log")}


TOOL_REFERENCE = {
    "read": {"path": "repo-relative or absolute", "start_line": 1, "lines": 180},
    "read_tail": {"path": "existing log file", "bytes": 14000,
                  "description": "Read at most 24000 final bytes without scanning the prefix or starting a helper; may begin within a line."},
    "list": {"path": ".", "pattern": "*"},
    "write": {"path": "file", "text": "complete content; changes recorded"},
    "image": {"path": "image file; delivered as image input next turn"},
    "run": {"argv": ["executable", "arguments"], "cwd": ".", "resource": "configured slot",
            "env": {}, "service": False, "cpu_only": False,
            "purpose": "which construction or measurement this performs"},
    "poll": {"job_id": "job-00001", "wait_seconds": 20},
    "stop": {"job_id": "job-00001", "reason": "stop this workflow's server or unsuccessful job"},
}


def run_worker(directory):
    directory = Path(directory)
    spec = read_json(directory / "spec.json")
    started = time.monotonic()
    try:
        with (directory / "output.log").open("w") as stream:
            process = subprocess.Popen(spec["argv"], cwd=spec["cwd"],
                                       env={**os.environ, **spec["env"]}, stdout=stream, stderr=subprocess.STDOUT)
            write_json(directory / "process.json", {"pid": process.pid})
            code = process.wait()
        result = {"exit_code": code, "elapsed_seconds": time.monotonic() - started, "finished_at": utc_now()}
    except OSError as error:
        result = {"exit_code": None, "error": str(error), "elapsed_seconds": time.monotonic() - started}
    write_json(directory / "result.json", result)
