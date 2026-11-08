"""Execute the city workflow with mailbox or HTTP Agent transport."""
import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path

from map_generation.city_workflow import CityWorkflow, read_json, run_worker, write_json


def _main(stack):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["init", "resume", "status", "worker", "respond", "note", "reopen", "configure-agent"])
    parser.add_argument("--run", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--job", type=Path)
    parser.add_argument("--instruction", type=Path, help="User input file for respond, note or reopen")
    parser.add_argument("--max-actions", type=int, default=0,
                        help="Action budget for this resume invocation; 0 means unlimited (default)")
    args = parser.parse_args()
    if args.max_actions < 0:
        parser.error("--max-actions must be 0 (unlimited) or a positive integer")
    if args.action == "worker":
        run_worker(args.job)
        return
    workflow = CityWorkflow.create(args.run, read_json(args.config)) if args.action == "init" else CityWorkflow(args.run)
    if args.action in ("respond", "note", "reopen", "configure-agent"):
        if args.action == "configure-agent" and not args.config:
            parser.error("configure-agent requires --config FILE containing the Agent settings")
        if args.action != "configure-agent" and not args.instruction:
            parser.error(f"{args.action} requires --instruction FILE")
        lock = stack.enter_context((workflow.directory / "runner.lock").open("w"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        workflow = CityWorkflow(args.run)
        if args.action == "configure-agent":
            workflow.configure_agent(read_json(args.config))
            workflow.save()
            print(json.dumps({"status": workflow.state["status"], "stage": workflow.state["stage"],
                              "pending_request": workflow.state["pending_request"]}))
            return
        if args.action == "respond" and workflow.state["status"] != "blocked":
            raise ValueError("respond requires a blocked workflow")
        instruction = {"text": args.instruction.read_text(), "source": str(args.instruction.resolve())}
        if args.action == "reopen":
            workflow.reopen(instruction)
        else:
            workflow.state.setdefault("external_instructions", []).append(instruction)
            workflow.event("external_environment_note" if args.action == "note" else "user_instruction", instruction)
            if args.action == "respond":
                workflow.state["status"] = "running"
        workflow.save()
        print(json.dumps({"status": workflow.state["status"], "stage": workflow.state["stage"]}))
        return
    if args.action in ("init", "status"):
        print(json.dumps({k: workflow.state[k] for k in ("status", "stage", "revision", "pending_request")}, indent=2))
        return
    lock = stack.enter_context((workflow.directory / "runner.lock").open("w"))
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    workflow = CityWorkflow(args.run)
    from map_generation.agent_provider import FileMailboxProvider, OpenAICompatibleProvider, ResponsePending, ProviderError, AgentResponse
    actions = 0
    while workflow.state["status"] == "running":
        if args.max_actions and actions >= args.max_actions:
            break
        request = workflow.request()
        setting = workflow.agent_settings()
        if setting["provider"] == "mailbox":
            provider = FileMailboxProvider(workflow.directory / "mailbox", timeout_seconds=0)
        elif setting["provider"] == "http":
            provider = OpenAICompatibleProvider(base_url=setting["base_url"],
                        api_key=os.environ.get(setting.get("api_key_env", "MAP_AGENT_API_KEY")),
                        timeout_seconds=setting.get("timeout_seconds", 180))
        else:
            raise ValueError("Agent provider must be mailbox or http")
        saved_response = workflow.directory / "responses" / (request.request_id + ".json")
        try:
            response = AgentResponse(**read_json(saved_response)) if saved_response.exists() else provider.complete(request)
        except ResponsePending:
            print(json.dumps({"status": "awaiting_agent", "request_id": request.request_id,
                              "role": workflow.role(),
                              "request": str(workflow.directory / "mailbox/requests" / (request.request_id + ".json"))}))
            return
        except ProviderError as error:
            write_json(workflow.directory / "provider_errors" / (request.request_id + ".json"),
                       {"error": str(error), "raw_response": error.raw_response})
            raise
        write_json(workflow.directory / "responses" / (request.request_id + ".json"), response.to_dict())
        workflow.apply(response.action, request.request_id)
        actions += 1
    print(json.dumps({"status": workflow.state["status"], "stage": workflow.state["stage"],
                      "revision": workflow.state["revision"]}))


def main():
    with ExitStack() as stack:
        _main(stack)


if __name__ == "__main__":
    main()
