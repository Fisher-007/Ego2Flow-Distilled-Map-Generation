"""List registered tools and inspect their CLI declarations from source.

Static inspection works without runtime dependencies. Dynamic parsers expose
their complete interface through the tool's --help command.
"""

import argparse
import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def static_value(node, bindings):
    if isinstance(node, ast.Name) and node.id in bindings:
        return bindings[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return static_value(node.left, bindings) + static_value(node.right, bindings)
    if isinstance(node, (ast.List, ast.Tuple)):
        return [static_value(value, bindings) for value in node.elts]
    return ast.literal_eval(node)


def cli_declarations(path):
    """Expand literal argparse loops; leave dynamic values as source expressions."""
    tree = ast.parse(path.read_text(), filename=str(path))
    rows = []

    def value(node, bindings):
        try:
            return static_value(node, bindings)
        except (ValueError, TypeError, KeyError):
            return {"expression": ast.unparse(node)}

    def visit(node, bindings):
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            items = value(node.iter, bindings)
            if isinstance(items, (list, tuple)):
                for item in items:
                    nested = dict(bindings, **{node.target.id: item})
                    for statement in node.body:
                        visit(statement, nested)
                for statement in node.orelse:
                    visit(statement, bindings)
                return
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("add_argument", "add_parser")):
            rows.append({
                "kind": node.func.attr,
                "line": node.lineno,
                "names": [value(arg, bindings) for arg in node.args],
                "options": {item.arg: value(item.value, bindings)
                            for item in node.keywords if item.arg is not None},
            })
        for child in ast.iter_child_nodes(node):
            visit(child, bindings)

    visit(tree, {})
    return {"description": ast.get_docstring(tree), "declarations": rows,
            "scope": "Static argparse declarations; unresolved expressions are preserved."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path,
                        default=ROOT / "configs/workflows/city_tools.json")
    parser.add_argument("--tool", action="append", help="Registry tool ID; repeat to select several")
    parser.add_argument("--stage", help="Filter the workflow stage")
    parser.add_argument("--search", help="Case-insensitive search in tool descriptions")
    parser.add_argument("--signatures", action="store_true", help="Read CLI declarations from source")
    parser.add_argument("--module", help="Inspect an additional repository module, e.g. scripts.scene.build_roads")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text())
    selected = []
    for tool in registry["tools"]:
        if args.module and not (args.tool or args.stage or args.search):
            continue
        if args.tool and tool["id"] not in args.tool:
            continue
        if args.stage and args.stage not in tool["stages"]:
            continue
        if args.search and args.search.lower() not in json.dumps(tool).lower():
            continue
        row = dict(tool)
        entry = row["entrypoint"]
        path = ROOT / (entry["target"].replace(".", "/") + ".py"
                       if entry["kind"] == "module" else entry["target"])
        row["source"] = str(path)
        row["source_exists"] = path.is_file()
        if args.signatures and path.suffix == ".py" and path.is_file():
            row["cli"] = cli_declarations(path)
        selected.append(row)
    result = {"schema_version": registry["schema_version"], "tools": selected}
    if args.module:
        path = ROOT / (args.module.replace(".", "/") + ".py")
        result["additional_module"] = {"module": args.module, "source": str(path),
                                       "cli": cli_declarations(path)}
    encoded = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
