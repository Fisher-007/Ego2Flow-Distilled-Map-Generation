"""Clone a generated native map before a revised asset import."""
import json
from pathlib import Path


def clone_in_editor(source, target, output):
    import unreal
    level = unreal.EditorLevelLibrary
    if unreal.EditorAssetLibrary.does_asset_exist(target):
        raise ValueError("Use a new map name; preserve previous map revisions")
    if not level.new_level_from_template(target, source) or not level.save_current_level():
        raise RuntimeError("Could not save signal-facing map clone")
    Path(output).write_text(json.dumps({"source_map": source, "map": target,
        "operation": "Clone existing level; no mesh or material replacement", "saved": True}, indent=2) + "\n")

