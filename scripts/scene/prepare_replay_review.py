"""Prepare paired source and rendered replay views for review."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def prepare(history_path, replay_directory, output, interval=20, extra_frames=(), panel_width=768):
    history = json.loads(history_path.read_text())
    frames = history["frames"]
    if interval < 1:
        raise ValueError("Frame interval must be positive")
    selected = sorted(set(range(0, len(frames), interval)) | {0, len(frames)-1} | set(extra_frames))
    if any(i < 0 or i >= len(frames) for i in selected):
        raise ValueError("Requested review frame is outside this recorded window")
    output.mkdir(parents=True, exist_ok=True)
    rows, panels = [], []
    for i in selected:
        frame = frames[i]
        source_path = Path(frame["source_image"])
        native_path = replay_directory / "rgb" / f"{i:03d}.png"
        source, native = cv2.imread(str(source_path)), cv2.imread(str(native_path))
        if source is None or native is None:
            raise ValueError(f"Missing paired image at frame {i}")
        if source.shape != native.shape:
            raise ValueError(f"Different calibrated image sizes at frame {i}")
        height = round(source.shape[0] * panel_width / source.shape[1])
        pair = np.concatenate([cv2.resize(x, (panel_width, height), interpolation=cv2.INTER_AREA)
                               for x in (source, native)], axis=1)
        pair_path = output / f"pair_{i:03d}.png"
        cv2.imwrite(str(pair_path), pair)
        panel = np.full((height+32, pair.shape[1], 3), 245, np.uint8)
        panel[32:] = pair
        label = f"frame {i:03d} | timestamp {frame['timestamp_ns']} | source LEFT / native RIGHT"
        cv2.putText(panel, label, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, .62, (20, 20, 20), 1, cv2.LINE_AA)
        panels.append(panel)
        rows.append(dict(frame=i, timestamp_ns=frame["timestamp_ns"], source=str(source_path),
                         native=str(native_path), paired=str(pair_path),
                         reasons=[reason for flag, reason in [
                             (i in (0, len(frames)-1), "window_endpoint"),
                             (i % interval == 0, "regular_time_sample"),
                             (i in extra_frames, "recorded_feedback")] if flag]))
    sheets = []
    for start in range(0, len(panels), 2):
        path = output / f"sheet_{start//2:02d}.png"
        cv2.imwrite(str(path), np.concatenate(panels[start:start+2], axis=0))
        sheets.append(dict(path=str(path), frames=selected[start:start+2]))
    result = dict(history=str(history_path), replay=str(replay_directory), total_frames=len(frames),
                  interval_frames=interval, selected_frames=selected, images=rows, sheets=sheets,
                  image_processing="Both complete images equally resized; labels outside image pixels. Original source/native paths retained.",
                  visual_review_status="not_reviewed",
                  scope="Ordered temporal evidence including both endpoints and explicit feedback. Sampling is not exhaustive frame or pixel acceptance.")
    (output / "index.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=20)
    parser.add_argument("--frames", nargs="*", type=int, default=[])
    parser.add_argument("--panel-width", type=int, default=768)
    args = parser.parse_args()
    result = prepare(args.history, args.replay, args.output, args.interval, args.frames, args.panel_width)
    print(json.dumps({key: result[key] for key in ("total_frames", "selected_frames", "sheets", "visual_review_status")}), flush=True)


if __name__ == "__main__":
    main()
