"""Generate source-conditioned appearance with Cosmos 3."""
import argparse
import functools
import json
from pathlib import Path
import time


def local_safety_checker(model_root):
    """Keep official checks, reusing already downloaded checkpoint directories."""
    from cosmos_guardrail import cosmos_guardrail as guard
    locations = {
        'nvidia/Cosmos-1.0-Guardrail': model_root/'nvidia--Cosmos-Guardrail1',
        'Qwen/Qwen3Guard-Gen-0.6B': model_root/'Qwen--Qwen3Guard-Gen-0.6B',
    }
    original_download = guard.snapshot_download
    def local_download(repo_id, *args, **kwargs):
        if repo_id in locations:
            return str(locations[repo_id])
        return original_download(repo_id, *args, **kwargs)
    guard.snapshot_download = local_download
    originals = []
    def loader(original):
        @functools.wraps(original)
        def load(cls, name, *args, **kwargs):
            return original(str(locations.get(name, name)), *args, **kwargs)
        return classmethod(load)
    try:
        for cls in (guard.AutoTokenizer, guard.AutoModelForCausalLM):
            originals.append((cls, cls.__dict__['from_pretrained']))
            cls.from_pretrained = loader(cls.from_pretrained)
        return guard.CosmosSafetyChecker()
    finally:
        guard.snapshot_download = original_download
        for cls, method in originals:
            cls.from_pretrained = method


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--case', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--group-offload', action='store_true')
    p.add_argument('--guardrail-root', type=Path, required=True)
    p.add_argument('--information-conditions',type=Path,
                   help='Graph-compiled appearance requirements; native review still required')
    p.add_argument('--information-scope',choices=['scene_proposal','material_only'],default='scene_proposal')
    p.add_argument('--prepare-only',action='store_true',help='Export the actual conditioned input without loading weights')
    a = p.parse_args()
    case = json.loads(a.case.read_text())
    if a.information_conditions:
        from map_generation.information_graph import condition_appearance_case
        case=condition_appearance_case(case,json.loads(a.information_conditions.read_text()),a.information_scope)
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output/'conditioned_case.json').write_text(json.dumps(case,indent=2)+'\n')
    if a.prepare_only:
        print(a.output/'conditioned_case.json');return
    import torch
    from PIL import Image
    from diffusers import Cosmos3OmniModularPipeline, UniPCMultistepScheduler

    controls = {name: [Image.open(path).convert('RGB') for path in paths]
                for name, paths in case['control_images'].items()}
    started = time.monotonic()
    pipe = Cosmos3OmniModularPipeline.from_pretrained(
        str(a.model), dtype=torch.bfloat16,
        local_files_only=True)
    # Load the block graph required by the nested transfer loop.
    pipe.load_components(names=['text_tokenizer', 'vae', 'transformer', 'scheduler'],
                         dtype=torch.bfloat16, local_files_only=True,
                         pretrained_model_name_or_path=str(a.model))
    pipe.enable_safety_checker(local_safety_checker(a.guardrail_root))
    pipe.safety_checker.to(device='cuda')
    if a.group_offload:
        pipe.vae.to('cuda')
        pipe.transformer.enable_group_offload(
            onload_device=torch.device('cuda'), offload_device=torch.device('cpu'),
            offload_type='block_level', num_blocks_per_group=2, use_stream=False)
    else:
        pipe.to('cuda')
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=10.0, use_karras_sigmas=False)
    loaded = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    images = pipe(
        prompt=json.dumps(case['prompt']), negative_prompt=json.dumps(case['negative_prompt']),
        control_videos=controls, num_frames=len(next(iter(controls.values()))),
        height=case['height'], width=case['width'], fps=30.0,
        num_inference_steps=case.get('steps', 50), guidance_scale=3.0,
        control_guidance=case.get('control_guidance', 1.5),
        generator=torch.Generator(device='cuda').manual_seed(case['seed']),
        output='videos')
    for index, im in enumerate(images):
        im.save(a.output / f'generated_{index:03d}.png')
    report = {'model': str(a.model), 'pipeline': type(pipe).__name__, 'case': case,
              'group_offload': a.group_offload,
              'load_seconds': loaded-started, 'generation_seconds': time.monotonic()-loaded,
              'peak_allocated_gib': torch.cuda.max_memory_allocated()/1024**3,
              'generated_frames': len(images),
              'scope': 'Full conditional denoising with official pretrained weights. Appearance proposals only; native map materialization and driving are separate.'}
    (a.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
