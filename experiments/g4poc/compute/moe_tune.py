"""Run SGLang's fused-MoE Triton tuner on one GPU without Ray.

`benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` drives its workers
through Ray, which the study venv does not ship. With a single GPU, Ray only
round-robins calls onto one actor, so an in-process stand-in that runs each call
synchronously gives the same results. Arguments are the tuner's own, e.g.:

  python compute/moe_tune.py --tree <sglang tree> -- \
      --model /data/jooman/g4poc/models/bf16-meta --tp-size 1 \
      --dtype fp8_w8a8 --per-channel-quant --tune

Use the BF16 meta dir as --model: the tuner reads Gemma-4's MoE shape only from
`Gemma4ForConditionalGeneration`, and a compressed-tensors `quantization_config`
would send it down its group-quant branch. Without --tune it benchmarks whatever
config SGLang would load (set SGLANG_MOE_CONFIG_DIR to compare a tuned file).
"""

import argparse
import os
import sys
import types


class _Actor:
    """A synchronous stand-in for a Ray actor handle: `h.method.remote(*a)` calls it now."""

    def __init__(self, obj):
        self._obj = obj

    def __getattr__(self, name):
        method = getattr(self._obj, name)
        return types.SimpleNamespace(remote=method)


class _ActorClass:
    def __init__(self, cls):
        self._cls = cls

    def remote(self, *args, **kwargs):
        return _Actor(self._cls(*args, **kwargs))


def fake_ray() -> types.ModuleType:
    """The slice of the `ray` API the tuner uses, for exactly one visible GPU."""

    def remote(*args, **kwargs):
        if len(args) == 1 and isinstance(args[0], type) and not kwargs:
            return _ActorClass(args[0])
        return _ActorClass

    ray = types.ModuleType("ray")
    ray.remote = remote
    ray.init = lambda *a, **k: None
    ray.get = lambda x: x
    ray.available_resources = lambda: {"GPU": 1}
    ray.get_gpu_ids = lambda: [0]
    experimental = types.ModuleType("ray.experimental")
    tqdm_ray = types.ModuleType("ray.experimental.tqdm_ray")
    from tqdm import tqdm

    tqdm_ray.tqdm = tqdm
    ray.experimental = experimental
    experimental.tqdm_ray = tqdm_ray
    return ray


def install_fake_ray() -> None:
    ray = fake_ray()
    sys.modules.update({"ray": ray, "ray.experimental": ray.experimental,
                        "ray.experimental.tqdm_ray": ray.experimental.tqdm_ray})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tree", required=True, help="SGLang source tree whose tuner and kernels to use")
    p.add_argument("tuner_args", nargs=argparse.REMAINDER)
    args = p.parse_args()
    tuner_args = args.tuner_args[1:] if args.tuner_args[:1] == ["--"] else args.tuner_args
    bench_dir = os.path.join(args.tree, "benchmark", "kernels", "fused_moe_triton")
    sys.path[:0] = [bench_dir, os.path.join(args.tree, "python")]
    install_fake_ray()
    sys.argv = ["tuning_fused_moe_triton.py", *tuner_args]
    import runpy

    runpy.run_path(os.path.join(bench_dir, "tuning_fused_moe_triton.py"), run_name="__main__")


if __name__ == "__main__":
    main()
