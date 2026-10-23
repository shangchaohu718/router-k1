"""Native evaluator with optional reference-only gate sample recording.

Invoke through the shared-resource job guard. Labels come from the selected
motion's intended task, never from whether the physical robot actually slips.
"""

from __future__ import annotations
import atexit
import os
from pathlib import Path

import hydra
import numpy as np
import torch
from tensordict import TensorDict

from holomotion.src.algo.ppo_tf import PPOTF
from holomotion.src.evaluation.eval_motion_tracking_single import main


@hydra.main(
    config_path=str(Path(__file__).resolve().parents[2] / "holomotion/config"),
    config_name="evaluation/eval_isaaclab",
    version_base=None,
)
def evaluate(config):
    # Imported Hydra entry points otherwise resolve through package resources;
    # Use filesystem config lookup for this namespace-package checkout.
    main.__wrapped__(config)


def attach_gate_recorder(algo, destination):
    actor = algo.accelerator.unwrap_model(algo.actor)
    records = []
    batch_state = {"batch": None, "index": -1}
    command = algo.env._env.command_manager.get_term("ref_motion")

    def record(module, args, kwargs, result):
        if kwargs.get("mode") != "inference" or not args:
            return
        obs = args[0]
        if not isinstance(obs, TensorDict):
            return
        cache = command._motion_cache
        batch = cache.current_batch
        if batch_state["batch"] is not batch:
            batch_state.update(batch=batch, index=batch_state["index"] + 1)
        count = min(
            len(batch.motion_keys),
            int(obs.batch_size[0]),
            int(getattr(cache, "clip_count", len(batch.motion_keys))),
        )
        # Completed environments may already have another clip assignment
        # while the evaluator finishes longer peers in the original batch.
        # The current per-environment indices, not batch row order, determine
        # the reference actually present in this observation.
        clip_indices = command._clip_indices[:count]
        keys = [str(k) for k in cache.motion_keys_for_indices(clip_indices)]
        with torch.no_grad():
            flat = module.assembler(obs)
            flat = module._normalize_actor_obs(flat, update=False)
            indices = module.actor_module._router_feature_indices
            features = flat[:count].index_select(-1, indices).cpu().numpy()
        labels = np.array(
            [
                -1
                if k.startswith("router_transition_")
                else int("moonwalk" in k.lower())
                for k in keys[:count]
            ],
            dtype=np.float32,
        )
        records.append(
            {
                "features": features,
                "labels": labels,
                "motion_keys": np.array(keys),
                "env_ids": np.arange(count, dtype=np.int64),
                "record_steps": np.full(count, len(records), dtype=np.int64),
                "batch_ids": np.full(
                    count, batch_state["index"], dtype=np.int64
                ),
                "reference_frames": command._frame_indices[:count]
                .cpu()
                .numpy()
                .copy(),
                "gate_probability": result["slide_gate_probability"][:count]
                .detach()
                .cpu()
                .numpy()
                .ravel(),
                "gate_weight": result["slide_gate_weight"][:count]
                .detach()
                .cpu()
                .numpy()
                .ravel(),
            }
        )

    actor.register_forward_hook(record, with_kwargs=True)

    def save():
        if not records:
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                schema_version=np.array(2),
                **{
                    key: np.concatenate([r[key] for r in records])
                    for key in records[0]
                },
            )
        temporary.replace(destination)

    atexit.register(save)
    return save


if __name__ == "__main__":
    capture_path = os.environ.get("SLIDING_GATE_RECORD_PATH")
    if capture_path:
        original_setup = PPOTF._setup_models_and_optimizer

        def setup_with_recorder(self):
            original_setup(self)
            attach_gate_recorder(self, Path(capture_path))

        PPOTF._setup_models_and_optimizer = setup_with_recorder
    evaluate()
