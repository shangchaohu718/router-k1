"""Native simulator evaluation plus ordered full-observation/router recording.

Run only through the monitored GPU job guard. Saved inputs are raw assembled
Actor observations, suitable for rebuilding matching ordered teacher histories.
"""

from __future__ import annotations
import atexit
import os
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from holomotion.src.algo.ppo_tf import PPOTF
from tools.booster_k1_retarget.evaluate_sliding_router import evaluate


def attach_recorder(algo, destination):
    actor = algo.accelerator.unwrap_model(algo.actor)
    layer = actor.actor_module.layers[0]
    command = algo.env._env.command_manager.get_term("ref_motion")
    records = []
    current = {"batch": None, "id": -1}

    def record(module, args, kwargs, result):
        if (
            kwargs.get("mode") != "inference"
            or not args
            or not isinstance(args[0], TensorDict)
        ):
            return
        obs = args[0]
        if int(obs.batch_size[0]) != 1:
            raise ValueError(
                "Internal evaluation recording requires one environment "
                "to preserve clip outputs and ordered histories"
            )
        cache = command._motion_cache
        if current["batch"] is not cache.current_batch:
            current.update(batch=cache.current_batch, id=current["id"] + 1)
        keys = [
            str(k)
            for k in cache.motion_keys_for_indices(command._clip_indices[:1])
        ]
        with torch.no_grad():
            raw = module.assembler(obs)
            selected = layer.last_topk_indices[:, 0]
            logits = layer.last_expanded_logits[:, 0]
            weights = layer.last_topk_weights
            if weights is None:
                scores = (
                    logits.softmax(-1)
                    if layer.routing_score_fn == "softmax"
                    else logits.sigmoid()
                )
                weights = scores.gather(-1, selected)
                weights = weights / weights.sum(-1, keepdim=True).clamp_min(
                    1e-20
                )
            else:
                weights = weights[:, 0]
        records.append(
            {
                "observations": raw.detach().cpu().numpy(),
                "mu": result["mu"].detach().cpu().numpy(),
                "sigma": result["sigma"].detach().cpu().numpy(),
                "selected_experts": selected.cpu().numpy(),
                "selected_weights": weights.cpu().numpy(),
                "router_logits": logits.detach().cpu().numpy(),
                "motion_keys": np.asarray(keys),
                "record_steps": np.asarray([len(records)], dtype=np.int64),
                "batch_ids": np.asarray([current["id"]], dtype=np.int64),
                "reference_frames": command._frame_indices[:1]
                .cpu()
                .numpy()
                .copy(),
                "episode_positions": (
                    module.actor_module._kv_cache_abs_pos[:1] - 1
                )
                .cpu()
                .numpy()
                .copy(),
            }
        )

    actor.register_forward_hook(record, with_kwargs=True)

    def save():
        if not records:
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            np.savez_compressed(
                stream,
                schema_version=np.asarray(1),
                old_fine_experts=np.asarray(layer.num_fine_experts),
                added_fine_experts=np.asarray(layer.num_added_experts),
                **{
                    key: np.concatenate([r[key] for r in records])
                    for key in records[0]
                },
            )

    atexit.register(save)
    return save


if __name__ == "__main__":
    destination = os.environ.get("INTERNAL_MOE_RECORD_PATH")
    if destination:
        original_setup = PPOTF._setup_models_and_optimizer

        def setup_with_recorder(self):
            original_setup(self)
            attach_recorder(self, Path(destination))

        PPOTF._setup_models_and_optimizer = setup_with_recorder
    evaluate()
