from __future__ import annotations

import json
from collections import OrderedDict
from typing import Any, Dict, List, Sequence, Tuple

import gym
import numpy as np
from gym import spaces


def get_action_labels(env: Any) -> List[str]:
    """Return stable action labels for an env."""
    try:
        if hasattr(env, "env") and hasattr(env.env, "GVGAI"):
            return [str(a) for a in env.env.GVGAI.actions()]
    except Exception:
        pass

    try:
        # Fallback for non-GVGAI envs.
        n = int(env.action_space.n)
    except Exception:
        n = 0
    return [f"a{i}" for i in range(n)]


def inspect_env_specs(env_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Collect observation shapes and action labels for env ids."""
    specs: Dict[str, Dict[str, Any]] = {}
    for env_id in env_ids:
        env = gym.make(env_id)
        try:
            obs_shape = tuple(int(x) for x in env.observation_space.shape)
            action_labels = get_action_labels(env)
            specs[env_id] = {
                "obs_shape": list(obs_shape),
                "action_labels": action_labels,
            }
        finally:
            env.close()
    return specs


def build_unified_spec(specs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Build one shared observation shape and action vocabulary."""
    if not specs:
        raise RuntimeError("Cannot build unified spec from empty env specs.")

    obs_shapes = [tuple(v["obs_shape"]) for v in specs.values()]
    dims = len(obs_shapes[0])
    for shape in obs_shapes:
        if len(shape) != dims:
            raise RuntimeError("Observation rank mismatch across environments.")

    target_obs_shape = [max(shape[i] for shape in obs_shapes) for i in range(dims)]

    action_set: "OrderedDict[str, None]" = OrderedDict()
    for v in specs.values():
        for label in v["action_labels"]:
            action_set.setdefault(str(label), None)

    global_actions = list(action_set.keys())
    if not global_actions:
        raise RuntimeError(
            "Cannot build unified action space from empty action labels."
        )

    return {
        "target_obs_shape": target_obs_shape,
        "global_action_labels": global_actions,
        "env_specs": specs,
    }


def save_unified_spec(path: str, spec: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=2)


def load_unified_spec(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class UnifiedGVGAIWrapper(gym.Wrapper):
    """Unify obs/action spaces so one DQN can be reused across tasks."""

    def __init__(
        self,
        env: Any,
        target_obs_shape: Sequence[int],
        global_action_labels: Sequence[str],
        local_action_labels: Sequence[str],
    ) -> None:
        super().__init__(env)

        self.target_obs_shape = tuple(int(x) for x in target_obs_shape)
        self.global_action_labels = [str(x) for x in global_action_labels]
        self.local_action_labels = [str(x) for x in local_action_labels]

        # Marker used by evaluation code to choose action list strategy.
        self.is_unified_action_space = True

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=self.target_obs_shape,
            dtype=np.uint8,
        )
        self.action_space = spaces.Discrete(len(self.global_action_labels))

        local_index_by_label = {
            label: i for i, label in enumerate(self.local_action_labels)
        }
        self.global_to_local: Dict[int, int] = {}
        for global_idx, label in enumerate(self.global_action_labels):
            if label in local_index_by_label:
                self.global_to_local[global_idx] = int(local_index_by_label[label])

        # Keep legal local actions for fallback remapping.
        self.legal_local_actions = sorted(set(self.global_to_local.values()))
        if not self.legal_local_actions:
            # Defensive fallback for malformed env/action specs.
            self.legal_local_actions = [0]

    def _pad_or_crop_obs(self, obs: Any) -> np.ndarray:
        arr = np.asarray(obs)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        out = np.zeros(self.target_obs_shape, dtype=np.uint8)

        # Copy overlapping region.
        copy_slices = tuple(
            slice(0, min(arr.shape[i], out.shape[i]))
            for i in range(min(arr.ndim, out.ndim))
        )
        out[copy_slices] = arr[copy_slices]
        return out

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple):
            obs, info = out
            return self._pad_or_crop_obs(obs), info
        return self._pad_or_crop_obs(out)

    def step(self, action: int):
        action_int = int(action)
        local_action = self.global_to_local.get(action_int)
        if local_action is None:
            # Distribute invalid global actions across legal local actions
            # to avoid collapsing all invalid choices to a single command.
            local_action = self.legal_local_actions[
                action_int % len(self.legal_local_actions)
            ]
        out = self.env.step(local_action)

        if len(out) == 4:
            obs, reward, done, info = out
            return self._pad_or_crop_obs(obs), reward, done, info
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            return self._pad_or_crop_obs(obs), reward, terminated, truncated, info
        raise RuntimeError("Unknown env.step return format")
