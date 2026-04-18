from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import gym
import gym_gvgai  # noqa: F401
import matplotlib.pyplot as plt
from multitask_unify import UnifiedGVGAIWrapper, get_action_labels, load_unified_spec

# Swap only this import to evaluate a different agent module.
# import random_agent as AgentModule
import dqn_agent as AgentModule

# -----------------------------
# Config (edit here directly)
# -----------------------------
QUICK_TEST = True

# Evaluate this split by default.
# Allowed values: "train", "val", "test"
EVAL_SPLIT = "test"

# Full-mode split ratio on all lvl0 games.
FULL_SPLIT_TRAIN_RATIO = 0.70
FULL_SPLIT_VAL_RATIO = 0.15
FULL_SPLIT_TEST_RATIO = 0.15

# Quick-mode subset counts sampled from train/val/test splits.
QUICK_TRAIN_GAME_COUNT = 8
QUICK_VAL_GAME_COUNT = 3
QUICK_TEST_GAME_COUNT = 3

# Paper-scale production defaults.
PROD_EVAL_EPISODES = 100
PROD_MAX_STEPS_PER_EPISODE = 2000

# Lightweight local smoke-test overrides.
QUICK_EVAL_EPISODES = 3
QUICK_MAX_STEPS_PER_EPISODE = 1000

if QUICK_TEST:
    EVAL_EPISODES = QUICK_EVAL_EPISODES
    MAX_STEPS_PER_EPISODE = QUICK_MAX_STEPS_PER_EPISODE
else:
    EVAL_EPISODES = PROD_EVAL_EPISODES
    MAX_STEPS_PER_EPISODE = PROD_MAX_STEPS_PER_EPISODE

SEED = 42

OUT_DIR = "outputs/eval"
CSV_PATH = os.path.join(OUT_DIR, "agent_eval_scores.csv")
JSON_PATH = os.path.join(OUT_DIR, "agent_eval_scores.json")
PLOT_SCORE_PATH = os.path.join(OUT_DIR, "agent_eval_avg_score.png")
PLOT_WINRATE_PATH = os.path.join(OUT_DIR, "agent_eval_win_rate.png")
UNIFIED_SPEC_PATH = "outputs/models/dqn/multitask_spec.json"


def progress_mark(total: int, current: int, steps: int = 5) -> bool:
    """Return True when current hits an evenly spaced progress mark."""
    if total <= 0:
        return False
    if current == 1 or current == total:
        return True
    step_size = max(1, total // max(1, steps))
    return current % step_size == 0


def reset_output_files(paths: List[str]) -> None:
    """Remove old output files to avoid cross-run mixed logs/reports."""
    for path in paths:
        if os.path.exists(path):
            os.remove(path)


def discover_gvgai_env_ids() -> List[str]:
    registry = getattr(gym.envs, "registry", None)
    if registry is None:
        return []

    if hasattr(registry, "all"):
        env_ids = [spec.id for spec in registry.all() if spec.id.startswith("gvgai-")]
    elif hasattr(registry, "keys"):
        env_ids = [str(k) for k in registry.keys() if str(k).startswith("gvgai-")]
    else:
        env_ids = []
    return sorted(set(env_ids))


def only_lvl0_env_ids(env_ids: List[str]) -> List[str]:
    return [env_id for env_id in env_ids if "-lvl0-" in env_id]


def split_env_ids_70_15_15(
    env_ids: List[str],
) -> Tuple[List[str], List[str], List[str]]:
    ids = sorted(env_ids)
    n = len(ids)
    if n == 0:
        return [], [], []

    n_train = int(n * FULL_SPLIT_TRAIN_RATIO)
    n_val = int(n * FULL_SPLIT_VAL_RATIO)
    n_test = n - n_train - n_val

    if n >= 3:
        n_train = max(1, n_train)
        n_val = max(1, n_val)
        n_test = max(1, n_test)
        total = n_train + n_val + n_test
        while total > n:
            if n_train >= n_val and n_train >= n_test and n_train > 1:
                n_train -= 1
            elif n_val >= n_test and n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            total = n_train + n_val + n_test
        while total < n:
            n_train += 1
            total += 1
    else:
        n_train = n
        n_val = 0
        n_test = 0

    train_ids = ids[:n_train]
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val : n_train + n_val + n_test]
    return train_ids, val_ids, test_ids


def pick_quick_subsets(
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
) -> Tuple[List[str], List[str], List[str]]:
    return (
        train_ids[: min(QUICK_TRAIN_GAME_COUNT, len(train_ids))],
        val_ids[: min(QUICK_VAL_GAME_COUNT, len(val_ids))],
        test_ids[: min(QUICK_TEST_GAME_COUNT, len(test_ids))],
    )


def safe_reset(env: Any, seed: int) -> Any:
    try:
        out = env.reset(seed=seed)
    except TypeError:
        out = env.reset()
    if isinstance(out, tuple):
        return out[0]
    return out


def safe_step(env: Any, action: int) -> Tuple[Any, float, bool, Dict[str, Any]]:
    out = env.step(action)
    if len(out) == 4:
        obs, reward, done, info = out
        return obs, float(reward), bool(done), dict(info)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, float(reward), bool(terminated or truncated), dict(info)
    raise RuntimeError("Unknown env.step return format")


def is_win(info: Dict[str, Any], done: bool, score: float) -> bool:
    winner = str(info.get("winner", "")).upper()
    if "WIN" in winner and "LOSE" not in winner:
        return True
    if "LOSE" in winner:
        return False
    return bool(done and score > 0)


def get_actions_for_agent(env: Any) -> List[Any]:
    if getattr(env, "is_unified_action_space", False):
        return list(range(int(env.action_space.n)))

    try:
        if hasattr(env.env, "GVGAI"):
            return list(env.env.GVGAI.actions())
    except Exception:
        pass

    try:
        return list(env.unwrapped.get_action_meanings())
    except Exception:
        return list(range(int(env.action_space.n)))


def evaluate_one_env(
    agent: Any, env_id: str, unified_spec: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    base_env = gym_gvgai.make(env_id)
    env = base_env
    if unified_spec is not None:
        env = UnifiedGVGAIWrapper(
            env=base_env,
            target_obs_shape=unified_spec["target_obs_shape"],
            global_action_labels=unified_spec["global_action_labels"],
            local_action_labels=get_action_labels(base_env),
        )

    try:
        if hasattr(agent, "on_env_start"):
            agent.on_env_start(env)

        actions = get_actions_for_agent(env)
        scores: List[float] = []
        wins = 0

        for ep in range(EVAL_EPISODES):
            ep_idx = ep + 1
            state_obs = safe_reset(env, SEED + ep)
            current_score = 0.0
            won = False

            for _ in range(MAX_STEPS_PER_EPISODE):
                action_id = int(agent.act(state_obs, actions))
                state_obs, reward, done, info = safe_step(env, action_id)
                current_score += reward
                if done:
                    won = is_win(info, done=True, score=current_score)
                    break

            scores.append(current_score)
            wins += int(won)
            if progress_mark(EVAL_EPISODES, ep_idx, steps=4):
                print(
                    f"    [Eval Progress] episode={ep_idx}/{EVAL_EPISODES} "
                    f"score={current_score:.2f} win={int(won)}"
                )

        return {
            "avg_score": sum(scores) / len(scores),
            "win_rate": wins / EVAL_EPISODES,
            "status": "ok",
            "error": "",
        }
    finally:
        env.close()


def write_plots(rows: List[Dict[str, Any]]) -> None:
    ok_rows = [row for row in rows if row["status"] == "ok"]
    if not ok_rows:
        return

    env_ids = [row["env_id"] for row in ok_rows]
    avg_scores = [row["avg_score"] for row in ok_rows]
    win_rates = [row["win_rate"] for row in ok_rows]

    plt.figure(figsize=(12, 5))
    plt.bar(env_ids, avg_scores)
    plt.title("Agent Average Score by Env")
    plt.ylabel("Average Score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(PLOT_SCORE_PATH, dpi=150)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.bar(env_ids, win_rates)
    plt.title("Agent Win Rate by Env")
    plt.ylabel("Win Rate")
    plt.ylim(0.0, 1.0)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(PLOT_WINRATE_PATH, dpi=150)
    plt.close()


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    reset_output_files([CSV_PATH, JSON_PATH, PLOT_SCORE_PATH, PLOT_WINRATE_PATH])

    use_unified_for_dqn = bool(
        "dqn" in str(getattr(AgentModule, "__name__", "")).lower()
        and os.path.exists(UNIFIED_SPEC_PATH)
    )
    unified_spec: Dict[str, Any] | None = (
        load_unified_spec(UNIFIED_SPEC_PATH) if use_unified_for_dqn else None
    )

    print("[Stage] Discovering and splitting lvl0 environments...")

    discovered_ids = discover_gvgai_env_ids()
    lvl0_ids = only_lvl0_env_ids(discovered_ids)

    train_ids, val_ids, test_ids = split_env_ids_70_15_15(lvl0_ids)
    if QUICK_TEST:
        train_ids, val_ids, test_ids = pick_quick_subsets(train_ids, val_ids, test_ids)

    if EVAL_SPLIT == "train":
        env_ids = train_ids
    elif EVAL_SPLIT == "val":
        env_ids = val_ids
    else:
        env_ids = test_ids

    if not env_ids:
        raise RuntimeError("No target environments found for evaluation.")

    print(
        f"[Stage] Evaluation setup quick_test={QUICK_TEST} split={EVAL_SPLIT} "
        f"episodes={EVAL_EPISODES}"
    )
    if use_unified_for_dqn and unified_spec is not None:
        print(
            f"[Stage] Unified eval obs shape={tuple(unified_spec['target_obs_shape'])} "
            f"global actions={len(unified_spec['global_action_labels'])}"
        )
    print(
        f"[Stage] Split sizes train/val/test = {len(train_ids)}/{len(val_ids)}/{len(test_ids)}"
    )

    agent_name = str(getattr(AgentModule, "__name__", "agent"))
    rows: List[Dict[str, Any]] = []
    total_envs = len(env_ids)
    for env_index, env_id in enumerate(env_ids, start=1):
        print(f"[Eval Progress] env={env_index}/{total_envs} id={env_id}")
        agent = AgentModule.Agent()
        try:
            result = evaluate_one_env(agent, env_id, unified_spec=unified_spec)
            rows.append(
                {
                    "agent": agent_name,
                    "env_id": env_id,
                    "avg_score": result["avg_score"],
                    "win_rate": result["win_rate"],
                    "status": result["status"],
                    "error": result["error"],
                }
            )
            print(f"[OK] {agent_name} @ {env_id}")
        except Exception as exc:
            rows.append(
                {
                    "agent": agent_name,
                    "env_id": env_id,
                    "avg_score": 0.0,
                    "win_rate": 0.0,
                    "status": "error",
                    "error": str(exc),
                }
            )
            print(f"[ERR] {agent_name} @ {env_id}: {exc}")

    print("[Stage] Writing evaluation reports and plots...")

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["agent", "env_id", "avg_score", "win_rate", "status", "error"],
        )
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "agent_module": agent_name,
        "quick_test": QUICK_TEST,
        "eval_split": EVAL_SPLIT,
        "eval_episodes": EVAL_EPISODES,
        "max_steps_per_episode": MAX_STEPS_PER_EPISODE,
        "split": {
            "train_count": len(train_ids),
            "val_count": len(val_ids),
            "test_count": len(test_ids),
            "train_env_ids": train_ids,
            "val_env_ids": val_ids,
            "test_env_ids": test_ids,
        },
        "results": rows,
    }
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    write_plots(rows)

    print("[Stage] Evaluation run finished.")

    print(f"Evaluation CSV -> {CSV_PATH}")
    print(f"Evaluation JSON -> {JSON_PATH}")
    print(f"Evaluation plots -> {PLOT_SCORE_PATH}, {PLOT_WINRATE_PATH}")


if __name__ == "__main__":
    main()
