from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import gym
import gym_gvgai  # noqa: F401
import matplotlib.pyplot as plt

import dqn_agent as AgentModule
from multitask_unify import (
    UnifiedGVGAIWrapper,
    build_unified_spec,
    get_action_labels,
    inspect_env_specs,
    save_unified_spec,
)

# -----------------------------
# Config (edit here directly)
# -----------------------------
QUICK_TEST = True

# Full-mode split ratio on all lvl0 games.
FULL_SPLIT_TRAIN_RATIO = 0.70
FULL_SPLIT_VAL_RATIO = 0.15
FULL_SPLIT_TEST_RATIO = 0.15

# Quick-mode subset counts sampled from train/val/test splits.
QUICK_TRAIN_GAME_COUNT = 8
QUICK_VAL_GAME_COUNT = 3
QUICK_TEST_GAME_COUNT = 3

# Paper-scale production defaults.
PROD_TRAIN_TIMESTEPS = 1_000_000
PROD_TRAIN_ROUNDS = 3
PROD_POST_TRAIN_EVAL_EPISODES = 100
PROD_MAX_STEPS_PER_EPISODE = 2000

# Lightweight local smoke-test overrides.
QUICK_TRAIN_TIMESTEPS = 5000
QUICK_TRAIN_ROUNDS = 1
QUICK_POST_TRAIN_EVAL_EPISODES = 3
QUICK_MAX_STEPS_PER_EPISODE = 1000

if QUICK_TEST:
    TRAIN_TIMESTEPS = QUICK_TRAIN_TIMESTEPS
    TRAIN_ROUNDS = QUICK_TRAIN_ROUNDS
    POST_TRAIN_EVAL_EPISODES = QUICK_POST_TRAIN_EVAL_EPISODES
    MAX_STEPS_PER_EPISODE = QUICK_MAX_STEPS_PER_EPISODE
else:
    TRAIN_TIMESTEPS = PROD_TRAIN_TIMESTEPS
    TRAIN_ROUNDS = PROD_TRAIN_ROUNDS
    POST_TRAIN_EVAL_EPISODES = PROD_POST_TRAIN_EVAL_EPISODES
    MAX_STEPS_PER_EPISODE = PROD_MAX_STEPS_PER_EPISODE

SEED = 42

MODEL_DIR = "outputs/models/dqn"
OUT_DIR = "outputs/dqn_train"
CSV_PATH = os.path.join(OUT_DIR, "dqn_train_summary.csv")
JSON_PATH = os.path.join(OUT_DIR, "dqn_train_summary.json")
PLOT_SCORE_PATH = os.path.join(OUT_DIR, "dqn_train_avg_score.png")
PLOT_WINRATE_PATH = os.path.join(OUT_DIR, "dqn_train_win_rate.png")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "general_dqn_best.zip")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "general_dqn.zip")
UNIFIED_SPEC_PATH = os.path.join(MODEL_DIR, "multitask_spec.json")


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


def make_unified_env(env_id: str, unified_spec: Dict[str, Any]) -> Any:
    env = gym.make(env_id)
    local_labels = get_action_labels(env)
    return UnifiedGVGAIWrapper(
        env=env,
        target_obs_shape=unified_spec["target_obs_shape"],
        global_action_labels=unified_spec["global_action_labels"],
        local_action_labels=local_labels,
    )


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


def win_flag(info: Dict[str, Any], done: bool, score: float) -> bool:
    winner = str(info.get("winner", "")).upper()
    if "WIN" in winner and "LOSE" not in winner:
        return True
    if "LOSE" in winner:
        return False
    return bool(done and score > 0)


def evaluate_trained_agent(agent: Any, env: Any) -> Tuple[float, float]:
    scores: List[float] = []
    wins = 0

    if hasattr(agent, "bind_env"):
        agent.bind_env(env)

    if getattr(env, "is_unified_action_space", False):
        # Keep action indexing in the global unified action space.
        actions = list(range(int(env.action_space.n)))
    else:
        actions = (
            env.env.GVGAI.actions()
            if hasattr(env.env, "GVGAI")
            else list(range(env.action_space.n))
        )
    for episode in range(POST_TRAIN_EVAL_EPISODES):
        ep_idx = episode + 1
        state_obs = safe_reset(env, SEED + episode)
        current_score = 0.0
        won = False
        for _ in range(MAX_STEPS_PER_EPISODE):
            action_id = int(agent.act(state_obs, actions))
            state_obs, reward, done, info = safe_step(env, action_id)
            current_score += reward
            if done:
                won = win_flag(info, done=True, score=current_score)
                break
        scores.append(current_score)
        wins += int(won)
        if progress_mark(POST_TRAIN_EVAL_EPISODES, ep_idx, steps=4):
            print(
                f"    [Eval Progress] episode={ep_idx}/{POST_TRAIN_EVAL_EPISODES} "
                f"score={current_score:.2f} win={int(won)}"
            )

    avg_score = sum(scores) / len(scores) if scores else 0.0
    win_rate = wins / POST_TRAIN_EVAL_EPISODES if POST_TRAIN_EVAL_EPISODES else 0.0
    return avg_score, win_rate


def write_plots(rows: List[Dict[str, Any]]) -> None:
    ok_rows = [row for row in rows if row["status"] == "ok"]
    if not ok_rows:
        return

    env_ids = [row["env_id"] for row in ok_rows]
    avg_scores = [row["avg_score"] for row in ok_rows]
    win_rates = [row["win_rate"] for row in ok_rows]

    plt.figure(figsize=(12, 5))
    plt.bar(env_ids, avg_scores)
    plt.title("DQN General-Agent Final Test Average Score by Env")
    plt.ylabel("Average Score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(PLOT_SCORE_PATH, dpi=150)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.bar(env_ids, win_rates)
    plt.title("DQN General-Agent Final Test Win Rate by Env")
    plt.ylabel("Win Rate")
    plt.ylim(0.0, 1.0)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(PLOT_WINRATE_PATH, dpi=150)
    plt.close()


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    reset_output_files(
        [
            CSV_PATH,
            JSON_PATH,
            PLOT_SCORE_PATH,
            PLOT_WINRATE_PATH,
            BEST_MODEL_PATH,
            FINAL_MODEL_PATH,
            UNIFIED_SPEC_PATH,
        ]
    )

    print("[Stage] Discovering and splitting lvl0 environments...")

    discovered_ids = discover_gvgai_env_ids()
    lvl0_ids = only_lvl0_env_ids(discovered_ids)

    train_ids, val_ids, test_ids = split_env_ids_70_15_15(lvl0_ids)
    if QUICK_TEST:
        train_ids, val_ids, test_ids = pick_quick_subsets(train_ids, val_ids, test_ids)

    if not train_ids:
        raise RuntimeError("No train environments found for DQN training.")
    if not val_ids:
        raise RuntimeError("No val environments found for DQN validation.")
    if not test_ids:
        raise RuntimeError("No test environments found for DQN final test.")

    all_target_ids = train_ids + val_ids + test_ids
    env_specs = inspect_env_specs(all_target_ids)
    unified_spec = build_unified_spec(env_specs)
    save_unified_spec(UNIFIED_SPEC_PATH, unified_spec)

    total_segments = len(train_ids) * TRAIN_ROUNDS
    if total_segments <= 0:
        raise RuntimeError("Invalid training schedule: empty train split or rounds.")

    base_steps = TRAIN_TIMESTEPS // total_segments
    remainder_steps = TRAIN_TIMESTEPS % total_segments

    print(
        f"[Stage] Training setup quick_test={QUICK_TEST} "
        f"total_train_timesteps={TRAIN_TIMESTEPS} train_rounds={TRAIN_ROUNDS} "
        f"eval_episodes={POST_TRAIN_EVAL_EPISODES}"
    )
    print(
        f"[Stage] Split sizes train/val/test = {len(train_ids)}/{len(val_ids)}/{len(test_ids)}"
    )
    print(
        f"[Stage] Unified obs shape={tuple(unified_spec['target_obs_shape'])} "
        f"global actions={len(unified_spec['global_action_labels'])}"
    )
    print(
        f"[Stage] Training schedule segments={total_segments} "
        f"base_steps_per_segment={base_steps} remainder={remainder_steps}"
    )

    rows: List[Dict[str, Any]] = []
    agent = AgentModule.Agent(seed=SEED, model_dir=MODEL_DIR)
    trained_steps_total = 0
    segment_index = 0
    best_val_win_rate = -1.0
    best_val_score = float("-inf")
    best_round = -1

    print("[Stage] Unified training across train split...")
    for round_index in range(1, TRAIN_ROUNDS + 1):
        print(
            f"[Round] {round_index}/{TRAIN_ROUNDS} training over {len(train_ids)} environments"
        )
        for train_env_idx, env_id in enumerate(train_ids, start=1):
            segment_index += 1
            segment_steps = base_steps + (1 if segment_index <= remainder_steps else 0)
            if segment_steps <= 0:
                print(
                    f"  [Train Skip] segment={segment_index}/{total_segments} env={env_id} steps=0"
                )
                continue

            print(
                f"  [Train Progress] round={round_index}/{TRAIN_ROUNDS} "
                f"env={train_env_idx}/{len(train_ids)} id={env_id} steps={segment_steps}"
            )
            env = make_unified_env(env_id, unified_spec)
            try:
                safe_reset(env, seed=SEED + round_index + train_env_idx)
                trained_steps = agent.train(env, segment_steps)
                trained_steps_total += trained_steps
            finally:
                env.close()

        print(
            f"[Round] {round_index}/{TRAIN_ROUNDS} validating on {len(val_ids)} environments"
        )
        val_rows_round: List[Dict[str, Any]] = []
        for val_env_idx, env_id in enumerate(val_ids, start=1):
            print(f"  [Val Progress] env={val_env_idx}/{len(val_ids)} id={env_id}")
            env = make_unified_env(env_id, unified_spec)
            try:
                avg_score, win_rate = evaluate_trained_agent(agent, env)
                row = {
                    "phase": "val",
                    "round": round_index,
                    "group_key": "global",
                    "env_id": env_id,
                    "status": "ok",
                    "train_steps": trained_steps_total,
                    "avg_score": avg_score,
                    "win_rate": win_rate,
                    "model_path": "",
                    "error": "",
                }
            except Exception as exc:
                row = {
                    "phase": "val",
                    "round": round_index,
                    "group_key": "global",
                    "env_id": env_id,
                    "status": "error",
                    "train_steps": trained_steps_total,
                    "avg_score": 0.0,
                    "win_rate": 0.0,
                    "model_path": "",
                    "error": str(exc),
                }
                print(f"  [Val ERR] {env_id}: {exc}")
            finally:
                env.close()

            val_rows_round.append(row)
            rows.append(row)

        val_ok_rows = [x for x in val_rows_round if x["status"] == "ok"]
        if val_ok_rows:
            mean_val_win_rate = sum(x["win_rate"] for x in val_ok_rows) / len(
                val_ok_rows
            )
            mean_val_score = sum(x["avg_score"] for x in val_ok_rows) / len(val_ok_rows)
            print(
                f"[Round Val] round={round_index} mean_win_rate={mean_val_win_rate:.4f} "
                f"mean_avg_score={mean_val_score:.4f}"
            )
            is_better = (mean_val_win_rate > best_val_win_rate) or (
                abs(mean_val_win_rate - best_val_win_rate) <= 1e-12
                and mean_val_score > best_val_score
            )
            if is_better:
                best_val_win_rate = mean_val_win_rate
                best_val_score = mean_val_score
                best_round = round_index
                agent.save(BEST_MODEL_PATH)
                print(
                    f"[Checkpoint] New best validation model saved -> {BEST_MODEL_PATH}"
                )
        else:
            print(f"[Round Val] round={round_index} no successful validation runs")

    print("[Stage] Saving final unified model...")
    agent.save(FINAL_MODEL_PATH)

    model_for_test = (
        BEST_MODEL_PATH if os.path.exists(BEST_MODEL_PATH) else FINAL_MODEL_PATH
    )
    print(f"[Stage] Final test will use model -> {model_for_test}")

    print(f"[Stage] Running final test on {len(test_ids)} environments...")
    test_rows: List[Dict[str, Any]] = []
    for test_env_idx, env_id in enumerate(test_ids, start=1):
        print(f"  [Test Progress] env={test_env_idx}/{len(test_ids)} id={env_id}")
        env = make_unified_env(env_id, unified_spec)
        try:
            if hasattr(agent, "load"):
                agent.load(model_for_test, env)
            avg_score, win_rate = evaluate_trained_agent(agent, env)
            row = {
                "phase": "test",
                "round": best_round if best_round > 0 else TRAIN_ROUNDS,
                "group_key": "global",
                "env_id": env_id,
                "status": "ok",
                "train_steps": trained_steps_total,
                "avg_score": avg_score,
                "win_rate": win_rate,
                "model_path": model_for_test,
                "error": "",
            }
            print(
                f"  [Test OK] {env_id} avg_score={avg_score:.3f} win_rate={win_rate:.3f}"
            )
        except Exception as exc:
            row = {
                "phase": "test",
                "round": best_round if best_round > 0 else TRAIN_ROUNDS,
                "group_key": "global",
                "env_id": env_id,
                "status": "error",
                "train_steps": trained_steps_total,
                "avg_score": 0.0,
                "win_rate": 0.0,
                "model_path": model_for_test,
                "error": str(exc),
            }
            print(f"  [Test ERR] {env_id}: {exc}")
        finally:
            env.close()

        test_rows.append(row)
        rows.append(row)

    print("[Stage] Writing training reports and plots...")

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "phase",
                "round",
                "group_key",
                "env_id",
                "status",
                "train_steps",
                "avg_score",
                "win_rate",
                "model_path",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "quick_test": QUICK_TEST,
        "train_timesteps_total": TRAIN_TIMESTEPS,
        "train_rounds": TRAIN_ROUNDS,
        "train_segments": total_segments,
        "eval_episodes": POST_TRAIN_EVAL_EPISODES,
        "best_round": best_round,
        "best_val_win_rate": best_val_win_rate if best_round > 0 else None,
        "best_val_score": best_val_score if best_round > 0 else None,
        "model_paths": {
            "best": BEST_MODEL_PATH if os.path.exists(BEST_MODEL_PATH) else "",
            "final": FINAL_MODEL_PATH,
            "used_for_test": model_for_test,
            "unified_spec": UNIFIED_SPEC_PATH,
        },
        "unified": {
            "target_obs_shape": unified_spec["target_obs_shape"],
            "global_action_count": len(unified_spec["global_action_labels"]),
        },
        "split": {
            "train_count": len(train_ids),
            "val_count": len(val_ids),
            "test_count": len(test_ids),
            "train_env_ids": train_ids,
            "val_env_ids": val_ids,
            "test_env_ids": test_ids,
        },
        "targets": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
        },
        "ok": sum(1 for x in rows if x["status"] == "ok"),
        "error": sum(1 for x in rows if x["status"] != "ok"),
        "results": rows,
    }
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    write_plots(test_rows)
    print("[Stage] Training run finished.")
    print(f"Training CSV -> {CSV_PATH}")
    print(f"Training JSON -> {JSON_PATH}")
    print(f"Training plots -> {PLOT_SCORE_PATH}, {PLOT_WINRATE_PATH}")


if __name__ == "__main__":
    main()
