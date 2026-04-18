from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


class Agent:
    """DQN baseline with the same act signature as the original Agent.py."""

    def __init__(
        self,
        policy: str = "CnnPolicy",
        learning_rate: float = 1e-4,
        buffer_size: int = 50000,
        batch_size: int = 32,
        target_update_interval: int = 500,
        seed: int = 42,
        model_dir: str = "outputs/models/dqn",
    ) -> None:
        self.name = "dqnAgent"
        self.policy = policy
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.seed = seed
        self.model_dir = model_dir
        self.model: Optional[Any] = None
        self.current_env_id: Optional[str] = None
        self.total_trained_steps = 0

    def _import_dqn(self):
        try:
            from stable_baselines3 import DQN
        except Exception as exc:
            raise RuntimeError(
                "stable-baselines3 is required. Install with: pip install stable-baselines3 shimmy"
            ) from exc
        return DQN

    def _extract_env_id(self, env: Any) -> str:
        spec = getattr(env, "spec", None)
        if spec is not None and getattr(spec, "id", None):
            return str(spec.id)
        return "unknown_env"

    def _build_model(self, env: Any) -> Any:
        DQN = self._import_dqn()
        return DQN(
            policy=self.policy,
            env=env,
            learning_rate=self.learning_rate,
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            target_update_interval=self.target_update_interval,
            seed=self.seed,
            verbose=0,
        )

    def create(self, env: Any) -> None:
        if self.model is not None:
            return
        self.model = self._build_model(env)
        self.current_env_id = self._extract_env_id(env)

    def bind_env(self, env: Any) -> None:
        """Bind an existing model to a new env with compatible spaces."""
        self.create(env)
        assert self.model is not None
        try:
            self.model.set_env(env)
        except ValueError:
            # Mixed GVGAI games can have different observation/action spaces.
            # Recreate model for the new space instead of crashing.
            self.model = self._build_model(env)
        self.current_env_id = self._extract_env_id(env)

    def train(self, env: Any, timesteps: int) -> int:
        self.bind_env(env)
        assert self.model is not None
        self.model.learn(
            total_timesteps=int(timesteps),
            reset_num_timesteps=False,
            progress_bar=False,
        )
        self.total_trained_steps += int(timesteps)
        return int(timesteps)

    def save(self, model_path: str) -> None:
        if self.model is None:
            raise RuntimeError("DQN model is not initialized.")
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(model_path)

    def load(self, model_path: str, env: Any) -> None:
        DQN = self._import_dqn()
        self.model = DQN.load(model_path, env=env)
        self.current_env_id = self._extract_env_id(env)

    def on_env_start(self, env: Any) -> None:
        env_id = self._extract_env_id(env)
        if self.current_env_id == env_id and self.model is not None:
            return

        # Prefer env-specific checkpoints; fallback to generic checkpoints.
        candidate_paths = [
            Path(self.model_dir) / f"{env_id}.zip",
            Path(self.model_dir) / "general_dqn_best.zip",
            Path(self.model_dir) / "general_dqn.zip",
        ]

        for candidate in candidate_paths:
            if candidate.exists():
                self.load(str(candidate), env)
                return

        raise RuntimeError(
            "Missing DQN model. Checked: "
            + ", ".join(str(path) for path in candidate_paths)
        )

    def act(self, stateObs: Any, actions: Any) -> int:
        if self.model is None:
            raise RuntimeError("DQN model is not loaded/trained.")
        action, _ = self.model.predict(stateObs, deterministic=True)
        action_id = int(action)

        try:
            action_count = len(actions)
        except Exception:
            action_count = 0
        if action_count > 0:
            action_id = action_id % action_count

        return action_id


# Compatibility alias for newer-style imports.
DQNAgent = Agent
