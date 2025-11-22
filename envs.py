"""
Environment definitions for FBÇT simulations.

Currently implements a configurable multi-armed bandit with optional context.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class BanditEnv:
    """
    Multi-armed bandit environment.

    Each arm exposes a stationary reward distribution (Gaussian by default,
    Bernoulli if `reward_stds` is None). The state embeds minimal signals that
    the agent can treat as sensory input: timestep, last action, last reward,
    and optional context.
    """

    n_arms: int
    reward_means: np.ndarray
    reward_stds: Optional[np.ndarray] = None
    context_dim: int = 0
    switch_step: Optional[int] = None
    switched_reward_means: Optional[np.ndarray] = None
    obs_mask_prob: float = 0.0
    obs_noise_std: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.reward_means.shape[0] != self.n_arms:
            raise ValueError("reward_means must have shape (n_arms,)")
        if self.reward_stds is not None and self.reward_stds.shape[0] != self.n_arms:
            raise ValueError("reward_stds must have shape (n_arms,) when provided")
        if self.switch_step is not None and self.switched_reward_means is None:
            raise ValueError("switched_reward_means must be provided when switch_step is set")
        if self.switched_reward_means is not None and self.switched_reward_means.shape[0] != self.n_arms:
            raise ValueError("switched_reward_means must have shape (n_arms,) when provided")

        self.rng = np.random.default_rng(self.seed)
        self.timestep = 0
        self.current_step = 0
        self.last_action = -1  # -1 indicates no action yet
        self.last_reward = 0.0
        self.context = np.zeros(self.context_dim, dtype=float)
        self.active_reward_means = self.reward_means.copy()

    # Public API ---------------------------------------------------------
    def reset(self) -> np.ndarray:
        """
        Reset environment to start state.

        Returns:
            Initial sensory state vector S_0.
        """
        self.timestep = 0
        self.current_step = 0
        self.last_action = -1
        self.last_reward = 0.0
        self.active_reward_means = self.reward_means.copy()
        return self.get_state()

    def _construct_state(self) -> np.ndarray:
        """
        Construct current sensory state S_t.

        For the basic bandit we provide:
        - Normalized timestep (0..1) for a nominal horizon of 1000 steps.
        - Last action chosen (one-hot with a slot for "no action yet").
        - Last reward observed.
        - Optional context vector.
        """
        time_feature = np.array([min(self.timestep / 1000.0, 1.0)], dtype=float)

        # One-hot encoding with extra slot for "no previous action"
        action_one_hot = np.zeros(self.n_arms + 1, dtype=float)
        action_index = self.last_action + 1  # shift by 1 to map -1 -> 0
        action_one_hot[action_index] = 1.0

        reward_feature = np.array([self.last_reward], dtype=float)

        return np.concatenate([time_feature, action_one_hot, reward_feature, self.context])

    def get_state(self) -> np.ndarray:
        """Public accessor for current state (without noise/masking)."""
        return self._construct_state()

    def step(self, action: int) -> Tuple[float, np.ndarray, dict]:
        """
        Execute an action and advance the environment.

        Supports optional context switches and partial observability via masking
        and additive noise on the returned sensory state.

        Returns:
            reward: sampled reward
            new_state: sensory state after the action (possibly masked/noisy)
            info: metadata including observability and true_state
        """
        if action < 0 or action >= self.n_arms:
            raise ValueError(f"Action {action} out of bounds for {self.n_arms} arms")

        # Apply context switch if configured
        if (
            self.switch_step is not None
            and self.switched_reward_means is not None
            and self.current_step == self.switch_step
        ):
            print(f"\n🔄 CONTEXT SWITCH at t={self.current_step}")
            print(f"   Old means: {self.active_reward_means}")
            print(f"   New means: {self.switched_reward_means}")
            self.active_reward_means = self.switched_reward_means.copy()

        mean = self.active_reward_means[action]
        if self.reward_stds is None:
            reward = float(self.rng.binomial(1, mean))
        else:
            reward = float(self.rng.normal(loc=mean, scale=self.reward_stds[action]))

        # Update internal state before constructing observation
        self.last_action = action
        self.last_reward = reward
        self.timestep += 1
        self.current_step += 1

        true_state = self._construct_state()

        observable = self.rng.random() > self.obs_mask_prob
        if observable:
            noise = self.rng.normal(loc=0.0, scale=self.obs_noise_std, size=true_state.shape)
            obs_state = true_state + noise
            info = {
                "observable": True,
                "noise_level": self.obs_noise_std,
                "true_state": true_state.copy(),
            }
        else:
            obs_state = np.full_like(true_state, np.nan)
            info = {
                "observable": False,
                "noise_level": float("inf"),
                "true_state": true_state.copy(),
            }

        return reward, obs_state, info

    def get_optimal_arm(self) -> int:
        """Return index of arm with highest mean reward."""
        return int(np.argmax(self.reward_means))


# --------------------------------------------------------------------------- #
# Boss Battle Environment (multi-phase, delayed rewards)
# --------------------------------------------------------------------------- #
class BossBattleEnv:
    """
    Adversarial multi-phase environment for Scenario 7.

    Four phases of 200 steps each with distinct reward means, observability,
    noise, and optional delayed rewards in Phase 3.
    """

    def __init__(self, n_arms: int = 4, delay: int = 10, seed: int = 0, corruption_prob: float = 0.1) -> None:
        self.n_arms = n_arms
        self.delay = delay
        self.seed = seed
        self.corruption_prob = corruption_prob

        # Phase configuration
        self.phases: Dict[int, Dict] = {
            1: {"t_start": 0, "t_end": 200, "means": np.array([0.1, 0.3, 0.5, 0.7]), "mask_prob": 0.0, "sigma": 0.0, "delay": 0},
            2: {"t_start": 200, "t_end": 400, "means": np.array([0.7, 0.5, 0.3, 0.1]), "mask_prob": 0.3, "sigma": 0.3, "delay": 0},
            3: {"t_start": 400, "t_end": 600, "means": np.array([0.6, 0.4, 0.2, 0.1]), "mask_prob": 0.5, "sigma": 0.5, "delay": delay},
            4: {"t_start": 600, "t_end": 800, "means": np.array([0.2, 0.8, 0.1, 0.3]), "mask_prob": 0.1, "sigma": 0.1, "delay": 0},
        }

        self.rng = np.random.default_rng(self.seed)
        self.reward_std = 0.1  # fixed std for Gaussian rewards
        self.total_horizon = 800.0

        # Buffers for delayed rewards
        self.action_buffer: deque = deque(maxlen=max(1, delay + 1))

        self.current_step = 0
        self.last_action = -1
        self.last_reward = 0.0
        self.last_info: Dict = {"observable": True, "noise_level": 0.0}

    # ------------------------------------------------------------------ #
    def _get_current_phase(self) -> int:
        t = self.current_step
        if t < 200:
            return 1
        elif t < 400:
            return 2
        elif t < 600:
            return 3
        else:
            return 4

    def get_optimal_arm(self, phase: Optional[int] = None) -> int:
        ph = phase if phase is not None else self._get_current_phase()
        return int(np.argmax(self.phases[ph]["means"]))

    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.last_action = -1
        self.last_reward = 0.0
        self.action_buffer.clear()
        self.last_info = {"observable": True, "noise_level": 0.0}
        return self.get_state()

    def _construct_state(self) -> np.ndarray:
        # time normalized by total horizon
        time_feature = np.array([min(self.current_step / self.total_horizon, 1.0)], dtype=float)
        action_one_hot = np.zeros(self.n_arms + 1, dtype=float)
        action_one_hot[self.last_action + 1] = 1.0
        reward_feature = np.array([self.last_reward], dtype=float)
        return np.concatenate([time_feature, action_one_hot, reward_feature])

    def get_state(self) -> np.ndarray:
        return self._construct_state()

    def get_obs_info(self) -> Dict:
        """Return current observability parameters for the active phase."""
        phase = self._get_current_phase()
        return {"observable": True, "noise_level": self.phases[phase]["sigma"]}

    def step(self, action: int) -> Tuple[float, np.ndarray, Dict]:
        """
        Execute action with phase-dependent masking/noise and optional delayed reward.
        """
        phase = self._get_current_phase()
        cfg = self.phases[phase]

        # Determine which action yields reward (delay in phase 3)
        current_delay = cfg["delay"]
        if current_delay > 0 and len(self.action_buffer) >= current_delay:
            reward_action = self.action_buffer[-current_delay]
        else:
            reward_action = action

        self.action_buffer.append(action)

        mean = cfg["means"][reward_action]
        reward = float(self.rng.normal(loc=mean, scale=self.reward_std))

        # Optional corruption in phase 3
        if phase == 3 and self.rng.random() < self.corruption_prob:
            reward *= -0.5

        self.last_action = action
        self.last_reward = reward
        self.current_step += 1

        true_state = self._construct_state()
        observable = self.rng.random() > cfg["mask_prob"]
        if observable:
            noise = self.rng.normal(loc=0.0, scale=cfg["sigma"], size=true_state.shape)
            obs_state = true_state + noise
            info = {"observable": True, "noise_level": cfg["sigma"], "true_state": true_state.copy(), "phase": phase}
        else:
            obs_state = np.full_like(true_state, np.nan)
            info = {"observable": False, "noise_level": float("inf"), "true_state": true_state.copy(), "phase": phase}

        self.last_info = info
        return reward, obs_state, info
