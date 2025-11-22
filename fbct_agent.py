"""
FBÇT Agent implementation.

This module provides a minimal but faithful realization of the Functional
Consciousness Collapse Theory (FBÇT) for a tabular multi-armed bandit setting.
All major theoretical components are explicit and instrumented for analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Memory State
# --------------------------------------------------------------------------- #
@dataclass
class MemoryState:
    """
    Structured memory storage M_t.

    Attributes:
        max_episodes: maximum number of episodic entries to retain (FIFO).
        n_arms: number of actions/arms.
        episodes: list of (action, reward, timestep) tuples.
        value_estimates: running mean reward estimates per arm.
        visit_counts: number of times each arm has been selected.
        uncertainty: simple uncertainty proxy per arm (decays with visits).
    """

    max_episodes: int = 100
    n_arms: int = 4

    episodes: List[Tuple[int, float, int]] = field(default_factory=list)
    value_estimates: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=float))
    visit_counts: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=float))
    uncertainty: np.ndarray = field(default_factory=lambda: np.ones(4, dtype=float))

    def __post_init__(self) -> None:
        # Ensure arrays reflect current n_arms
        self.value_estimates = np.zeros(self.n_arms, dtype=float)
        self.visit_counts = np.zeros(self.n_arms, dtype=float)
        self.uncertainty = np.ones(self.n_arms, dtype=float)

    def add_episode(self, action: int, reward: float, timestep: int) -> None:
        """Add new experience to episodic memory with FIFO eviction."""
        self.episodes.append((action, reward, timestep))
        if len(self.episodes) > self.max_episodes:
            self.episodes.pop(0)

    def update_value_estimate(self, action: int, reward: float) -> None:
        """Update incremental mean and uncertainty for the chosen action."""
        self.visit_counts[action] += 1.0
        count = self.visit_counts[action]

        # Incremental mean
        delta = reward - self.value_estimates[action]
        self.value_estimates[action] += delta / count

        # Uncertainty shrinks with more evidence; stays bounded away from 0
        self.uncertainty[action] = 1.0 / np.sqrt(count + 1e-6)

    def get_familiarity(self, action: int) -> float:
        """Return normalized familiarity score (0-1) based on visit counts."""
        max_count = max(self.visit_counts.max(), 1.0)
        return float(self.visit_counts[action] / max_count)

    def get_uncertainty(self, action: int) -> float:
        """Return current uncertainty proxy for an action."""
        return float(self.uncertainty[action])


# --------------------------------------------------------------------------- #
# FBÇT Agent
# --------------------------------------------------------------------------- #
class FBCTAgent:
    """
    FBÇT Agent implementing functional consciousness collapse theory.

    State components:
        - S_t : sensory state (provided by environment)
        - M_t : MemoryState instance
        - W_t : value/priority vector
        - α_t : context weights [α_S, α_M, α_W]
        - β_t : inverse temperature controlling policy sharpness
    """

    def __init__(
        self,
        n_arms: int,
        state_dim: int,
        initial_W: Optional[np.ndarray] = None,
        initial_alpha: Optional[np.ndarray] = None,
        initial_beta: float = 4.0,
        seed: int = 0,
        obs_noise_std: float = 0.0,
        baseline_policy: Optional[np.ndarray] = None,
    ) -> None:
        self.n_arms = n_arms
        self.state_dim = state_dim
        self.obs_noise_std = float(obs_noise_std)

        self.M = MemoryState(n_arms=n_arms, max_episodes=200)
        self.W = initial_W if initial_W is not None else np.ones(n_arms, dtype=float) / n_arms
        self.alpha = initial_alpha if initial_alpha is not None else np.array([0.33, 0.34, 0.33])
        self.beta = initial_beta

        # Baseline policy reference P_t (empirical if provided, else uniform)
        if baseline_policy is not None:
            P = np.array(baseline_policy, dtype=float)
            P = P / (P.sum() + 1e-12)
            self.P_baseline = P
        else:
            self.P_baseline = np.ones(n_arms, dtype=float) / n_arms
        self.timestep = 0
        self.history: List[Dict] = []

        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # Component Score Functions
    # ------------------------------------------------------------------ #
    def _unpack_state(self, S: np.ndarray) -> Tuple[int, float]:
        """
        Decode compact sensory features:
        returns (last_action, last_reward).
        """
        # S structure: [time, one_hot(n_arms+1), last_reward, context...]
        action_one_hot = S[1 : self.n_arms + 2]
        last_action = int(np.argmax(action_one_hot) - 1)
        last_reward = float(S[self.n_arms + 2])
        return last_action, last_reward

    def compute_f_S(self, action: int, S: np.ndarray, info: Optional[Dict] = None) -> float:
        """
        Sensory fitness: how well an action fits immediate sensory evidence.

        Heuristic:
        - If last reward was positive, repeating the action is attractive.
        - If last reward was negative, switching away is encouraged.
        - Early in training (low timestep feature) modest bonus to novelty.
        """
        info = info or {"observable": True, "noise_level": 0.0}

        # Strict handling: if masked or corrupted, no sensory evidence.
        if not info.get("observable", True) or np.isnan(S).any():
            return 0.0

        last_action, last_reward = self._unpack_state(S)
        time_feature = float(S[0])

        novelty_bonus = 0.0
        if last_action != -1 and action != last_action:
            novelty_bonus = 0.1 * (1.0 - min(time_feature, 1.0))

        if last_action == -1:
            base = 0.0  # no sensory prior yet
        elif action == last_action:
            base = 0.4 * last_reward
        else:
            base = -0.2 * last_reward

        noise_level = info.get("noise_level", 0.0)
        if noise_level > 0:
            confidence = float(np.exp(-2.0 * noise_level))  # stronger penalty
            base *= confidence
            novelty_bonus *= confidence

        # Additional noise reliability scaling using agent-level noise_std (for ramp)
        reliability = 1.0 / (1.0 + 3.0 * (self.obs_noise_std ** 2))
        base *= reliability
        novelty_bonus *= reliability

        return float(base + novelty_bonus)

    def compute_f_M(self, action: int, M: MemoryState) -> float:
        """
        Memory compatibility: familiarity and past success.
        """
        familiarity = M.get_familiarity(action)
        value_est = M.value_estimates[action]
        uncertainty = M.get_uncertainty(action)

        # Known-good actions are boosted; uncertainty slightly penalized.
        f_M = familiarity * value_est - 0.5 * uncertainty
        return float(f_M)

    def compute_f_W(self, action: int, W: np.ndarray) -> float:
        """
        Value alignment: expected utility based on W.
        """
        return float(W[action])

    def compute_unified_score(
        self,
        action: int,
        S: np.ndarray,
        M: MemoryState,
        W: np.ndarray,
        alpha: np.ndarray,
        info: Optional[Dict] = None,
    ) -> float:
        """
        Unified score f(x) = α_S f_S + α_M f_M + α_W f_W.
        """
        f_S = self.compute_f_S(action, S, info)
        f_M = self.compute_f_M(action, M)
        f_W = self.compute_f_W(action, W)
        return float(alpha[0] * f_S + alpha[1] * f_M + alpha[2] * f_W)

    def compute_all_scores(
        self,
        S: np.ndarray,
        M: MemoryState,
        W: np.ndarray,
        alpha: np.ndarray,
        info: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Compute per-action scores and component breakdowns.
        """
        f_S_all = np.array(
            [self.compute_f_S(a, S, info) for a in range(self.n_arms)], dtype=float
        )
        f_M_all = np.array([self.compute_f_M(a, M) for a in range(self.n_arms)], dtype=float)
        f_W_all = np.array([self.compute_f_W(a, W) for a in range(self.n_arms)], dtype=float)

        scores = alpha[0] * f_S_all + alpha[1] * f_M_all + alpha[2] * f_W_all
        components = {"f_S": f_S_all, "f_M": f_M_all, "f_W": f_W_all}
        return scores, components

    # ------------------------------------------------------------------ #
    # Candidate Generation and Policy
    # ------------------------------------------------------------------ #
    def generate_candidates(self, S: np.ndarray, M: MemoryState, W: np.ndarray) -> np.ndarray:
        """
        Base measure μ_t. We bias slightly toward uncertain actions to
        encourage exploration even before temperature modulation.
        """
        base = np.ones(self.n_arms, dtype=float) / self.n_arms
        uncertainty_bonus = M.uncertainty / (np.sum(M.uncertainty) + 1e-9)
        mu_t = 0.8 * base + 0.2 * uncertainty_bonus
        mu_t /= mu_t.sum()
        return mu_t

    def compute_policy(
        self,
        S: np.ndarray,
        M: MemoryState,
        W: np.ndarray,
        alpha: np.ndarray,
        beta: float,
        info: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Policy π_t(x) ∝ exp(β f(x)) μ_t(x).
        """
        mu_t = self.generate_candidates(S, M, W)
        scores, components = self.compute_all_scores(S, M, W, alpha, info)
        effective_beta = beta
        if info is not None and info.get("observable") is False:
            effective_beta = beta * 0.5  # force more exploration when blind

        logits = effective_beta * scores + np.log(mu_t + 1e-10)

        logits_max = logits.max()
        exp_logits = np.exp(logits - logits_max)
        pi_t = exp_logits / exp_logits.sum()

        details = {"scores": scores, "components": components, "mu_t": mu_t, "logits": logits}
        return pi_t, details

    # ------------------------------------------------------------------ #
    # Collapse and Consciousness Metrics
    # ------------------------------------------------------------------ #
    def sample_collapse(self, pi_t: np.ndarray) -> int:
        """Sample action from policy (collapse)."""
        return int(self.rng.choice(self.n_arms, p=pi_t))

    def compute_L(self, pi_t: np.ndarray, P_baseline: Optional[np.ndarray] = None) -> float:
        """
        Consciousness level: pure KL(π_t || P_t) as in the theory.

        L_t = Σ_x π_t(x) [log π_t(x) - log P_t(x)]
        """
        eps = 1e-12
        pi = np.clip(np.array(pi_t, dtype=float), eps, 1.0)
        P = np.array(self.P_baseline if P_baseline is None else P_baseline, dtype=float)
        P = np.clip(P / (P.sum() + eps), eps, 1.0)
        return float(np.sum(pi * (np.log(pi) - np.log(P))))

    def compute_consciousness_signature(
        self,
        pi_t: np.ndarray,
        scores: np.ndarray,
        components: Dict[str, np.ndarray],
        info: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """
        Extended metrics capturing decisiveness (H), integration (I),
        differentiation (D), and baseline KL (L).
        """
        L_val = self.compute_L(pi_t)
        H = float(-np.sum(pi_t * np.log(pi_t + 1e-10)))

        component_matrix = np.stack(
            [components["f_S"], components["f_M"], components["f_W"]], axis=0
        )
        I = float(1.0 / (1.0 + np.std(component_matrix, axis=0).mean()))
        D = float(np.std(scores))

        return {"L": L_val, "H": H, "I": I, "D": D}

    # ------------------------------------------------------------------ #
    # Learning Updates
    # ------------------------------------------------------------------ #
    def update_memory(self, action: int, reward: float, S: np.ndarray) -> None:
        """Update episodic buffer and value statistics."""
        self.M.add_episode(action, reward, self.timestep)
        self.M.update_value_estimate(action, reward)

    def update_W(
        self,
        action: int,
        reward: float,
        S: np.ndarray,
        observability_history: Optional[List[bool]] = None,
    ) -> None:
        """
        Update value/priority vector W_t -> W_{t+1}.

        Simple delta rule blending reward feedback with slow decay toward uniform.
        """
        learning_rate = 0.1
        decay = 0.97

        target = np.zeros(self.n_arms, dtype=float)
        target[action] = max(reward, 0.0)  # only positive rewards push priorities

        self.W = decay * self.W + learning_rate * target

        # Partial observability noise correction: dampen W when observations are scarce
        if observability_history is not None and len(observability_history) > 0:
            obs_rate = float(np.mean(observability_history[-50:]))
            noise_scale = max(0.0, 1.0 - obs_rate)
            self.W = self.W * (1.0 - noise_scale)

        self.W = self.W / (self.W.sum() + 1e-9)

    def update_alpha(
        self, pi_t: Optional[np.ndarray], reward_history: List[float], observability_history: Optional[List[bool]] = None
    ) -> None:
        """
        Update context weights α_t based on recent volatility and outcomes.
        """
        if len(reward_history) < 10:
            return

        recent_rewards = np.array(reward_history[-10:], dtype=float)
        reward_var = float(np.var(recent_rewards))
        reward_mean = float(np.mean(recent_rewards))

        obs_rate = 1.0
        if observability_history is not None and len(observability_history) >= 10:
            obs_rate = float(np.mean(observability_history[-10:]))

        # Base path: reward/variance driven
        if reward_var > 0.1:
            base_target = np.array([0.5, 0.3, 0.2], dtype=float)
        elif reward_mean > 0.6:
            base_target = np.array([0.2, 0.6, 0.2], dtype=float)
        else:
            base_target = np.array([0.2, 0.2, 0.6], dtype=float)

        # Observability/noise path (separate)
        noise = self.obs_noise_std
        noise_quality = 1.0 / (1.0 + 3.0 * (noise ** 2))
        effective_obs = obs_rate * noise_quality
        if effective_obs < 0.7:
            obs_target = np.array([0.15, 0.60, 0.25], dtype=float)
            blend = 0.6
        elif effective_obs < 0.9:
            obs_target = np.array([0.25, 0.45, 0.30], dtype=float)
            blend = 0.35
        else:
            obs_target = None
            blend = 0.0

        if obs_target is not None:
            target_alpha = (1.0 - blend) * base_target + blend * obs_target
        else:
            target_alpha = base_target

        # Final explicit noise regularizer to enforce α_S ↓, α_M ↑ with σ
        reliability = 1.0 / (1.0 + 4.0 * (self.obs_noise_std ** 2))
        bias = 0.6 * (1.0 - reliability)  # 0 when σ=0, up to ~0.6 when σ≈1
        target_alpha = target_alpha.copy()
        target_alpha[0] *= max(0.0, 1.0 - bias)
        target_alpha[1] *= 1.0 + bias
        # α_W unchanged
        target_alpha = target_alpha / (target_alpha.sum() + 1e-9)

        # Learning rate: faster when effective observability is low
        if effective_obs < 0.7:
            alpha_lr = 0.20
        elif reward_var > 0.1:
            alpha_lr = 0.10
        else:
            alpha_lr = 0.05

        self.alpha = (1 - alpha_lr) * self.alpha + alpha_lr * target_alpha
        self.alpha = self.alpha / self.alpha.sum()

    def update_beta(self, reward_history: List[float], observability_history: Optional[List[bool]] = None) -> None:
        """
        Adapt inverse temperature β_t using reward variance.
        """
        if len(reward_history) < 10:
            return

        # Compute recent observability and variance over a moderate window
        obs_rate = 1.0
        if observability_history is not None and len(observability_history) >= 20:
            obs_rate = float(np.mean(observability_history[-60:]))
        window_rewards = reward_history[-60:] if len(reward_history) >= 60 else reward_history
        reward_var = float(np.var(window_rewards))

        # Variance rule
        if reward_var > 0.10:
            beta_var = 3.5
        elif reward_var > 0.03:
            beta_var = 4.5
        else:
            beta_var = 6.0

        # Effective observability incorporating noise level
        noise = self.obs_noise_std
        noise_quality = 1.0 / (1.0 + 6.0 * (noise ** 2))
        effective_obs = obs_rate * noise_quality

        # Observability rule using effective_obs
        if effective_obs < 0.60:
            beta_obs = 2.8
        elif effective_obs < 0.75:
            beta_obs = 3.5
        elif effective_obs < 0.90:
            beta_obs = 4.5
        else:
            beta_obs = 6.0

        # Combine: obs dominates when effective_obs is low
        if effective_obs < 0.70:
            w_obs = 0.85
        elif effective_obs < 0.90:
            w_obs = 0.35
        else:
            w_obs = 0.25
        target_beta = w_obs * beta_obs + (1.0 - w_obs) * beta_var

        # Stronger drop under heavy masking (POMDP)
        if obs_rate < 0.55:
            target_beta = min(target_beta, 3.0)

        lr = 0.17
        self.beta = (1 - lr) * self.beta + lr * target_beta
        self.beta = float(np.clip(self.beta, 1.5, 10.0))

    def update(
        self,
        S: np.ndarray,
        action: int,
        reward: float,
        new_S: np.ndarray,
        reward_history: List[float],
        observability_history: Optional[List[bool]] = None,
    ) -> None:
        """
        Full update F(M, W, α, β | S, C, r).
        """
        self.update_memory(action, reward, S)
        self.update_W(action, reward, S, observability_history)
        self.update_alpha(None, reward_history, observability_history)
        self.update_beta(reward_history, observability_history)
        # Context-change rapid adaptation: if recent reward mean drops sharply, force re-exploration
        if len(reward_history) >= 60:
            recent_short = np.mean(reward_history[-15:])
            recent_long = np.mean(reward_history[-60:-30]) if len(reward_history) >= 90 else np.mean(reward_history[-60:])
            if recent_long > 0 and recent_short < recent_long - 0.2:
                # Forget some W, lower beta, lean sensory to recover
                self.W = 0.9 * self.W + 0.1 * (np.ones(self.n_arms, dtype=float) / self.n_arms)
                self.W = self.W / (self.W.sum() + 1e-9)
                self.beta = min(self.beta, 3.5)
                reset_alpha = np.array([0.35, 0.45, 0.20], dtype=float)
                self.alpha = 0.7 * self.alpha + 0.3 * reset_alpha
                self.alpha = self.alpha / (self.alpha.sum() + 1e-9)
        self.timestep += 1

    # ------------------------------------------------------------------ #
    # Self-reporting
    # ------------------------------------------------------------------ #
    def compute_confidence(self, pi_t: np.ndarray, action: int) -> float:
        """
        Confidence as probability margin between top two actions.
        """
        if self.n_arms < 2:
            return float(pi_t[action])
        sorted_probs = np.sort(pi_t)
        margin = sorted_probs[-1] - sorted_probs[-2]
        return float(margin)

    def explain_choice(
        self, action: int, S: np.ndarray, M: MemoryState, components: Dict[str, np.ndarray]
    ) -> str:
        """
        Produce a compact natural-language explanation of a decision.
        """
        f_S = components["f_S"][action]
        f_M = components["f_M"][action]
        f_W = components["f_W"][action]

        dominant_component = int(np.argmax(self.alpha))
        component_names = ["sensory", "memory", "value"]
        dominant_name = component_names[dominant_component]

        explanation = (
            f"Chose arm {action}. "
            f"Components: f_S={f_S:.2f}, f_M={f_M:.2f}, f_W={f_W:.2f}. "
            f"Decision was {dominant_name}-driven (α={self.alpha[dominant_component]:.2f})."
        )
        return explanation
