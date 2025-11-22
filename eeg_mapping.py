"""
EEG proxy mappings for FBÇT variables.

This module provides theoretically grounded, falsifiable mappings from
FBÇT state variables (α_S, α_M, α_W, β, L_t, entropy) to EEG/ERP proxies.
Mappings draw on canonical findings:
- Frontal midline theta ~ uncertainty/conflict (Cavanagh & Frank, 2014)
- Posterior alpha ~ internal focus/memory retrieval (Klimesch, 2012)
- Frontal beta ~ confidence/stable control (Engel & Fries, 2010)
- P3 amplitude ~ context updating/surprise (Polich, 2007)
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    return float(1.0 / (1.0 + np.exp(-x)))


def compute_theta_power(entropy: float, L: float, beta: float) -> float:
    """
    Frontal midline theta (4-8 Hz) proxy.

    Rationale:
    - Increases with uncertainty (entropy ↑)
    - Increases when integration is weak (L low)
    - Increases when confidence/exploitation is low (β low)
    """
    baseline = 2.0
    unc_term = 3.0 * sigmoid(entropy - 0.5)
    integ_term = 2.0 * (1.0 - sigmoid(L - 0.5))
    expl_term = 1.5 * (1.0 - sigmoid(beta - 4.0))
    return float(np.clip(baseline + unc_term + integ_term + expl_term, 0.0, 10.0))


def compute_alpha_power(alpha_m: float, alpha_s: float, L: float) -> float:
    """
    Posterior alpha (8-12 Hz) proxy.

    Rationale:
    - Increases with memory reliance (α_M)
    - Decreases with sensory weighting (α_S)
    - Increases with better integration (L)
    """
    baseline = 2.0
    mem_term = 4.0 * alpha_m
    sens_term = -3.0 * alpha_s
    integ_term = 2.0 * sigmoid(L - 0.5)
    return float(np.clip(baseline + mem_term + sens_term + integ_term, 0.0, 10.0))


def compute_beta_power(beta_param: float, L: float, entropy: float) -> float:
    """
    Frontal beta (13-30 Hz) proxy.

    Rationale:
    - Increases with higher β (confidence/exploitation)
    - Increases with integration (L)
    - Decreases with uncertainty (entropy)
    """
    baseline = 1.5
    beta_term = 0.6 * beta_param  # β in [1,10] → contributes ~0.6..6
    integ_term = 1.5 * sigmoid(L - 0.5)
    unc_term = -2.0 * sigmoid(entropy - 0.4)
    return float(np.clip(baseline + beta_term + integ_term + unc_term, 0.0, 10.0))


def compute_gamma_power(L: float, alpha_s: float, alpha_m: float) -> float:
    """
    Gamma (30-80 Hz) proxy for integration/binding.
    """
    baseline = 0.5
    integ_term = 4.0 * sigmoid(L - 0.5)
    balance_term = 1.5 * (1.0 - abs(alpha_s - alpha_m))
    return float(np.clip(baseline + integ_term + balance_term, 0.0, 10.0))


def compute_p3_amplitude(delta_L: float, context_switch: bool) -> float:
    """
    P3 ERP amplitude proxy (~300-600 ms).

    Rationale:
    - Scales with surprise/prediction error (|ΔL|)
    - Boosted when context switch flag is true
    """
    base = 0.5
    surprise_term = 3.0 * sigmoid(abs(delta_L) - 0.1)
    switch_boost = 1.5 if context_switch else 0.0
    return float(np.clip(base + surprise_term + switch_boost, 0.0, 5.0))


def compute_theta_beta_ratio(theta_power: float, beta_power: float) -> float:
    """Theta/beta ratio commonly used as exploration index."""
    if beta_power <= 1e-6:
        return 5.0
    return float(np.clip(theta_power / beta_power, 0.0, 5.0))


def compute_delta_power(entropy: float, reward: float) -> float:
    """
    Optional delta (1-4 Hz) proxy tied to motivational salience.
    Simple mapping: increases with uncertainty and negative reward.
    """
    baseline = 0.5
    unc_term = 2.0 * sigmoid(entropy - 0.3)
    valence_term = 1.5 * sigmoid(-reward)  # negative reward → higher delta
    return float(np.clip(baseline + unc_term + valence_term, 0.0, 10.0))


def eeg_proxies_from_state(
    alpha_s: float,
    alpha_m: float,
    alpha_w: float,
    beta: float,
    L: float,
    entropy: float,
    reward: float = 0.0,
    context_switch: bool = False,
    delta_L: float = 0.0,
) -> Dict[str, float]:
    """
    Compute EEG proxies from FBÇT state variables.

    Args:
        alpha_s: Sensory weight [0,1]
        alpha_m: Memory weight [0,1]
        alpha_w: Value weight [0,1]
        beta: Temperature parameter (~1-10)
        L: Consciousness level (KL-based)
        entropy: Policy entropy
        reward: Immediate reward (for valence-related delta)
        context_switch: Whether a context switch just occurred
        delta_L: Change in L from previous timestep

    Returns:
        Dictionary of EEG proxies (theta/alpha/beta/gamma/P3/TBR/delta).
    """
    theta_power = compute_theta_power(entropy, L, beta)
    alpha_power = compute_alpha_power(alpha_m, alpha_s, L)
    beta_power = compute_beta_power(beta, L, entropy)
    gamma_power = compute_gamma_power(L, alpha_s, alpha_m)
    p3_amplitude = compute_p3_amplitude(delta_L, context_switch)
    theta_beta_ratio = compute_theta_beta_ratio(theta_power, beta_power)
    delta_power = compute_delta_power(entropy, reward)

    return {
        "theta_power": theta_power,
        "alpha_power": alpha_power,
        "beta_power": beta_power,
        "gamma_power": gamma_power,
        "p3_amplitude": p3_amplitude,
        "theta_beta_ratio": theta_beta_ratio,
        "delta_power": delta_power,
    }


def validate_eeg_proxies(proxies: Dict[str, float]) -> None:
    """
    Validate proxy ranges (basic sanity checks).
    """
    ranges = {
        "theta_power": (0.0, 10.0),
        "alpha_power": (0.0, 10.0),
        "beta_power": (0.0, 10.0),
        "gamma_power": (0.0, 10.0),
        "p3_amplitude": (0.0, 5.0),
        "theta_beta_ratio": (0.0, 5.0),
        "delta_power": (0.0, 10.0),
    }
    for k, (lo, hi) in ranges.items():
        v = proxies.get(k, 0.0)
        if not (lo - 1e-6 <= v <= hi + 1e-6):
            raise ValueError(f"Proxy {k} out of range [{lo},{hi}]: {v}")
