"""
EEG prediction pipeline for FBÇT scenarios.

Runs existing scenarios, maps FBÇT variables to EEG proxies, and exports:
- results/eeg_proxies_raw.csv : timestep-level data
- results/eeg_proxies_summary.csv : aggregated stats
- results/eeg_summary.tex : LaTeX table
- results/eeg_mapping_summary.md : Markdown report
- plots/eeg_*.png : publication-ready figures
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from eeg_mapping import eeg_proxies_from_state
from simulation import (
    scenario_1_basic_learning,
    scenario_2_context_switch,
    scenario_3_high_uncertainty,
    scenario_4_variance_comparison,
    scenario_5_partial_observability,
    scenario_6_noise_ramp,
    run_scenario_7_boss_battle,
    run_experiment,
)


def _compute_entropy(pi: np.ndarray) -> float:
    pi = np.asarray(pi, dtype=float)
    return float(-np.sum(pi * np.log(pi + 1e-10)))


def collect_from_trajectories(
    scenario: int,
    trajectories: List[Dict],
    condition: str = "default",
    phase_by_step: List[int] | None = None,
    context_switch_steps: List[int] | None = None,
) -> pd.DataFrame:
    """
    Convert trajectories to a tidy DataFrame with EEG proxies.
    """
    rows: List[Dict] = []
    for run_id, traj in enumerate(trajectories):
        actions = traj["actions"]
        alphas = np.array(traj["alphas"])
        betas = np.array(traj["betas"])
        Ls = np.array(traj["L_values"])
        rewards = np.array(traj["rewards"])
        policies = np.array(traj["policies"])
        T = len(actions)
        phases = phase_by_step if phase_by_step is not None else [0] * T

        for t in range(T):
            alpha_s, alpha_m, alpha_w = alphas[t]
            beta = betas[t]
            L = Ls[t]
            entropy = _compute_entropy(policies[t])
            delta_L = 0.0 if t == 0 else Ls[t] - Ls[t - 1]
            ctx_switch = False
            if context_switch_steps is not None:
                ctx_switch = t in context_switch_steps
            proxies = eeg_proxies_from_state(
                alpha_s=alpha_s,
                alpha_m=alpha_m,
                alpha_w=alpha_w,
                beta=beta,
                L=L,
                entropy=entropy,
                reward=rewards[t],
                context_switch=ctx_switch,
                delta_L=delta_L,
            )
            rows.append(
                {
                    "scenario": scenario,
                    "condition": condition,
                    "phase": phases[t],
                    "run": run_id,
                    "t": t,
                    "alpha_s": alpha_s,
                    "alpha_m": alpha_m,
                    "alpha_w": alpha_w,
                    "beta": beta,
                    "L": L,
                    "entropy": entropy,
                    "reward": rewards[t],
                    "context_switch": ctx_switch,
                    "delta_L": delta_L,
                    **proxies,
                }
            )
    return pd.DataFrame(rows)


def run_single_scenario_with_eeg(scenario_id: int, n_runs: int = 3) -> pd.DataFrame:
    """
    Run one scenario and return a dataframe of EEG proxies.
    """
    if scenario_id == 1:
        trajs = scenario_1_basic_learning()
        return collect_from_trajectories(1, trajs, condition="baseline")
    if scenario_id == 2:
        trajs = scenario_2_context_switch()
        ctx_steps = [400]
        return collect_from_trajectories(2, trajs, condition="switch", context_switch_steps=ctx_steps)
    if scenario_id == 3:
        trajs = scenario_3_high_uncertainty()
        return collect_from_trajectories(3, trajs, condition="high_uncertainty")
    if scenario_id == 4:
        low, high = scenario_4_variance_comparison()
        df_low = collect_from_trajectories(4, low, condition="low_variance")
        df_high = collect_from_trajectories(4, high, condition="high_variance")
        return pd.concat([df_low, df_high], ignore_index=True)
    if scenario_id == 5:
        # Partial and full need separate runs
        env_partial = {
            "n_arms": 4,
            "reward_means": np.array([0.2, 0.4, 0.6, 0.8]),
            "reward_stds": np.array([0.1, 0.1, 0.1, 0.1]),
            "obs_mask_prob": 0.35,
            "obs_noise_std": 0.3,
        }
        env_full = {
            "n_arms": 4,
            "reward_means": np.array([0.2, 0.4, 0.6, 0.8]),
            "reward_stds": np.array([0.1, 0.1, 0.1, 0.1]),
            "obs_mask_prob": 0.0,
            "obs_noise_std": 0.0,
        }
        agent_cfg = {"n_arms": 4, "initial_alpha": np.array([0.33, 0.34, 0.33]), "initial_beta": 4.0}
        traj_partial = run_experiment(env_partial, agent_cfg, n_steps=800, n_runs=n_runs, save_plots=False)
        traj_full = run_experiment(env_full, agent_cfg, n_steps=800, n_runs=n_runs, save_plots=False)
        df_partial = collect_from_trajectories(5, traj_partial, condition="partial_obs")
        df_full = collect_from_trajectories(5, traj_full, condition="full_obs")
        return pd.concat([df_partial, df_full], ignore_index=True)
    if scenario_id == 6:
        results = scenario_6_noise_ramp()
        dfs = []
        for sigma, trajs in results.items():
            dfs.append(collect_from_trajectories(6, trajs, condition=f"sigma_{sigma}"))
        return pd.concat(dfs, ignore_index=True)
    if scenario_id == 7:
        trajs = run_scenario_7_boss_battle(n_runs=n_runs, T=800, delay=10)
        # phases already logged; use them
        phase_by_step = trajs[0]["phases"] if trajs else None
        return collect_from_trajectories(7, trajs, condition="boss", phase_by_step=phase_by_step, context_switch_steps=[200, 400, 600])
    raise ValueError(f"Unknown scenario id: {scenario_id}")


def aggregate_all_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate by scenario/condition/phase (mean/std for key metrics).
    """
    grouped = df.groupby(["scenario", "condition", "phase"])
    stats = grouped.agg(
        L_mean=("L", "mean"),
        L_std=("L", "std"),
        H_mean=("entropy", "mean"),
        H_std=("entropy", "std"),
        theta_mean=("theta_power", "mean"),
        theta_std=("theta_power", "std"),
        alpha_mean=("alpha_power", "mean"),
        alpha_std=("alpha_power", "std"),
        beta_mean=("beta_power", "mean"),
        beta_std=("beta_power", "std"),
        p3_mean=("p3_amplitude", "mean"),
        p3_std=("p3_amplitude", "std"),
        tbr_mean=("theta_beta_ratio", "mean"),
        tbr_std=("theta_beta_ratio", "std"),
        reward_mean=("reward", "mean"),
        reward_std=("reward", "std"),
    )
    return stats.reset_index()


def generate_latex_table(summary: pd.DataFrame, path: str) -> None:
    """
    Generate a compact LaTeX table without requiring jinja2 (avoids optional pandas deps).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = ["scenario", "condition", "phase", "L_mean", "H_mean", "theta_mean", "alpha_mean", "beta_mean", "p3_mean"]
    hdr = ["Scenario", "Condition", "Phase", "$L_t$", "$H$", "Theta", "Alpha", "Beta", "P3"]
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{FBÇT $\\to$ EEG Predictions Across Scenarios}")
    lines.append("\\label{tab:eeg-predictions}")
    lines.append("\\begin{tabular}{llrrrrrrrr}")
    lines.append("\\toprule")
    lines.append(" & ".join(hdr) + " \\\\")
    lines.append("\\midrule")
    for _, row in summary[cols].iterrows():
        lines.append(
            f"{int(row['scenario'])} & {row['condition']} & {int(row['phase'])} & "
            f"{row['L_mean']:.3f} & {row['H_mean']:.3f} & {row['theta_mean']:.3f} & "
            f"{row['alpha_mean']:.3f} & {row['beta_mean']:.3f} & {row['p3_mean']:.3f} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved LaTeX table: {path}")


def generate_markdown_report(summary: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = ["# FBÇT → EEG Prediction Mapping Summary", ""]
    for _, row in summary.iterrows():
        lines.append(
            f"- Scenario {int(row['scenario'])}, condition={row['condition']}, phase={int(row['phase'])}: "
            f"L={row['L_mean']:.3f}, H={row['H_mean']:.3f}, "
            f"Theta={row['theta_mean']:.2f}, Alpha={row['alpha_mean']:.2f}, Beta={row['beta_mean']:.2f}, P3={row['p3_mean']:.2f}"
        )
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved markdown summary: {path}")


def plot_eeg_scenario_comparison(summary: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Take phase-aggregated means (phase mean per scenario/condition)
    agg = summary.groupby(["scenario", "condition"]).mean(numeric_only=True).reset_index()
    scenarios = agg["scenario"].astype(int)
    x = np.arange(len(agg))
    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, agg["theta_mean"], width, label="Theta")
    ax.bar(x, agg["alpha_mean"], width, label="Alpha")
    ax.bar(x + width, agg["beta_mean"], width, label="Beta")
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{int(s)}-{c}" for s, c in zip(agg["scenario"], agg["condition"])], rotation=45, ha="right")
    ax.set_ylabel("Power (a.u.)")
    ax.set_title("EEG proxies across scenarios")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_noise_ramp(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    subset = df[df["scenario"] == 6].copy()
    subset["sigma"] = subset["condition"].str.replace("sigma_", "").astype(float)
    grouped = subset.groupby("sigma").mean(numeric_only=True).reset_index()
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(grouped["sigma"], grouped["theta_power"], label="Theta", color="purple")
    ax1.plot(grouped["sigma"], grouped["alpha_power"], label="Alpha", color="green")
    ax1.plot(grouped["sigma"], grouped["beta_power"], label="Beta", color="orange")
    ax1.set_xlabel("Noise σ")
    ax1.set_ylabel("Power (a.u.)")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(grouped["sigma"], grouped["L"], label="L_t", color="blue", linestyle="--")
    ax2.set_ylabel("L_t")
    fig.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_boss_battle_phases(summary: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    boss = summary[(summary["scenario"] == 7) & (summary["condition"] == "boss")]
    boss = boss.sort_values("phase")
    phases = boss["phase"].astype(int)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(phases - 0.2, boss["theta_mean"], width=0.2, label="Theta")
    ax.bar(phases, boss["alpha_mean"], width=0.2, label="Alpha")
    ax.bar(phases + 0.2, boss["beta_mean"], width=0.2, label="Beta")
    ax.plot(phases, boss["p3_mean"], color="red", marker="o", label="P3 (line)")
    ax.set_xlabel("Phase")
    ax.set_ylabel("Power (a.u.)")
    ax.set_title("Boss Battle EEG Proxies by Phase")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_context_switch_timeseries(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    subset = df[df["scenario"] == 2]
    run0 = subset[subset["run"] == 0]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(run0["t"], run0["L"], label="L_t", color="purple")
    ax.plot(run0["t"], run0["theta_power"], label="Theta", color="orange")
    ax.plot(run0["t"], run0["p3_amplitude"], label="P3", color="green")
    ax.axvline(400, color="red", linestyle="--", label="Context switch")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Value")
    ax.set_title("Context Switch Dynamics (Scenario 2)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_partial_obs_comparison(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sub = df[df["scenario"] == 5]
    agg = sub.groupby("condition").mean(numeric_only=True)
    labels = agg.index.tolist()
    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, agg["theta_power"], width, label="Theta")
    ax.bar(x, agg["alpha_power"], width, label="Alpha")
    ax.bar(x + width, agg["beta_power"], width, label="Beta")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Power (a.u.)")
    ax.set_title("Partial vs Full Observability EEG Proxies")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def main() -> None:
    all_data = []
    for scenario_id in range(1, 8):
        data = run_single_scenario_with_eeg(scenario_id, n_runs=3)
        all_data.append(data)
    df = pd.concat(all_data, ignore_index=True)

    os.makedirs("results", exist_ok=True)
    df.to_csv("results/eeg_proxies_raw.csv", index=False)
    print("Saved: results/eeg_proxies_raw.csv")

    summary = aggregate_all_scenarios(df)
    summary.to_csv("results/eeg_proxies_summary.csv", index=False)
    print("Saved: results/eeg_proxies_summary.csv")

    generate_latex_table(summary, "results/eeg_summary.tex")
    generate_markdown_report(summary, "results/eeg_mapping_summary.md")

    plot_eeg_scenario_comparison(summary, "plots/eeg_scenario_comparison.png")
    plot_noise_ramp(df, "plots/eeg_noise_ramp.png")
    plot_boss_battle_phases(summary, "plots/eeg_boss_battle_phases.png")
    plot_context_switch_timeseries(df, "plots/eeg_context_switch_timeseries.png")
    plot_partial_obs_comparison(df, "plots/eeg_partial_obs_comparison.png")


if __name__ == "__main__":
    main()
