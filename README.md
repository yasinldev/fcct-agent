# FCCT Agent and Quantum-FCCT Simulations

This repository contains implementations and analysis utilities for the Functional Consciousness Collapse Theory (FCCT), including:
- FCCT agent for multi-armed bandit environments (`fbct_agent.py`, `simulation.py`, `envs.py`, `analysis.py`)
- EEG prediction mapping and analysis (`eeg_mapping.py`, `run_eeg_analysis.py`)
- Quantum-FCCT measurement simulator with single- and two-stage collapse tests (`quantum_fbct_sim.py`)

## Getting Started
1. Create a virtual environment and install dependencies (numpy, pandas, matplotlib, scipy).
2. Run the full FCCT scenario suite:
   ```bash
   python simulation.py
   ```
3. Generate EEG proxy outputs and plots:
   ```bash
   python run_eeg_analysis.py
   ```
4. Run quantum collapse simulations (Born recovery, context influence, two-stage collapse):
   ```bash
   python quantum_fbct_sim.py
   ```

Outputs are written to `results/` (CSV, LaTeX, markdown) and `plots/` (PNG figures).

## Components
- `simulation.py`: Scenarios 1–7 for the FCCT agent, with logging and analysis plots.
- `analysis.py`: Plotting and validation utilities for trajectories.
- `envs.py`: Bandit environments, including the BossBattleEnv for multi-phase stress tests.
- `eeg_mapping.py` and `run_eeg_analysis.py`: Map FCCT state variables to EEG/ERP proxies and produce aggregated reports.
- `quantum_fbct_sim.py`: Probabilistic quantum collapse simulations, including two-stage compositional tests.
- `README_EEG.md`: Detailed guidance for EEG proxy usage and outputs.

## Reference / Citation
If you use this code or the associated predictions in your research, please cite:

Özkaya, M. Y. (2025). Functional Consciousness Collapse Theory: A Computational Framework. arXiv preprint.
