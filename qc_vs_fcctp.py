"""
QUANTUM vs FCCT (or fbçt in turkish doesn't matter) SUPERPOSITION COMPARISON
========================================
Full test suite including Deutsch-Jozsa and Grover
For validation purposes - not all will go in paper
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import math

# =========================================================
# QUANTUM ENGINE
# =========================================================

class QuantumEngine:
    def __init__(self, n_qubits: int = 3):
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        
    def hadamard_matrix(self, n: int) -> np.ndarray:
        H1 = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        H_n = H1
        for _ in range(n - 1):
            H_n = np.kron(H_n, H1)
        return H_n
    
    def oracle_constant_0(self, state: np.ndarray) -> np.ndarray:
        return state.copy()
    
    def oracle_constant_1(self, state: np.ndarray) -> np.ndarray:
        return -state
    
    def oracle_balanced(self, state: np.ndarray) -> np.ndarray:
        oracle_state = state.copy()
        for x in range(self.dim):
            if x % 2 == 1:
                oracle_state[x] *= -1
        return oracle_state
    
    def run(self, oracle_type: str) -> Tuple[np.ndarray, np.ndarray]:
        H = self.hadamard_matrix(self.n_qubits)
        initial = np.zeros(self.dim)
        initial[0] = 1.0
        superposition = H @ initial
        
        if oracle_type == "constant_0":
            after_oracle = self.oracle_constant_0(superposition)
        elif oracle_type == "constant_1":
            after_oracle = self.oracle_constant_1(superposition)
        elif oracle_type == "balanced":
            after_oracle = self.oracle_balanced(superposition)
        else:
            raise ValueError(f"Unknown oracle: {oracle_type}")
        
        final_amplitudes = H @ after_oracle
        probabilities = np.abs(final_amplitudes) ** 2
        
        return final_amplitudes, probabilities

# =========================================================
# FBÇT ENGINE
# =========================================================

class FBCTEngine:
    def __init__(self, n_bits: int = 3):
        self.n_bits = n_bits
        self.dim = 2 ** n_bits
        
    def initialize_amplitudes(self) -> np.ndarray:
        return np.ones(self.dim, dtype=float) / np.sqrt(self.dim)

    def apply_oracle_phase(self, amplitudes: np.ndarray, oracle_type: str) -> np.ndarray:
        out = amplitudes.copy()
        for x in range(self.dim):
            if oracle_type == "constant_0":
                continue
            elif oracle_type == "constant_1":
                out[x] *= -1
            elif oracle_type == "balanced":
                if x % 2 == 1:
                    out[x] *= -1
            else:
                raise ValueError(f"Unknown oracle: {oracle_type}")
        return out

    def hadamard_matrix(self) -> np.ndarray:
        H1 = np.array([[1, 1], [1, -1]], dtype=float) / np.sqrt(2)
        H = H1
        for _ in range(self.n_bits - 1):
            H = np.kron(H, H1)
        return H

    def fbct_interference(self, amplitudes: np.ndarray) -> np.ndarray:
        H = self.hadamard_matrix()
        return H @ amplitudes

    def amplitudes_to_beliefs(self, amplitudes: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        probs = amplitudes ** 2
        probs = np.maximum(probs, 0.0)
        probs /= probs.sum() + eps
        return probs
    
    def collapse(self, beliefs: np.ndarray, alpha_S: float = 1.0, temperature: float = 0.1) -> np.ndarray:
        M = np.ones(self.dim) / self.dim
        W = np.ones(self.dim) / self.dim
        S = beliefs
        
        combined = alpha_S * S + (1-alpha_S) * 0.5 * (M + W)
        combined /= combined.sum()
        
        if temperature > 0:
            collapsed = np.exp(combined / temperature)
            collapsed /= collapsed.sum()
        else:
            collapsed = np.zeros(self.dim)
            collapsed[np.argmax(combined)] = 1.0
        
        return collapsed

    def run_dj(self, oracle_type: str, alpha_S: float = 1.0, temperature: float = 0.1) -> np.ndarray:
        a = self.initialize_amplitudes()
        a = self.apply_oracle_phase(a, oracle_type)
        a = self.fbct_interference(a)
        beliefs = self.amplitudes_to_beliefs(a)
        collapsed = self.collapse(beliefs, alpha_S=alpha_S, temperature=temperature)
        return collapsed

# =========================================================
# GROVER FUNCTIONS
# =========================================================

def hadamard_n(n_bits: int) -> np.ndarray:
    H1 = np.array([[1, 1], [1, -1]], dtype=float) / np.sqrt(2)
    H = H1
    for _ in range(n_bits - 1):
        H = np.kron(H, H1)
    return H

def grover_quantum(n_bits: int = 3, marked: str = "101", n_iter: int = None) -> Dict:
    dim = 2 ** n_bits
    H = hadamard_n(n_bits)
    init = np.zeros(dim, dtype=complex)
    init[0] = 1.0
    psi = H @ init
    s = np.ones(dim, dtype=complex) / np.sqrt(dim)

    def oracle(amp: np.ndarray) -> np.ndarray:
        out = amp.copy()
        idx = int(marked, 2)
        out[idx] *= -1
        return out

    def diffusion(amp: np.ndarray) -> np.ndarray:
        proj = np.vdot(s, amp) * s
        return 2 * proj - amp

    if n_iter is None:
        n_iter = int(round((math.pi / 4) * math.sqrt(dim)))

    history = []
    for k in range(n_iter):
        prob_marked = float(np.abs(psi[int(marked, 2)]) ** 2)
        history.append((k, prob_marked))
        psi = oracle(psi)
        psi = diffusion(psi)

    prob_marked = float(np.abs(psi[int(marked, 2)]) ** 2)
    history.append((n_iter, prob_marked))
    probs = np.abs(psi) ** 2
    probs /= probs.sum()
    return {"amplitudes": psi, "probs": probs, "history": history, "iterations": n_iter}

def fbct_grover(
    n_bits: int = 3,
    marked: str = "101",
    n_iter: int | None = None,
    evidence_gain: float = 2.0,
    interference_strength: float = 0.8,
    temperature: float = 0.05,
) -> Dict:
    dim = 2**n_bits
    idx_marked = int(marked, 2)
    
    # Initialize: uniform AMPLITUDE superposition
    H = hadamard_n(n_bits)
    init = np.zeros(dim)
    init[0] = 1.0
    a = H @ init  # Real amplitudes
    
    if n_iter is None:
        n_iter = int(round((math.pi / 4) * math.sqrt(dim)))
    
    def oracle(amp: np.ndarray) -> np.ndarray:
        """Sign flip on marked state (AMPLITUDE space!)"""
        out = amp.copy()
        out[idx_marked] *= -1
        return out
    
    def diffusion(amp: np.ndarray) -> np.ndarray:
        """Grover diffusion operator (AMPLITUDE space!)"""
        mean_val = np.mean(amp)
        out = 2 * mean_val - amp
        return out
    
    history = []
    for k in range(n_iter):
        # Record probability (for plotting)
        prob_marked = float(a[idx_marked] ** 2)
        history.append((k, prob_marked))
        
        # Oracle: sign flip
        a = oracle(a)
        
        # Diffusion: inversion about mean
        a = diffusion(a)
        
        # Normalize (drift control, NOT collapse!)
        norm = np.linalg.norm(a)
        if norm > 0:
            a = a / norm
    
    # FINAL: Born rule (NO temperature sharpening!)
    prob_marked = float(a[idx_marked] ** 2)
    history.append((n_iter, prob_marked))
    
    probs = a ** 2
    probs = np.clip(probs, 0.0, None)
    probs /= probs.sum()
    
    return {"probs": probs, "history": history, "iterations": n_iter}

def grover_metrics(q_probs: np.ndarray, f_probs: np.ndarray) -> Dict:
    eps = 1e-12
    kl = float(np.sum(q_probs * np.log((q_probs + eps) / (f_probs + eps))))
    l1 = float(np.sum(np.abs(q_probs - f_probs)))
    l2 = float(np.sqrt(np.sum((q_probs - f_probs) ** 2)))
    cos = float(np.dot(q_probs, f_probs) / (np.linalg.norm(q_probs) * np.linalg.norm(f_probs) + eps))
    match = int(np.argmax(q_probs) == np.argmax(f_probs))
    return {"kl": kl, "l1": l1, "l2": l2, "cos": cos, "max_match": match}

def run_grover_comparison():
    print("\nGrover comparison:")
    n_bits = 3
    marked = "101"

    q = grover_quantum(n_bits=n_bits, marked=marked)
    fb = fbct_grover(n_bits=n_bits, marked=marked)

    metrics = grover_metrics(q["probs"], fb["probs"])

    # Plot marked prob vs iteration
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot([h[0] for h in q["history"]], [h[1] for h in q["history"]], label="Quantum", marker="o")
    plt.plot([h[0] for h in fb["history"]], [h[1] for h in fb["history"]], label="FBCT", marker="s")
    plt.xlabel("Iteration")
    plt.ylabel(f"P({marked})")
    plt.title("Grover Amplification")
    plt.legend()
    plt.grid(alpha=0.3)

    # Plot final distributions
    plt.subplot(1, 2, 2)
    x = np.arange(2**n_bits)
    width = 0.35
    plt.bar(x - width/2, q["probs"], width, label="Quantum", alpha=0.7)
    plt.bar(x + width/2, fb["probs"], width, label="FBCT", alpha=0.7)
    plt.xticks(x, [format(i, f"0{n_bits}b") for i in x], rotation=45)
    plt.ylabel("Probability")
    plt.title("Final Distribution")
    plt.legend()
    plt.grid(alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig("grover_comparison.png", dpi=300)
    plt.close()

    print(f"Quantum P(marked)={q['probs'][int(marked,2)]:.4f}")
    print(f"FBCT   P(marked)={fb['probs'][int(marked,2)]:.4f}")
    print(f"Metrics: {metrics}")
    print("Plot saved: grover_comparison.png")

# =========================================================
# DEUTSCH-JOZSA COMPARISON
# =========================================================

def compare_engines(oracle_type: str = "balanced", alpha_S: float = 1.0, temperature: float = 0.1, n_qubits: int = 3) -> Dict:
    qe = QuantumEngine(n_qubits)
    q_amps, q_probs = qe.run(oracle_type)
    
    fe = FBCTEngine(n_qubits)
    f_probs = fe.run_dj(oracle_type, alpha_S, temperature)
    
    kl_div = np.sum(q_probs * np.log((q_probs + 1e-12) / (f_probs + 1e-12)))
    l1_dist = np.sum(np.abs(q_probs - f_probs))
    l2_dist = np.sqrt(np.sum((q_probs - f_probs)**2))
    cos_sim = np.dot(q_probs, f_probs) / (np.linalg.norm(q_probs) * np.linalg.norm(f_probs))
    
    print(f"Oracle {oracle_type}: L1={l1_dist:.6f}, KL={kl_div:.6f}, Cos={cos_sim:.6f}")
    return {'oracle': oracle_type, 'quantum': q_probs, 'fbct': f_probs, 'l1': l1_dist}

def visualize_comparison(results: List[Dict], save_path: str = None):
    fig, axes = plt.subplots(len(results), 2, figsize=(12, 4*len(results)))
    
    if len(results) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, result in enumerate(results):
        q_probs = result['quantum']
        f_probs = result['fbct']
        
        axes[idx, 0].bar(range(len(q_probs)), q_probs, alpha=0.7, color='blue')
        axes[idx, 0].set_title(f"Quantum: {result['oracle']}")
        axes[idx, 0].set_ylabel("Probability")
        axes[idx, 0].grid(alpha=0.3, axis='y')
        
        axes[idx, 1].bar(range(len(f_probs)), f_probs, alpha=0.7, color='green')
        axes[idx, 1].set_title(f"FBÇT: {result['oracle']}")
        axes[idx, 1].set_ylabel("Probability")
        axes[idx, 1].grid(alpha=0.3, axis='y')
        
        metrics_text = f"L1: {result['l1']:.3f}"
        axes[idx, 1].text(0.95, 0.95, metrics_text, transform=axes[idx, 1].transAxes,
                         verticalalignment='top', horizontalalignment='right',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved: {save_path}")
    plt.show()

# =========================================================
# MAIN
# =========================================================

def main():
    # PART 1: Deutsch-Jozsa
    print("PART 1: Deutsch-Jozsa")
    
    oracles = ["constant_0", "constant_1", "balanced"]
    all_results = []
    
    for oracle in oracles:
        result = compare_engines(oracle_type=oracle, alpha_S=1.0, temperature=0.1)
        all_results.append(result)
    
    visualize_comparison(all_results, save_path="deutsch_jozsa_comparison.png")
    
    # PART 2: Grover
    print("\nPART 2: Grover")
    
    run_grover_comparison()
    
    # Summary
    print("\nSUMMARY")
    for r in all_results:
        print(f"DJ {r['oracle']:15s}: L1={r['l1']:.4f}")
    print("Plots saved: deutsch_jozsa_comparison.png, grover_comparison.png")

if __name__ == "__main__":
    main()
