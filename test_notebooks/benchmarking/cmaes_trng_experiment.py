#!/usr/bin/env python3
"""
cmaes_trng_experiment.py

Run CMA-ES using either normal PRNG (numpy) or your TRNG_RNG as the randomness source.
Produces per-generation best traces for multiple runs and draws convergence plots (median ± IQR).
Per-run outputs are saved to CSV; summary statistics and Wilcoxon test are printed.

Dependencies:
 - numpy
 - matplotlib
 - pandas
 - scipy (for wilcoxon). If not available, script will still run but Wilcoxon will be skipped.
 - Your TRNG_RNG class must be in trng_rng.py or same file; adjust import accordingly.

Usage examples:
 python cmaes_trng_experiment.py --rng prng --func rastrigin --dim 10 --runs 30
 python cmaes_trng_experiment.py --rng trng --trng-file readings_1.txt --func ackley --dim 10 --runs 30
"""

import argparse
import os
import time
import csv
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# try to import Wilcoxon test
try:
    from scipy.stats import ranksums
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# If your TRNG_RNG class is in another file, import it accordingly.
# from trng_rng import TRNG_RNG
# For convenience, if TRNG_RNG is in the same file, ensure it is available in the namespace.

# -------------------------
# Benchmark functions
# -------------------------
def sphere(x):
    return float(np.sum(x**2))

def rastrigin(x, A=10):
    n = x.size
    return float(A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x)))

def ackley(x, a=20, b=0.2, c=2*np.pi):
    n = x.size
    s1 = np.sum(x**2)
    s2 = np.sum(np.cos(c * x))
    return float(-a * np.exp(-b * np.sqrt(s1 / n)) - np.exp(s2 / n) + a + np.e)

def rosenbrock(x):
    x = x.ravel()
    return float(np.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (x[:-1] - 1.0)**2.0))

BENCHMARKS = {
    "sphere": sphere,
    "rastrigin": rastrigin,
    "ackley": ackley,
    "rosenbrock": rosenbrock,
}

# -------------------------
# RNG wrappers
# -------------------------
class PRNG_Wrapper:
    """Wrap numpy RNG to provide same methods as TRNG_RNG interface used below."""
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def random(self):
        # return float in [0,1)
        return float(self.rng.random())

    def randbits(self, k):
        # return integer with k random bits
        # Use numpy's integers to emulate
        nbytes = (k + 7) // 8
        val = int.from_bytes(self.rng.integers(0, 256, size=nbytes, dtype=np.uint8).tobytes(), "big")
        return val & ((1 << k) - 1)

    def randint(self, low, high=None):
        if high is None:
            low, high = 0, low
        return int(self.rng.integers(low, high))

    def choice(self, seq):
        return self.rng.choice(seq)

    def shuffle(self, arr):
        self.rng.shuffle(arr)

# -------------------------
# Gaussian sampler using RNG.random() (Box-Muller)
# -------------------------
def randn_standard(rng, size):
    """
    Generate `size` standard normal samples using Box-Muller on top of rng.random().
    rng must implement .random() returning floats in [0,1).
    Returns a numpy array of shape (size,)
    """
    size = int(size)
    out = np.empty(size, dtype=float)
    i = 0
    while i < size:
        # generate two uniforms in (0,1]
        u1 = rng.random()
        u2 = rng.random()
        # numeric stability: avoid log(0)
        if u1 <= 0.0:
            u1 = 1e-16
        z0 = np.sqrt(-2.0 * np.log(u1)) * np.cos(2*np.pi * u2)
        z1 = np.sqrt(-2.0 * np.log(u1)) * np.sin(2*np.pi * u2)
        out[i] = z0
        if i+1 < size:
            out[i+1] = z1
        i += 2
    return out

# -------------------------
# Simple CMA-ES implementation (mu-lambda)
# -------------------------
class SimpleCMAES:
    def __init__(self, dim, popsize=None, sigma0=0.3, rng_wrapper=None):
        self.n = dim
        self.sigma = sigma0
        self.rng = rng_wrapper if rng_wrapper is not None else PRNG_Wrapper(0)

        # population size
        if popsize is None:
            self.lambda_ = 4 + int(3 * np.log(self.n))
        else:
            self.lambda_ = popsize
        self.mu = self.lambda_ // 2

        # strategy parameters
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu+1))
        self.weights = self.weights / np.sum(self.weights)
        self.mu_eff = 1.0 / np.sum(self.weights**2)

        # learning rates
        self.c_c = (4 + self.mu_eff / self.n) / (self.n + 4 + 2 * self.mu_eff / self.n)
        self.c_s = (self.mu_eff + 2) / (self.n + self.mu_eff + 5)
        self.c1 = 2 / ((self.n + 1.3)**2 + self.mu_eff)
        self.cmu = min(1 - self.c1, 2 * (self.mu_eff - 2 + 1/self.mu_eff) / ((self.n + 2)**2 + self.mu_eff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.n + 1)) - 1) + self.c_s

        # initialize dynamic state
        self.mean = np.zeros(self.n)
        self.p_c = np.zeros(self.n)
        self.p_s = np.zeros(self.n)
        self.B = np.eye(self.n)
        self.D = np.ones(self.n)
        self.C = np.eye(self.n)
        self.inv_sqrt_C = np.eye(self.n)
        self.eig_update_count = 0
        self.chi_n = np.sqrt(self.n) * (1 - 1/(4*self.n) + 1/(21*self.n**2))

    def ask(self):
        # sample lambda offspring from multivariate normal N(mean, sigma^2 C)
        zs = randn_standard(self.rng, self.lambda_ * self.n).reshape(self.lambda_, self.n)
        ys = (zs @ (self.B * self.D))  # this is incorrect dimensionally if not careful; use proper transform
        # Correct transform: X = mean + sigma * B @ (D * z)
        # We'll compute per-sample:
        xs = []
        for z in zs:
            y = self.B.dot(self.D * z)
            x = self.mean + self.sigma * y
            xs.append(x)
        return np.array(xs), zs

    def tell(self, xs, zs, fitnesses):
        # xs: (lambda, n), zs: (lambda, n), fitnesses: (lambda,)
        idx = np.argsort(fitnesses)
        x_selected = xs[idx[:self.mu]]
        z_selected = zs[idx[:self.mu]]
        # update mean: weighted recombination in z-space
        y_w = np.sum((self.weights.reshape(-1,1) * z_selected), axis=0)
        old_mean = self.mean.copy()
        self.mean = self.mean + self.sigma * self.B.dot(self.D * y_w)

        # update evolution paths
        inv_sqrt_C_times = self.inv_sqrt_C.dot(self.B.dot(self.D * y_w))
        self.p_s = (1 - self.c_s) * self.p_s + np.sqrt(self.c_s * (2 - self.c_s) * self.mu_eff) * inv_sqrt_C_times
        norm_p_s = np.linalg.norm(self.p_s)
        h_sigma = 1.0 if norm_p_s / np.sqrt(1 - (1 - self.c_s)**(2 * (1))) < (1.4 + 2/(self.n+1)) * self.chi_n else 0.0
        self.p_c = (1 - self.c_c) * self.p_c + h_sigma * np.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff) * self.B.dot(self.D * y_w)

        # rank-mu and rank-one update of covariance
        artmp = (1.0 / self.sigma) * (x_selected - old_mean)
        delta_h = (1 - h_sigma) * self.c1 * self.c_c * (2 - self.c_c)
        self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * np.outer(self.p_c, self.p_c) + delta_h * self.C
        for k in range(self.mu):
            wk = self.weights[k]
            z = z_selected[k]
            self.C += self.cmu * wk * np.outer(self.B.dot(self.D * z), self.B.dot(self.D * z))

        # adapt step-size
        self.sigma *= np.exp((self.c_s / self.damps) * (norm_p_s / self.chi_n - 1.0))

        # decompose C to update B and D occasionally
        self.eig_update_count += 1
        if self.eig_update_count >= max(1, int(1.0 / (self.c1 + self.cmu) / self.n / 10)):
            self.eig_update_count = 0
            # enforce symmetry
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            D2, B = np.linalg.eigh(self.C)
            D2 = np.where(D2 < 1e-20, 1e-20, D2)
            self.D = np.sqrt(D2)
            self.B = B
            self.inv_sqrt_C = B.dot(np.diag(1.0 / self.D)).dot(B.T)

    def best_of_population(self, xs, fitnesses):
        idx = np.argmin(fitnesses)
        return xs[idx].copy(), float(fitnesses[idx])

# -------------------------
# Experiment runner
# -------------------------
def run_single_cmaes(dim, func, budget_evals, rng_mode="prng", trng_file=None, seed=0):
    # rng_mode: "prng" or "trng"
    if rng_mode == "trng":
        if trng_file is None:
            raise ValueError("trng_file required for trng mode")
        from trng_rng import TRNG_RNG  # adjust path if needed
        rng = TRNG_RNG(trng_file)
    else:
        rng = PRNG_Wrapper(seed)

    cma = SimpleCMAES(dim=dim, sigma0=0.5, rng_wrapper=rng)
    evals = 0
    gen = 0
    best_trace = []  # store best fitness after each generation
    best_global = None

    # each generation uses lambda_ evaluations
    while evals < budget_evals:
        xs, zs = cma.ask()
        fitnesses = np.array([func(x) for x in xs])
        evals += xs.shape[0]
        cma.tell(xs, zs, fitnesses)
        best_x, best_f = cma.best_of_population(xs, fitnesses)
        if best_global is None or best_f < best_global:
            best_global = best_f
        best_trace.append(best_global)
        gen += 1
    return np.array(best_trace)

def run_experiment(func_name, dim, rng_mode, trng_file, runs, budget_evals, out_dir):
    func = BENCHMARKS[func_name]
    all_runs = []
    os.makedirs(out_dir, exist_ok=True)
    for run_i in range(runs):
        print(f"[{time.strftime('%H:%M:%S')}] Running {func_name} dim={dim} run {run_i+1}/{runs} rng={rng_mode}")
        trng_file_arg = trng_file if rng_mode == "trng" else None
        seed = run_i  # deterministic seed for PRNG baseline
        trace = run_single_cmaes(dim, func, budget_evals, rng_mode=rng_mode, trng_file=trng_file_arg, seed=seed)
        # save per-run CSV
        df = pd.DataFrame({"gen": np.arange(len(trace)), "best_f": trace})
        fname = os.path.join(out_dir, f"{func_name}_dim{dim}_rng-{rng_mode}_run{run_i+1}.csv")
        df.to_csv(fname, index=False)
        all_runs.append(trace)
    # align runs to same length (they will by design)
    all_runs = np.array(all_runs)
    return all_runs

# -------------------------
# Stats and plotting
# -------------------------
def summarize_and_plot(all_runs_prng, all_runs_trng, func_name, dim, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    def stats_from_runs(all_runs):
        # all_runs: (runs, gens)
        median = np.median(all_runs, axis=0)
        p25 = np.percentile(all_runs, 25, axis=0)
        p75 = np.percentile(all_runs, 75, axis=0)
        mean = np.mean(all_runs[:, -1])
        std = np.std(all_runs[:, -1])
        return {"median": median, "p25": p25, "p75": p75, "mean_last": mean, "std_last": std}

    s_pr = stats_from_runs(all_runs_prng)
    s_tr = stats_from_runs(all_runs_trng)

    # plot median ± IQR
    gens = np.arange(len(s_pr["median"]))
    plt.figure(figsize=(8,5))
    plt.fill_between(gens, s_pr["p25"], s_pr["p75"], alpha=0.25, label="PRNG IQR")
    plt.plot(gens, s_pr["median"], label="PRNG median", linestyle='--')
    plt.fill_between(gens, s_tr["p25"], s_tr["p75"], alpha=0.25, label="TRNG IQR")
    plt.plot(gens, s_tr["median"], label="TRNG median", linestyle='-')
    plt.yscale('log')  # optional: often fitness spans orders
    plt.xlabel("Generation")
    plt.ylabel("Best fitness (log scale)")
    plt.title(f"CMA-ES convergence on {func_name} (dim={dim})")
    plt.legend()
    plt.grid(True, which="both", ls="--", lw=0.3)
    plotfile = os.path.join(out_dir, f"convergence_{func_name}_dim{dim}.png")
    plt.savefig(plotfile, dpi=200)
    plt.close()
    print(f"Saved convergence plot to {plotfile}")

    # compute Wilcoxon/ranksum on final bests
    final_pr = all_runs_prng[:, -1]
    final_tr = all_runs_trng[:, -1]
    print(f"PRNG final bests: mean={np.mean(final_pr):.6e}, std={np.std(final_pr):.6e}")
    print(f"TRNG final bests: mean={np.mean(final_tr):.6e}, std={np.std(final_tr):.6e}")
    if SCIPY_AVAILABLE:
        stat, p = ranksums(final_pr, final_tr)
        print(f"Ranksum test statistic={stat:.4f}, p-value={p:.4e}")
    else:
        print("scipy not available: skipping ranksum test (install scipy for p-values).")

    # save summary CSV
    summary = {
        "prng_mean_last": s_pr["mean_last"],
        "prng_std_last": s_pr["std_last"],
        "trng_mean_last": s_tr["mean_last"],
        "trng_std_last": s_tr["std_last"],
    }
    summary_fname = os.path.join(out_dir, f"summary_{func_name}_dim{dim}.csv")
    pd.DataFrame([summary]).to_csv(summary_fname, index=False)
    print(f"Saved summary to {summary_fname}")

# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rng", choices=["prng", "trng"], default="prng", help="RNG mode")
    parser.add_argument("--trng-file", type=str, default="readings_1.txt", help="TRNG bit file (if rng==trng)")
    parser.add_argument("--func", choices=list(BENCHMARKS.keys()), default="rastrigin")
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--budget-evals", type=int, default=10000)
    parser.add_argument("--out", type=str, default="results_cmaes")
    args = parser.parse_args()

    # Run both PRNG and TRNG for direct comparison
    print("Running PRNG baseline...")
    prng_runs = run_experiment(args.func, args.dim, "prng", None, args.runs, args.budget_evals, os.path.join(args.out, "prng"))
    print("Running TRNG (this may be slower if TRNG throughput is low)...")
    trng_runs = run_experiment(args.func, args.dim, "trng", args.trng_file, args.runs, args.budget_evals, os.path.join(args.out, "trng"))

    # Summarize and plot
    summarize_and_plot(prng_runs, trng_runs, args.func, args.dim, args.out)

def delete(path):
    import os
    import shutil

    folder_path = path  # Replace with the actual path to your folder

    # Option 1: Delete all files and subdirectories, then recreate the folder (if needed)
    # shutil.rmtree(folder_path)
    # os.makedirs(folder_path)

    # Option 2: Delete only files within the folder, leaving subdirectories intact
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path) # Deletes subdirectories and their contents
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
if __name__ == "__main__":
    main()
    delete("TRNG-Pi5/test_notebooks/benchmarking/out_ackley/prng")
    delete("TRNG-Pi5/test_notebooks/benchmarking/out_ackley/trng")
    