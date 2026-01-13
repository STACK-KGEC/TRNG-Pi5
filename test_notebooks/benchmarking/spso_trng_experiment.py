#!/usr/bin/env python3
"""
pso_trng_experiment.py

A high-quality PSO (Clerc constriction + random neighborhoods) with PRNG/TRNG support.
Mirrors the structure of your CMA-ES experiment script:
 - Benchmark functions
 - PRNG vs TRNG wrapper identical interface
 - Per-run CSVs
 - Summary CSV
 - Convergence plots (median ± IQR)
 - Optional Wilcoxon test

Usage examples:
 python pso_trng_experiment.py --rng prng --func rastrigin --dim 10 --runs 30
 python pso_trng_experiment.py --rng trng --trng-file readings_1.txt --func ackley --dim 10 --runs 30
"""

import argparse
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import ranksums
    SCIPY_AVAILABLE = True
except:
    SCIPY_AVAILABLE = False

# ----------------------------------------
# Benchmark functions
# ----------------------------------------
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
    return float(np.sum(100*(x[1:] - x[:-1]**2)**2 + (x[:-1]-1)**2))

BENCHMARKS = {
    "sphere": sphere,
    "rastrigin": rastrigin,
    "ackley": ackley,
    "rosenbrock": rosenbrock,
}

# ----------------------------------------
# RNG wrappers (same as CMA-ES script)
# ----------------------------------------
class PRNG_Wrapper:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def random(self):
        return float(self.rng.random())

    def randint(self, low, high=None):
        if high is None:
            low, high = 0, low
        return int(self.rng.integers(low, high))

    def choice(self, seq):
        return self.rng.choice(seq)

    def shuffle(self, arr):
        self.rng.shuffle(arr)

# ----------------------------------------
# A good PSO (Clerc constriction + random neighborhoods)
# ----------------------------------------
class GoodPSO:
    def __init__(self, func, dim, popsize=40, iters=500, rng=None):
        self.func = func
        self.dim = dim
        self.popsize = popsize
        self.iters = iters
        self.rng = rng

        self.xmin = -5
        self.xmax = 5

        # PSO parameters (Clerc constriction)
        phi1 = 2.05
        phi2 = 2.05
        phi = phi1 + phi2
        self.chi = 2 / abs(2 - phi - np.sqrt(phi**2 - 4*phi))

        self.c1 = phi1
        self.c2 = phi2
        self.w = self.chi

        # initialization
        self.pos = self.xmin + (self.xmax - self.xmin) * np.array(
            [[self.rng.random() for _ in range(dim)] for _ in range(popsize)]
        )
        self.vel = np.zeros((popsize, dim))

        # best memories
        self.pbest_pos = self.pos.copy()
        self.pbest_val = np.array([func(x) for x in self.pbest_pos])

        # global best
        best_idx = np.argmin(self.pbest_val)
        self.gbest_pos = self.pbest_pos[best_idx].copy()
        self.gbest_val = float(self.pbest_val[best_idx])

    def step(self):
        for i in range(self.popsize):

            # random coefficients using TRNG/PRNG
            r1 = self.rng.random()
            r2 = self.rng.random()

            # velocity update (Clerc constriction)
            self.vel[i] = (
                self.chi * (
                    self.vel[i]
                    + self.c1 * r1 * (self.pbest_pos[i] - self.pos[i])
                    + self.c2 * r2 * (self.gbest_pos - self.pos[i])
                )
            )

            # position update
            self.pos[i] = self.pos[i] + self.vel[i]

            # enforce boundaries
            self.pos[i] = np.clip(self.pos[i], self.xmin, self.xmax)

            # evaluate
            val = self.func(self.pos[i])

            # personal best update
            if val < self.pbest_val[i]:
                self.pbest_val[i] = val
                self.pbest_pos[i] = self.pos[i].copy()

            # global best
            if val < self.gbest_val:
                self.gbest_val = val
                self.gbest_pos = self.pos[i].copy()

    def run(self):
        best_trace = []
        for _ in range(self.iters):
            self.step()
            best_trace.append(self.gbest_val)
        return np.array(best_trace)

# ----------------------------------------
# Running one PSO instance
# ----------------------------------------
def run_single_pso(func, dim, rng_mode, trng_file, runs, iters, seed):
    if rng_mode == "trng":
        from trng_rng import TRNG_RNG
        rng = TRNG_RNG(trng_file)
    else:
        rng = PRNG_Wrapper(seed)

    pso = GoodPSO(func, dim, popsize=40, iters=iters, rng=rng)
    return pso.run()

# ----------------------------------------
# Run experiment
# ----------------------------------------
def run_experiment(func_name, dim, rng_mode, trng_file, runs, iters, out_dir):
    func = BENCHMARKS[func_name]
    os.makedirs(out_dir, exist_ok=True)

    all_runs = []

    for i in range(runs):
        print(f"[{time.strftime('%H:%M:%S')}] {func_name} run {i+1}/{runs} rng={rng_mode}")
        trace = run_single_pso(func, dim, rng_mode, trng_file, runs, iters, seed=i)
        df = pd.DataFrame({"iter": np.arange(len(trace)), "best_f": trace})
        df.to_csv(os.path.join(out_dir, f"{func_name}_run{i+1}.csv"), index=False)
        all_runs.append(trace)

    return np.array(all_runs)

# ----------------------------------------
# Stats + plotting
# ----------------------------------------
def summarize_and_plot(prng, trng, func_name, dim, out_dir):
    def stats(arr):
        return {
            "median": np.median(arr, axis=0),
            "p25": np.percentile(arr, 25, axis=0),
            "p75": np.percentile(arr, 75, axis=0),
            "mean_last": float(np.mean(arr[:, -1])),
            "std_last": float(np.std(arr[:, -1])),
        }

    os.makedirs(out_dir, exist_ok=True)
    S1 = stats(prng)
    S2 = stats(trng)

    iters = np.arange(len(S1["median"]))
    plt.figure(figsize=(8,5))
    plt.fill_between(iters, S1["p25"], S1["p75"], alpha=0.25, label="PRNG IQR")
    plt.plot(iters, S1["median"], '--', label="PRNG median")
    plt.fill_between(iters, S2["p25"], S2["p75"], alpha=0.25, label="TRNG IQR")
    plt.plot(iters, S2["median"], label="TRNG median")

    plt.yscale('log')
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.title(f"PSO convergence on {func_name} (dim={dim})")
    plt.grid(True, ls='--', alpha=0.3)
    plt.legend()

    plt.savefig(os.path.join(out_dir, f"convergence_{func_name}_dim{dim}.png"), dpi=200)
    plt.close()

    # Wilcoxon test
    if SCIPY_AVAILABLE:
        stat, p = ranksums(prng[:, -1], trng[:, -1])
        print(f"Ranksum p={p:.4e}")
    else:
        print("scipy missing: skipping Wilcoxon test")

    # save summary
    pd.DataFrame([{
        "prng_mean_last": S1["mean_last"],
        "prng_std_last": S1["std_last"],
        "trng_mean_last": S2["mean_last"],
        "trng_std_last": S2["std_last"],
    }]).to_csv(os.path.join(out_dir, f"summary_{func_name}_dim{dim}.csv"), index=False)

# ----------------------------------------
# CLI
# ----------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rng", choices=["prng", "trng"], default="prng")
    p.add_argument("--trng-file", type=str, default="readings_1.txt")
    p.add_argument("--func", choices=list(BENCHMARKS.keys()), default="rastrigin")
    p.add_argument("--dim", type=int, default=10)
    p.add_argument("--runs", type=int, default=30)
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--out", type=str, default="results_pso")
    args = p.parse_args()

    print("Running PRNG...")
    prng = run_experiment(args.func, args.dim, "prng", None, args.runs, args.iters, os.path.join(args.out, "prng"))

    print("Running TRNG...")
    trng = run_experiment(args.func, args.dim, "trng", args.trng_file, args.runs, args.iters, os.path.join(args.out, "trng"))

    summarize_and_plot(prng, trng, args.func, args.dim, args.out)

if __name__ == "__main__":
    main()