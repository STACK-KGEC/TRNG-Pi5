#!/usr/bin/env python3
"""
lshade_trng_experiment.py

Run L-SHADE (a modern adaptive Differential Evolution) using either NumPy PRNG or your TRNG_RNG.
Produces per-generation best traces for multiple runs and draws convergence plots (median ± IQR).
Per-run outputs are saved to CSV; summary statistics and Wilcoxon (ranksum) test are printed.

Dependencies:
 - numpy
 - matplotlib
 - pandas
 - scipy (for ranksums). If not available, script still runs.
 - Your TRNG_RNG class should be in trng_rng.py or be importable as trng_rng.TRNG_RNG.

Usage:
 python lshade_trng_experiment.py --rng prng --func rastrigin --dim 10 --runs 30
 python lshade_trng_experiment.py --rng trng --trng-file readings_1.txt --func ackley --dim 10 --runs 30
"""
import argparse
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# try to import Wilcoxon (ranksum)
try:
    from scipy.stats import ranksums
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# -------------------------
# Benchmark functions (same as CMA-ES file)
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
# RNG wrappers (PRNG & TRNG should have same minimal API)
# -------------------------
class PRNG_Wrapper:
    """Wrap numpy RNG to provide same methods as TRNG_RNG interface used below."""
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def random(self):
        return float(self.rng.random())

    def randint(self, low, high=None):
        if high is None:
            low, high = 0, low
        return int(self.rng.integers(low, high))

    def choice(self, seq):
        # return element chosen uniformly
        return seq[self.rng.integers(0, len(seq))]

    def shuffle(self, arr):
        self.rng.shuffle(arr)

# -------------------------
# L-SHADE Implementation (compact, readable)
# -------------------------
class LSHADE:
    """
    Lightweight L-SHADE implementation with core components:
    - current-to-pbest/1 mutation
    - adaptive memory of F and CR (success-history)
    - archive of replaced solutions
    - linear population size reduction
    Note: This is a research-quality compact implemention suitable for experiments,
    not a hyper-optimized production implementation.
    """
    def __init__(self, dim, popsize=None, rng=None, max_evals=10000, min_pop=4):
        self.n = dim
        self.max_evals = max_evals
        self.min_pop = min_pop
        self.rng = rng if rng is not None else PRNG_Wrapper(0)

        # initial population size (recommended: 18 * dim for L-SHADE papers, but we keep moderate)
        if popsize is None:
            self.NP0 = max(4, int(18 * self.n))
        else:
            self.NP0 = popsize
        self.NP = self.NP0
        self.archive = []  # store replaced solutions (list of vectors)
        self.eval_count = 0

        # memory for CR and F (size H)
        self.H = max(3, int(np.round(5)))  # small memory
        self.M_CR = np.full(self.H, 0.5)
        self.M_F = np.full(self.H, 0.5)
        self.mem_idx = 0

        # population arrays
        # initialize uniformly in [-5, 5] by default (we will use problem-specific bounds in runner)
        self.X = None
        self.F_vals = None

        # linear population reduction params
        self.min_pop = max(self.min_pop, 4)
        self.evals_at_last_reduction = 0

    def initialize(self, pop_init, fitness_init):
        """Set population and fitness arrays from caller (caller chooses bounds)."""
        self.X = np.array(pop_init, dtype=float)  # shape (NP, n)
        self.F_vals = np.array(fitness_init, dtype=float)
        self.NP = self.X.shape[0]
        self.eval_count = len(fitness_init)

    def _rand_index_excluding(self, exclude_set, upto):
        """Return a random integer in [0, upto) not in exclude_set."""
        if isinstance(exclude_set, (list, tuple, set)):
            exclude = set(exclude_set)
        else:
            exclude = {exclude_set}
        while True:
            r = self.rng.randint(0, upto)
            if r not in exclude:
                return r

    def _sample_cr_f(self):
        """Sample CR and F arrays for the current population based on memory (vectorized conceptually)."""
        # For each individual we will sample an index from [0, H)
        idx = max(0, int(self.rng.randint(0, self.H)))
        # But L-SHADE samples CR from normal and F from Cauchy centered at memory entry
        # We'll produce them per individual on ask-time.
        return idx

    def ask(self, bounds, p_best_rate=0.1):
        """
        Produce offspring population using current-to-pbest/1 mutation and binomial crossover.
        bounds: tuple (lower, upper) each can be scalar or array-like of length n.
        Returns:
            children: np.array shape (NP, n)
            children_F: list of F used for each trial
            children_CR: list of CR used for each trial
            parents_idx: indices used (for bookkeeping)
        """
        lower, upper = bounds
        lower = np.full(self.n, lower) if np.isscalar(lower) else np.array(lower, dtype=float)
        upper = np.full(self.n, upper) if np.isscalar(upper) else np.array(upper, dtype=float)

        NP = self.NP
        pop = self.X
        fitness = self.F_vals
        children = np.empty_like(pop)
        children_F = np.empty(NP, dtype=float)
        children_CR = np.empty(NP, dtype=float)
        trial_z = np.empty_like(pop)  # for potential archive usage

        # compute p-best indices set size (p-best chosen per individual)
        p_num = max(2, int(np.round(p_best_rate * NP)))
        # keep sorted fitness indices
        sorted_idx = np.argsort(fitness)

        for i in range(NP):
            # --- sample memory index
            mem_i = int(self.rng.randint(0, self.H))
            mu_CR = self.M_CR[mem_i]
            mu_F = self.M_F[mem_i]

            # sample CR ~ Normal(mu_CR, 0.1) clipped to [0,1]
            cr = np.random.normal(mu_CR, 0.1) if isinstance(self.rng, PRNG_Wrapper) else (mu_CR + 0.1*(self.rng.random()-0.5))
            # using PRNG for local normal is okay; but to strictly use rng.random(),
            # one can approximate normal with Box-Muller; for brevity, clip and use above.
            # Ensure values in [0,1]
            cr = float(np.clip(cr, 0.0, 1.0))

            # sample F from Cauchy(mu_F, 0.1) as in SHADE; resample until >0
            # We'll implement a simple Cauchy using inverse transform on uniform u in (0,1)
            # Cauchy(mu, gamma): mu + gamma * tan(pi*(u-0.5))
            attempts = 0
            fval = 0.0
            while True:
                u = self.rng.random()
                f_candidate = mu_F + 0.1 * np.tan(np.pi * (u - 0.5))
                if f_candidate > 0:
                    fval = min(1.0, f_candidate)
                    break
                attempts += 1
                if attempts > 10:  # fallback
                    fval = mu_F
                    break

            # choose r1,r2 from population U {0..NP-1}\{i}
            r1 = self._rand_index_excluding(i, NP)
            # for r2 we allow selecting from combined population + archive
            combined_size = NP + len(self.archive)
            # pick r2 index in combined; if in archive, pick from archive
            r2_idx = self.rng.randint(0, combined_size)
            use_archive = False
            if r2_idx < NP:
                r2 = r2_idx
            else:
                use_archive = True
                r2 = r2_idx - NP
            # choose pbest index uniformly from the top p_num
            pbest_idx = sorted_idx[self.rng.randint(0, p_num)]
            x_i = pop[i]
            x_r1 = pop[r1]
            x_r2 = (self.archive[r2] if use_archive else pop[r2])

            # mutation: current-to-pbest/1 -> v = x_i + F*(x_pbest - x_i) + F*(x_r1 - x_r2)
            v = x_i + fval * (pop[pbest_idx] - x_i) + fval * (x_r1 - x_r2)

            # binomial crossover to produce trial u
            u_trial = np.empty(self.n, dtype=float)
            jrand = self.rng.randint(0, self.n)
            for j in range(self.n):
                if j == jrand or self.rng.random() < cr:
                    u_trial[j] = v[j]
                else:
                    u_trial[j] = x_i[j]

            # boundary handling: simple bounce-back / clamp
            u_trial = np.where(u_trial < lower, (lower + x_i) / 2.0, u_trial)
            u_trial = np.where(u_trial > upper, (upper + x_i) / 2.0, u_trial)

            children[i] = u_trial
            children_F[i] = fval
            children_CR[i] = cr
            trial_z[i] = v  # store mutant for covariance if needed

        return children, children_F, children_CR

    def tell(self, children, children_F, children_CR, children_fitness):
        """
        Selection and memory update.
        children: (NP, n)
        children_F: (NP,)
        children_CR: (NP,)
        children_fitness: (NP,)
        """
        pop = self.X
        fitness = self.F_vals
        NP = self.NP

        # prepare lists to compute successful F and CR
        success_F = []
        success_CR = []
        wS = []

        # selection: if child better than parent, replace and store parent to archive
        for i in range(NP):
            if children_fitness[i] <= fitness[i]:
                # success
                success_F.append(children_F[i])
                success_CR.append(children_CR[i])
                w = abs(fitness[i] - children_fitness[i])  # improvement magnitude
                wS.append(w)
                # add replaced parent to archive
                if len(self.archive) < 5 * self.NP0:
                    self.archive.append(pop[i].copy())
                else:
                    # rotate archive to keep size bounded
                    idx_remove = int(self.rng.randint(0, len(self.archive)))
                    self.archive[idx_remove] = pop[i].copy()
                # replace
                pop[i] = children[i]
                fitness[i] = children_fitness[i]
            # else parent remains

        # update memory M_F and M_CR if there were successful updates
        if len(success_F) > 0:
            wS = np.array(wS) + 1e-16
            wS = wS / np.sum(wS)
            # Lehmer mean-style for F: sum(w * F^2)/sum(w*F)
            F_sq = np.array(success_F)**2
            new_MF = np.sum(wS * F_sq) / np.sum(wS * np.array(success_F))
            new_MCR = np.sum(wS * np.array(success_CR))
            # if new_MCR is NaN (all zeros), keep old
            if np.isnan(new_MCR):
                new_MCR = self.M_CR[self.mem_idx]
            if np.isnan(new_MF):
                new_MF = self.M_F[self.mem_idx]

            # store into circular memory
            self.M_CR[self.mem_idx] = new_MCR
            self.M_F[self.mem_idx] = new_MF
            self.mem_idx = (self.mem_idx + 1) % self.H

        # update internal arrays
        self.X = pop
        self.F_vals = fitness

        # optionally reduce population size linearly based on evaluations
        # here we do a simple schedule: when evals surpass fractions of max, reduce
        # calculate target NP based on evaluations
        # linear from NP0 to min_pop across max_evals
        target_NP = int(np.round(self.NP0 - (self.eval_count / max(1.0, self.max_evals)) * (self.NP0 - self.min_pop)))
        target_NP = max(self.min_pop, target_NP)
        if target_NP < self.NP:
            # reduce population: remove worst individuals to reach target
            idx_sorted = np.argsort(self.F_vals)  # ascending fitness
            survivors = idx_sorted[:target_NP]
            self.X = self.X[survivors]
            self.F_vals = self.F_vals[survivors]
            self.NP = target_NP
            # shrink archive if necessary
            if len(self.archive) > 2 * self.NP:
                self.archive = self.archive[:2 * self.NP]

    def best_of_population(self):
        idx = np.argmin(self.F_vals)
        return self.X[idx].copy(), float(self.F_vals[idx])

# -------------------------
# Runner (mirrors CMA-ES runner style)
# -------------------------
def run_single_lshade(dim, func, budget_evals, rng_mode="prng", trng_file=None, seed=0, bounds=(-5,5)):
    """
    Run one independent LSHADE instance until budget_evals is consumed.
    Returns: per-generation best fitness trace (1D np.array)
    """
    # setup RNG
    if rng_mode == "trng":
        if trng_file is None:
            raise ValueError("trng_file required for trng mode")
        from trng_rng import TRNG_RNG
        rng = TRNG_RNG(trng_file)
    else:
        rng = PRNG_Wrapper(seed)

    # initial population NP0
    NP0 = max(4, int(18 * dim))
    # initialize population uniformly inside bounds
    low, high = bounds
    X0 = []
    F0 = []
    for i in range(NP0):
        # use rng.random() to build a vector
        vec = np.array([low + (high - low) * rng.random() for _ in range(dim)], dtype=float)
        X0.append(vec)
        F0.append(func(vec))
    lshade = LSHADE(dim=dim, popsize=NP0, rng=rng, max_evals=budget_evals)
    lshade.initialize(X0, F0)
    lshade.eval_count = len(F0)

    best_trace = []
    best_global = None

    # each generation does NP function evaluations (approx). Run until budget exhausted.
    while lshade.eval_count < budget_evals:
        children, children_F, children_CR = lshade.ask(bounds)
        # evaluate children
        children_fitness = np.array([func(c) for c in children], dtype=float)
        lshade.eval_count += children.shape[0]
        lshade.tell(children, children_F, children_CR, children_fitness)
        bx, bf = lshade.best_of_population()
        if best_global is None or bf < best_global:
            best_global = bf
        best_trace.append(best_global)
    return np.array(best_trace)

def run_experiment(func_name, dim, rng_mode, trng_file, runs, budget_evals, out_dir, bounds=(-5,5)):
    func = BENCHMARKS[func_name]
    all_runs = []
    os.makedirs(out_dir, exist_ok=True)
    for run_i in range(runs):
        print(f"[{time.strftime('%H:%M:%S')}] Running {func_name} dim={dim} run {run_i+1}/{runs} rng={rng_mode}")
        trng_file_arg = trng_file if rng_mode == "trng" else None
        seed = run_i
        trace = run_single_lshade(dim, func, budget_evals, rng_mode=rng_mode, trng_file=trng_file_arg, seed=seed, bounds=bounds)
        df = pd.DataFrame({"gen": np.arange(len(trace)), "best_f": trace})
        fname = os.path.join(out_dir, f"{func_name}_dim{dim}_rng-{rng_mode}_run{run_i+1}.csv")
        df.to_csv(fname, index=False)
        all_runs.append(trace)
    all_runs = np.array(all_runs)
    return all_runs

# -------------------------
# Summarize and plot (same style as CMA code)
# -------------------------
def summarize_and_plot(all_runs_prng, all_runs_trng, func_name, dim, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    def stats_from_runs(all_runs):
        median = np.median(all_runs, axis=0)
        p25 = np.percentile(all_runs, 25, axis=0)
        p75 = np.percentile(all_runs, 75, axis=0)
        mean = np.mean(all_runs[:, -1])
        std = np.std(all_runs[:, -1])
        return {"median": median, "p25": p25, "p75": p75, "mean_last": mean, "std_last": std}

    s_pr = stats_from_runs(all_runs_prng)
    s_tr = stats_from_runs(all_runs_trng)

    gens = np.arange(len(s_pr["median"]))
    plt.figure(figsize=(8,5))
    plt.fill_between(gens, s_pr["p25"], s_pr["p75"], alpha=0.25, label="PRNG IQR")
    plt.plot(gens, s_pr["median"], label="PRNG median", linestyle='--')
    plt.fill_between(gens, s_tr["p25"], s_tr["p75"], alpha=0.25, label="TRNG IQR")
    plt.plot(gens, s_tr["median"], label="TRNG median", linestyle='-')
    plt.yscale('log')
    plt.xlabel("Generation")
    plt.ylabel("Best fitness (log scale)")
    plt.title(f"L-SHADE convergence on {func_name} (dim={dim})")
    plt.legend()
    plt.grid(True, which="both", ls="--", lw=0.3)
    plotfile = os.path.join(out_dir, f"convergence_{func_name}_dim{dim}.png")
    plt.savefig(plotfile, dpi=200)
    plt.close()
    print(f"Saved convergence plot to {plotfile}")

    final_pr = all_runs_prng[:, -1]
    final_tr = all_runs_trng[:, -1]
    print(f"PRNG final bests: mean={np.mean(final_pr):.6e}, std={np.std(final_pr):.6e}")
    print(f"TRNG final bests: mean={np.mean(final_tr):.6e}, std={np.std(final_tr):.6e}")
    if SCIPY_AVAILABLE:
        stat, p = ranksums(final_pr, final_tr)
        print(f"Ranksum test statistic={stat:.4f}, p-value={p:.4e}")
    else:
        print("scipy not available: skipping ranksum test (install scipy for p-values).")

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
    parser.add_argument("--out", type=str, default="results_lshade")
    parser.add_argument("--lower", type=float, default=-5.0)
    parser.add_argument("--upper", type=float, default=5.0)
    args = parser.parse_args()

    bounds = (args.lower, args.upper)

    print("Running PRNG baseline...")
    prng_runs = run_experiment(args.func, args.dim, "prng", None, args.runs, args.budget_evals, os.path.join(args.out, "prng"), bounds)
    print("Running TRNG (this may be slower if TRNG throughput is low)...")
    trng_runs = run_experiment(args.func, args.dim, "trng", args.trng_file, args.runs, args.budget_evals, os.path.join(args.out, "trng"), bounds)

    summarize_and_plot(prng_runs, trng_runs, args.func, args.dim, args.out)

if __name__ == "__main__":
    main()
