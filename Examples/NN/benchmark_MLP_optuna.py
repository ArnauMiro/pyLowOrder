#!/usr/bin/env python

import os
import sys
import json
import shlex
import argparse
import subprocess as sp
from typing import List, Dict

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE = os.path.join(THIS_DIR, "example_MLP_DLR_airfoil.py")


def _run(cmd: List[str]) -> int:
    print("$", " ".join(shlex.quote(c) for c in cmd), flush=True)
    return sp.call(cmd)


def load_results(resudir: str, mode: str) -> Dict:
    f = os.path.join(resudir, f"training_results_mlp_{mode}.npy")
    data = np.load(f, allow_pickle=True).item()
    return data


def get_final_test_loss(res: Dict) -> float:
    tl = res.get('test_loss') or []
    return float(tl[-1]) if len(tl) else float('inf')


def suggest_space(trial):
    # Default search space; must match Example accepted params once wired
    hp = {
        'lr': trial.suggest_float('lr', 1e-4, 5e-3, log=True),
        'lr_gamma': trial.suggest_float('lr_gamma', 0.95, 0.9999),
        'n_layers': trial.suggest_int('n_layers', 3, 8),
        'hidden_size': trial.suggest_int('hidden_size', 64, 256, log=True),
        'p_dropouts': trial.suggest_float('p_dropouts', 0.0, 0.5),
        # epochs fixed to 50 per requirement
    }
    return hp


def launch_example(resudir: str, ddp_on: bool, nproc: int, basedir: str, casestr: str, params: Dict, batch_size: int = None, epochs: int = 50) -> int:
    os.makedirs(resudir, exist_ok=True)
    # Write hparams.json for the Example to consume (keeps toflexo clean)
    hp = dict(params)
    hp['epochs'] = int(epochs)
    if batch_size is not None:
        hp['batch_size'] = int(batch_size)
    hp_path = os.path.join(resudir, 'hparams.json')
    with open(hp_path, 'w') as f:
        json.dump(hp, f)
    if ddp_on:
        cmd = [
            os.environ.get('TORCHRUN_BIN', 'torchrun'), '--standalone', '--nproc_per_node', str(nproc),
            EXAMPLE, '--ddp', 'on', '--resudir', resudir, '--basedir', basedir, '--casestr', casestr,
        ]
    else:
        cmd = [sys.executable, EXAMPLE, '--ddp', 'off', '--resudir', resudir, '--basedir', basedir, '--casestr', casestr]
    return _run(cmd)


def optimize_for_g(out_dir: str, g: int, trials: int, basedir: str, casestr: str, base_batch: int, ddp: bool, fixed_batch: int = None, epochs: int = 50) -> Dict:
    import optuna
    study = optuna.create_study(direction='minimize')

    def objective(trial):
        hp = suggest_space(trial)
        # For 1‑GPU equivalent branch, batch_size is fixed (must match DDP best batch)
        bs = None if fixed_batch is None else int(fixed_batch)
        run_dir = os.path.join(out_dir, f"trial_{trial.number}")
        rc = launch_example(run_dir, ddp_on=ddp, nproc=g if ddp else 1, basedir=basedir, casestr=casestr, params=hp, batch_size=bs, epochs=epochs)
        if rc != 0:
            raise optuna.TrialPruned()
        mode = 'ddp' if ddp else 'single'
        try:
            res = load_results(run_dir, mode)
            loss = get_final_test_loss(res)
        except Exception:
            loss = float('inf')
        return loss

    study.optimize(objective, n_trials=trials)
    best = study.best_params
    best['epochs'] = int(epochs)
    if fixed_batch is not None:
        best['batch_size'] = int(fixed_batch)
    # Persist best params
    bp_dir = os.path.join(out_dir, 'optuna_ddp' if ddp else 'optuna_single')
    os.makedirs(bp_dir, exist_ok=True)
    with open(os.path.join(bp_dir, 'best_params.json'), 'w') as f:
        json.dump(best, f, indent=2)
    # Return best config
    return best


def aggregate_runs(base_dir: str, mode: str, repeats: int) -> Dict:
    runs = []
    for i in range(1, repeats + 1):
        rdir = os.path.join(base_dir, f"run{i}")
        try:
            runs.append(load_results(rdir, mode))
        except Exception as e:
            print(f"[WARN] Skipping run {i} at {rdir}: {e}")
    def _ms(key):
        vals = [r.get(key) for r in runs if r.get(key) is not None]
        if not vals:
            return None, None
        arr = np.asarray(vals, dtype=float)
        return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    out = {'mode': mode, 'runs': len(runs)}
    for k in ['cr_total_time_s', 'avg_epoch_time_s', 'avg_throughput_global']:
        m, s = _ms(k)
        out[f"{k}_mean"] = m
        out[f"{k}_std"] = s
    # Curves mean
    tl = [r.get('test_loss') for r in runs if r.get('test_loss') is not None]
    if tl:
        min_len = min(len(x) for x in tl)
        arr = np.stack([np.asarray(x[:min_len], dtype=float) for x in tl], axis=0)
        out['test_loss_curve_mean'] = arr.mean(axis=0).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--resudir-base', required=True)
    ap.add_argument('--gpus', default='1,2,3,4')
    ap.add_argument('--trials', type=int, default=30)
    ap.add_argument('--repeats', type=int, default=10)
    ap.add_argument('--basedir', default='/home/airbus/CETACEO_cp_interp/DATA/DLR_pylom/')
    ap.add_argument('--casestr', default='NRL7301')
    ap.add_argument('--base-batch', type=int, default=119)
    ap.add_argument('--epochs', type=int, default=50)
    args = ap.parse_args()

    base = args.resudir_base
    g_list = [int(x) for x in str(args.gpus).split(',') if str(x).strip()]
    os.makedirs(base, exist_ok=True)

    summary = {
        'repeats': args.repeats,
        'trials': args.trials,
        'base_batch': args.base_batch,
        'entries': [],
        'entries_equiv': [],
    }

    # For speedup baseline (after g1 DDP aggregate)
    t0 = None

    for g in g_list:
        gdir = os.path.join(base, f"g{g}")
        os.makedirs(gdir, exist_ok=True)
        # 1) Optimize DDP @ g GPUs
        best_ddp = optimize_for_g(gdir, g, args.trials, args.basedir, args.casestr, args.base_batch, ddp=True, epochs=args.epochs)
        # 2) Consolidate with repeats
        for i in range(1, args.repeats + 1):
            rdir = os.path.join(gdir, f"run{i}")
            launch_example(rdir, ddp_on=True, nproc=g, basedir=args.basedir, casestr=args.casestr, params=best_ddp, epochs=args.epochs)
        ddp_agg = aggregate_runs(gdir, 'ddp', args.repeats)
        ddp_agg['gpus'] = g
        summary['entries'].append(ddp_agg)
        if t0 is None:
            t0 = ddp_agg.get('cr_total_time_s_mean') or ddp_agg.get('avg_epoch_time_s_mean')

        # 3) Optimize 1‑GPU with fixed batch = best_ddp['batch_size']
        gdir_e = os.path.join(base, f"g{g}_equiv")
        os.makedirs(gdir_e, exist_ok=True)
        bs_fix = int(best_ddp.get('batch_size', args.base_batch * g))
        best_single = optimize_for_g(gdir_e, 1, args.trials, args.basedir, args.casestr, args.base_batch, ddp=False, fixed_batch=bs_fix, epochs=args.epochs)
        # 4) Consolidate with repeats
        for i in range(1, args.repeats + 1):
            rdir = os.path.join(gdir_e, f"run{i}")
            launch_example(rdir, ddp_on=False, nproc=1, basedir=args.basedir, casestr=args.casestr, params=best_single, batch_size=bs_fix, epochs=args.epochs)
        single_agg = aggregate_runs(gdir_e, 'single', args.repeats)
        single_agg['gpus'] = g
        single_agg['equivalent_batch'] = bs_fix
        summary['entries_equiv'].append(single_agg)

    # Save summary JSON
    with open(os.path.join(base, 'scaling_summary_optuna.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {os.path.join(base, 'scaling_summary_optuna.json')}")

    # Reuse existing equivalence plotting script to generate figures from aggregated runs
    try:
        import subprocess as _sp
        cmd = [
            sys.executable,
            os.path.join(THIS_DIR, 'benchmark_MLP_equiv.py'),
            '--resudir-base', base,
            '--gpus', ','.join(str(x) for x in g_list),
            '--repeats', str(args.repeats),
        ]
        print('$', ' '.join(shlex.quote(c) for c in cmd))
        _sp.check_call(cmd)
    except Exception as e:
        print(f"[WARN] Could not regenerate figures via benchmark_MLP_equiv.py: {e}")


if __name__ == '__main__':
    main()
