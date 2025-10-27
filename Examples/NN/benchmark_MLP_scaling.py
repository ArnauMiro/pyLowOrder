#!/usr/bin/env python

import os
import sys
import json
import time
import shlex
import argparse
import subprocess as sp
from typing import List, Dict, Tuple

import numpy as np


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE = os.path.join(THIS_DIR, "example_MLP_DLR_airfoil.py")


def _run(cmd: List[str], env: Dict[str, str] = None) -> int:
    print("$", " ".join(shlex.quote(c) for c in cmd), flush=True)
    try:
        res = sp.run(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr)
        return res.returncode
    except FileNotFoundError as e:
        print(f"[ERR] Command not found: {e}")
        return 127


def launch_example(resudir: str, ddp_on: bool, nproc: int, basedir: str, casestr: str) -> int:
    os.makedirs(resudir, exist_ok=True)
    if ddp_on:
        torchrun = os.environ.get("TORCHRUN_BIN", "torchrun")
        cmd = [
            torchrun, "--standalone", "--nproc_per_node", str(nproc),
            EXAMPLE,
            "--ddp", "on",
            "--resudir", resudir,
            "--basedir", basedir,
            "--casestr", casestr,
        ]
    else:
        cmd = [
            sys.executable, EXAMPLE,
            "--ddp", "off",
            "--resudir", resudir,
            "--basedir", basedir,
            "--casestr", casestr,
        ]
    return _run(cmd)


def _load_results(resudir: str, mode: str) -> Dict:
    # mode: 'ddp' or 'single'
    fname = os.path.join(resudir, f"training_results_mlp_{mode}.npy")
    if not os.path.exists(fname):
        raise FileNotFoundError(fname)
    data = np.load(fname, allow_pickle=True).item()
    # Derive metrics consistently
    et = data.get("epoch_time_s")
    tl = data.get("test_loss")
    tr = data.get("train_loss")
    epochs = int(len(et)) if et is not None else (int(len(tl)) if tl is not None else None)
    total_time = data.get("cr_total_time_s")
    if total_time is None:
        if et is not None:
            et_np = np.asarray(et, dtype=float)
            total_time = float(np.nansum(et_np)) if et_np.size > 0 else None
    avg_epoch_time = (float(total_time) / float(epochs)) if (total_time is not None and epochs) else None
    thr_g = data.get("throughput_samples_per_sec_global")
    if thr_g is not None:
        thr_g_np = np.asarray(thr_g, dtype=float)
        avg_thr_g = float(np.nanmean(thr_g_np)) if thr_g_np.size > 0 else None
    else:
        avg_thr_g = None
    train = tr if tr is not None else []
    test = tl if tl is not None else []
    final_train = float(train[-1]) if len(train) else None
    final_test = float(test[-1]) if len(test) else None
    # Optional RMSE if preds are saved
    pred_path = os.path.join(resudir, f"scaled_preds_{mode}.npy")
    y_path = os.path.join(resudir, "scaled_y.npy")
    rmse = None
    if os.path.exists(pred_path) and os.path.exists(y_path):
        y_true = np.load(y_path)
        y_pred = np.load(pred_path)
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    return dict(
        mode=mode,
        epochs=epochs,
        total_time_s=total_time,
        avg_epoch_time_s=avg_epoch_time,
        avg_throughput_global=avg_thr_g,
        final_train_loss=final_train,
        final_test_loss=final_test,
        rmse=rmse,
    )


def _mean_std(items: List[Dict], key: str) -> Tuple[float, float]:
    vals = [x.get(key) for x in items if x.get(key) is not None]
    if not vals:
        return None, None
    arr = np.asarray(vals, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def aggregate_runs(base_dir: str, mode: str, repeats: int) -> Dict:
    runs = []
    for i in range(1, repeats + 1):
        run_dir = os.path.join(base_dir, f"run{i}")
        try:
            runs.append(_load_results(run_dir, mode))
        except Exception as e:
            print(f"[WARN] Skipping run {i} at {run_dir}: {e}")
    summary = {
        "mode": mode,
        "runs": len(runs),
    }
    for k in [
        "total_time_s", "avg_epoch_time_s", "avg_throughput_global",
        "final_train_loss", "final_test_loss", "rmse",
    ]:
        m, s = _mean_std(runs, k)
        summary[f"{k}_mean"] = m
        summary[f"{k}_std"] = s
    return summary


def task_compare(single_base: str, ddp_base: str, repeats: int, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    s = aggregate_runs(single_base, "single", repeats)
    d = aggregate_runs(ddp_base, "ddp", repeats)
    comp = {
        "single": s,
        "ddp": d,
        "repeats": repeats,
        "speedup_total_time_mean": None,
        "speedup_total_time_std": None,
    }
    try:
        if (s.get("total_time_s_mean") is not None and d.get("total_time_s_mean") is not None
                and d["total_time_s_mean"] > 0):
            comp["speedup_total_time_mean"] = float(s["total_time_s_mean"]) / float(d["total_time_s_mean"])
    except Exception:
        pass
    with open(os.path.join(out_dir, "comparison_avg.json"), "w") as f:
        json.dump(comp, f, indent=2)
    print(f"Saved: {os.path.join(out_dir, 'comparison_avg.json')}")

    # Optional: bar figures with error bars for selected metrics
    try:
        import matplotlib.pyplot as plt
        labels = ["Single", "DDP"]
        # Build metric arrays (mean, std) pairs
        metrics = [
            ("Total Training Time (s)",
             [s.get("total_time_s_mean"), d.get("total_time_s_mean")],
             [s.get("total_time_s_std"), d.get("total_time_s_std")],
             "Seconds",
             True),  # annotate speedup
            ("Avg Throughput (global)",
             [s.get("avg_throughput_global_mean"), d.get("avg_throughput_global_mean")],
             [s.get("avg_throughput_global_std"), d.get("avg_throughput_global_std")],
             "Samples/s",
             False),
            ("Final Test Loss",
             [s.get("final_test_loss_mean"), d.get("final_test_loss_mean")],
             [s.get("final_test_loss_std"), d.get("final_test_loss_std")],
             "Loss",
             False),
            ("RMSE",
             [s.get("rmse_mean"), d.get("rmse_mean")],
             [s.get("rmse_std"), d.get("rmse_std")],
             "RMSE",
             False),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        axes = axes.ravel()
        for i, (title, vals, errs, ylabel, annotate_sp) in enumerate(metrics):
            ax = axes[i]
            v_ok = [v for v in vals if v is not None]
            if len(v_ok) == 2:
                ax.bar(labels, vals, yerr=errs if all(e is not None for e in errs) else None, capsize=4)
                ax.set_title(title)
                ax.set_ylabel(ylabel)
                ax.grid(True, axis='y', linestyle='--', alpha=0.4)
                if annotate_sp and vals[0] and vals[1] and vals[1] > 0:
                    sp = vals[0] / vals[1]
                    ax.text(0.5, 0.95, f"Speedup DDP: ×{sp:.2f}", transform=ax.transAxes, ha='center', va='top', fontsize=9)
            else:
                ax.text(0.5, 0.5, "N/A", ha='center', va='center')
                ax.set_title(title)
                ax.axis('off')
        fig.tight_layout()
        out_png = os.path.join(out_dir, 'comparison_avg_bars.png')
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
        print(f"Saved: {out_png}")
    except Exception as e:
        print(f"[WARN] Could not generate comparison_avg_bars.png: {e}")


def task_scale(resudir_base: str, gpus: List[int], repeats: int, basedir: str, casestr: str, launch: bool, ddp_baseline: bool) -> None:
    os.makedirs(resudir_base, exist_ok=True)
    per_g = {}
    for g in gpus:
        gdir = os.path.join(resudir_base, f"g{g}")
        os.makedirs(gdir, exist_ok=True)
        print(f"[INFO] GPUs={g} repeats={repeats} dir={gdir}")
        mode = "ddp" if (ddp_baseline or g > 1) else "single"
        # Launch runs
        if launch:
            for i in range(1, repeats + 1):
                rdir = os.path.join(gdir, f"run{i}")
                print(f"[RUN] g={g} i={i} -> {rdir}")
                rc = launch_example(rdir, ddp_on=(mode == "ddp"), nproc=g if mode == "ddp" else 1, basedir=basedir, casestr=casestr)
                if rc != 0:
                    print(f"[ERR] Run failed (g={g}, i={i}) rc={rc}")
                    break
        # Aggregate
        per_g[g] = aggregate_runs(gdir, mode, repeats)

    # Compute speedup (vs first g)
    g0 = gpus[0]
    t0 = per_g[g0].get("total_time_s_mean")
    scaling = {"repeats": repeats, "ddp_baseline": ddp_baseline, "entries": []}
    for g in gpus:
        tg = per_g[g].get("total_time_s_mean")
        sp = (t0 / tg) if (t0 and tg and tg > 0) else None
        eff = (sp / g) if (sp and g > 0) else None
        scaling["entries"].append({
            "gpus": g,
            **per_g[g],
            "speedup_vs_{}".format(g0): sp,
            "efficiency": eff,
        })
    out_json = os.path.join(resudir_base, "scaling_summary.json")
    with open(out_json, "w") as f:
        json.dump(scaling, f, indent=2)
    print(f"Saved: {out_json}")

    # Optional plotting (no hard failure if matplotlib not present)
    try:
        import matplotlib.pyplot as plt
        gp = [e["gpus"] for e in scaling["entries"]]
        tt = [float(v) if v is not None else np.nan for v in [e.get("total_time_s_mean") for e in scaling["entries"]]]
        spv = [float(v) if v is not None else np.nan for v in [e.get(f"speedup_vs_{g0}") for e in scaling["entries"]]]
        rm = [float(v) if v is not None else np.nan for v in [e.get("rmse_mean") for e in scaling["entries"]]]
        rm_std = [float(v) if v is not None else np.nan for v in [e.get("rmse_std") for e in scaling["entries"]]]
        # Time
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar([str(x) for x in gp], tt)
        ax.set_title("Total Training Time (mean)")
        ax.set_xlabel("GPUs")
        ax.set_ylabel("Seconds")
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)
        fig.tight_layout()
        fig.savefig(os.path.join(resudir_base, "scaling_time.png"), dpi=300)
        plt.close(fig)
        # Speedup
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(gp, spv, marker='o')
        ax.set_title(f"Speedup vs {g0} GPU(s)")
        ax.set_xlabel("GPUs")
        ax.set_ylabel("Speedup")
        ax.grid(True, linestyle='--', alpha=0.4)
        fig.tight_layout()
        fig.savefig(os.path.join(resudir_base, "scaling_speedup.png"), dpi=300)
        plt.close(fig)
        # RMSE vs GPUs (with error bars)
        rm_arr = np.asarray(rm, dtype=float)
        rm_std_arr = np.asarray(rm_std, dtype=float)
        gp_arr = np.asarray(gp, dtype=int)
        mask = np.isfinite(rm_arr)
        if np.any(mask):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.errorbar(gp_arr[mask], rm_arr[mask], yerr=rm_std_arr[mask], fmt='-o', capsize=4)
            ax.set_title("RMSE vs GPUs")
            ax.set_xlabel("GPUs")
            ax.set_ylabel("RMSE (mean ± std)")
            ax.grid(True, linestyle='--', alpha=0.4)
            fig.tight_layout()
            fig.savefig(os.path.join(resudir_base, "scaling_rmse.png"), dpi=300)
            plt.close(fig)
        # RMSE vs Training Time (scatter with annotations)
        tt_arr = np.asarray(tt, dtype=float)
        mask2 = np.isfinite(tt_arr) & np.isfinite(rm_arr)
        if np.any(mask2):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(tt_arr[mask2], rm_arr[mask2], c='tab:blue')
            for x, y, g in zip(tt_arr[mask2], rm_arr[mask2], gp_arr[mask2]):
                ax.annotate(str(int(g)), (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
            # Pearson corr if >=2 points
            if np.count_nonzero(mask2) >= 2:
                try:
                    r = float(np.corrcoef(tt_arr[mask2], rm_arr[mask2])[0,1])
                    ax.set_title(f"RMSE vs Training Time (r={r:.3f})")
                except Exception:
                    ax.set_title("RMSE vs Training Time")
            else:
                ax.set_title("RMSE vs Training Time")
            ax.set_xlabel("Total time (s)")
            ax.set_ylabel("RMSE (mean)")
            ax.grid(True, linestyle='--', alpha=0.4)
            fig.tight_layout()
            fig.savefig(os.path.join(resudir_base, "rmse_vs_time.png"), dpi=300)
            plt.close(fig)
        print(f"Saved: {os.path.join(resudir_base, 'scaling_time.png')}, scaling_speedup.png, scaling_rmse.png, rmse_vs_time.png")
    except Exception as e:
        print(f"[WARN] Could not generate scaling figures: {e}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="task", required=True)

    # compare: average single vs ddp from two base dirs
    pc = sub.add_parser("compare", help="Average metrics over N runs for single vs ddp")
    pc.add_argument("--single-dir", required=True, help="Base directory with single/run1..runN")
    pc.add_argument("--ddp-dir", required=True, help="Base directory with ddp/run1..runN")
    pc.add_argument("--repeats", type=int, default=5)
    pc.add_argument("--out-dir", required=True)

    # scale: average over repeats for GPUs list, optional launching
    ps = sub.add_parser("scale", help="Scaling study over GPUs with repeated runs")
    ps.add_argument("--resudir-base", required=True, help="Base directory where g*/run*/ are created/used")
    ps.add_argument("--gpus", default="1,2,3,4", help="Comma-separated list of GPU counts")
    ps.add_argument("--repeats", type=int, default=5)
    ps.add_argument("--basedir", default="/home/airbus/CETACEO_cp_interp/DATA/DLR_pylom/")
    ps.add_argument("--casestr", default="NRL7301")
    ps.add_argument("--launch", action="store_true", help="Actually launch training runs; otherwise only aggregate")
    ps.add_argument("--ddp-baseline", action="store_true", help="Use DDP even for g=1 (recommended for fair scaling)")

    args = p.parse_args()

    if args.task == "compare":
        task_compare(args.single_dir, args.ddp_dir, args.repeats, args.out_dir)
    elif args.task == "scale":
        g_list = [int(x) for x in str(args.gpus).split(',') if str(x).strip()]
        task_scale(args.resudir_base, g_list, args.repeats, args.basedir, args.casestr, args.launch, args.ddp_baseline)
    else:
        p.error("Unknown task")


if __name__ == "__main__":
    main()
