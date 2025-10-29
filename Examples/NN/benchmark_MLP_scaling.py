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


def launch_example(resudir: str, ddp_on: bool, nproc: int, basedir: str, casestr: str,
                   epochs: int | None = None, batch_size: int | None = None) -> int:
    os.makedirs(resudir, exist_ok=True)
    # If epochs/batch are provided, write hparams.json for the Example to consume
    if epochs is not None or batch_size is not None:
        hp = {}
        if epochs is not None:
            hp['epochs'] = int(epochs)
        if batch_size is not None:
            hp['batch_size'] = int(batch_size)
        try:
            with open(os.path.join(resudir, 'hparams.json'), 'w') as f:
                json.dump(hp, f)
        except Exception:
            pass
    if ddp_on:
        torchrun = os.environ.get("TORCHRUN_BIN", "torchrun")
        cmd = [
            torchrun, "--standalone", "--nproc_per_node", str(nproc),
            EXAMPLE,
            "--ddp", "on",
            "--resudir", resudir,
            "--basedir", basedir,
            "--casestr", casestr,
            "--no-figures",
        ]
    else:
        cmd = [
            sys.executable, EXAMPLE,
            "--ddp", "off",
            "--resudir", resudir,
            "--basedir", basedir,
            "--casestr", casestr,
            "--no-figures",
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
        train_loss_curve=np.asarray(train, dtype=float) if len(train) else None,
        test_loss_curve=np.asarray(test, dtype=float) if len(test) else None,
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
    # Curves aggregation (align by min length)
    tr_curves = [r["train_loss_curve"] for r in runs if r.get("train_loss_curve") is not None]
    ts_curves = [r["test_loss_curve"] for r in runs if r.get("test_loss_curve") is not None]
    def _agg_curves(curves):
        if not curves:
            return None, None
        min_len = min(c.shape[0] for c in curves)
        if min_len <= 0:
            return None, None
        arr = np.stack([c[:min_len] for c in curves], axis=0)
        return arr.mean(axis=0), (arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros(min_len))
    tr_mean, tr_std = _agg_curves(tr_curves)
    ts_mean, ts_std = _agg_curves(ts_curves)

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
    # Attach curves (as lists for JSON serializable)
    if tr_mean is not None:
        summary["train_loss_curve_mean"] = tr_mean.tolist()
        summary["train_loss_curve_std"] = tr_std.tolist()
    if ts_mean is not None:
        summary["test_loss_curve_mean"] = ts_mean.tolist()
        summary["test_loss_curve_std"] = ts_std.tolist()
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
        fig_dir = os.path.join(resudir_base, 'figures')
        os.makedirs(fig_dir, exist_ok=True)
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


def task_scale(resudir_base: str, gpus: List[int], repeats: int, basedir: str, casestr: str, launch: bool, ddp_baseline: bool, equiv: bool, base_batch: int, epochs: int | None = None) -> None:
    os.makedirs(resudir_base, exist_ok=True)
    # Ensure central figures directory exists and is available for all plotting blocks
    fig_dir = os.path.join(resudir_base, 'figures')
    try:
        os.makedirs(fig_dir, exist_ok=True)
    except Exception:
        pass
    per_g = {}
    per_g_equiv = {}
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
                rc = launch_example(rdir, ddp_on=(mode == "ddp"), nproc=g if mode == "ddp" else 1,
                                    basedir=basedir, casestr=casestr, epochs=epochs)
                if rc != 0:
                    print(f"[ERR] Run failed (g={g}, i={i}) rc={rc}")
                    break
        # Aggregate
        per_g[g] = aggregate_runs(gdir, mode, repeats)

        # Per-run: write training losses log JSON inside each run directory
        for i in range(1, repeats + 1):
            rdir = os.path.join(gdir, f"run{i}")
            tr_file = os.path.join(rdir, f"training_results_mlp_{mode}.npy")
            if not os.path.exists(tr_file):
                continue
            try:
                data = np.load(tr_file, allow_pickle=True).item()
                outj = {
                    "mode": mode,
                    "gpus": g,
                    "run": i,
                    "epochs": int(len(data.get('test_loss'))) if data.get('test_loss') is not None else None,
                    "train_loss": (np.asarray(data.get('train_loss'), dtype=float).tolist()
                                    if data.get('train_loss') is not None else None),
                    "test_loss": (np.asarray(data.get('test_loss'), dtype=float).tolist()
                                   if data.get('test_loss') is not None else None),
                    "epoch_time_s": (np.asarray(data.get('epoch_time_s'), dtype=float).tolist()
                                      if data.get('epoch_time_s') is not None else None),
                    "notes": "Generated from training_results_mlp_*.npy by benchmark_MLP_scaling.py"
                }
                with open(os.path.join(rdir, "training_log.json"), "w") as f:
                    json.dump(outj, f, indent=2)
                # Also write a human-readable training.log (CSV-like)
                try:
                    tr = np.asarray(data.get('train_loss'), dtype=float) if data.get('train_loss') is not None else np.asarray([], dtype=float)
                    ts = np.asarray(data.get('test_loss'), dtype=float) if data.get('test_loss') is not None else np.asarray([], dtype=float)
                    et = np.asarray(data.get('epoch_time_s'), dtype=float) if data.get('epoch_time_s') is not None else np.asarray([], dtype=float)
                    n = int(max(tr.size, ts.size, et.size))
                    with open(os.path.join(rdir, "training.log"), "w") as tf:
                        tf.write("epoch,train_loss,test_loss,epoch_time_s\n")
                        for k in range(n):
                            a = tr[k] if k < tr.size else float('nan')
                            b = ts[k] if k < ts.size else float('nan')
                            c = et[k] if k < et.size else float('nan')
                            tf.write(f"{k+1},{a:.6e},{b:.6e},{c:.6e}\n")
                except Exception:
                    pass
            except Exception:
                pass

        # Optional: also run equivalent 1-GPU with batch = base_batch * g
        if equiv:
            gdir_e = os.path.join(resudir_base, f"g{g}_equiv")
            os.makedirs(gdir_e, exist_ok=True)
            if launch:
                bs = int(base_batch) * int(g)
                for i in range(1, repeats + 1):
                    rdir = os.path.join(gdir_e, f"run{i}")
                    print(f"[RUN][EQUIV] g={g} i={i} (1-GPU, batch={bs}) -> {rdir}")
                    rc = launch_example(rdir, ddp_on=False, nproc=1, basedir=basedir, casestr=casestr,
                                        epochs=epochs, batch_size=bs)
                    if rc != 0:
                        print(f"[ERR] Equiv run failed (g={g}, i={i}) rc={rc}")
                        break
            per_g_equiv[g] = aggregate_runs(gdir_e, 'single', repeats)
            # Per-run logs for equivalent branch
            for i in range(1, repeats + 1):
                rdir = os.path.join(gdir_e, f"run{i}")
                tr_file = os.path.join(rdir, f"training_results_mlp_single.npy")
                if not os.path.exists(tr_file):
                    continue
                try:
                    data = np.load(tr_file, allow_pickle=True).item()
                    outj = {
                        "mode": 'single',
                        "gpus": g,
                        "run": i,
                        "epochs": int(len(data.get('test_loss'))) if data.get('test_loss') is not None else None,
                        "train_loss": (np.asarray(data.get('train_loss'), dtype=float).tolist()
                                        if data.get('train_loss') is not None else None),
                        "test_loss": (np.asarray(data.get('test_loss'), dtype=float).tolist()
                                       if data.get('test_loss') is not None else None),
                        "epoch_time_s": (np.asarray(data.get('epoch_time_s'), dtype=float).tolist()
                                          if data.get('epoch_time_s') is not None else None),
                        "notes": "Generated from training_results_mlp_single.npy by benchmark_MLP_scaling.py"
                    }
                    with open(os.path.join(rdir, "training_log.json"), 'w') as f:
                        json.dump(outj, f, indent=2)
                    # Also write a human-readable training.log (CSV-like)
                    try:
                        tr = np.asarray(data.get('train_loss'), dtype=float) if data.get('train_loss') is not None else np.asarray([], dtype=float)
                        ts = np.asarray(data.get('test_loss'), dtype=float) if data.get('test_loss') is not None else np.asarray([], dtype=float)
                        et = np.asarray(data.get('epoch_time_s'), dtype=float) if data.get('epoch_time_s') is not None else np.asarray([], dtype=float)
                        n = int(max(tr.size, ts.size, et.size))
                        with open(os.path.join(rdir, "training.log"), "w") as tf:
                            tf.write("epoch,train_loss,test_loss,epoch_time_s\n")
                            for k in range(n):
                                a = tr[k] if k < tr.size else float('nan')
                                b = ts[k] if k < ts.size else float('nan')
                                c = et[k] if k < et.size else float('nan')
                                tf.write(f"{k+1},{a:.6e},{b:.6e},{c:.6e}\n")
                    except Exception:
                        pass
                except Exception:
                    pass

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
    if equiv:
        scaling["entries_equiv"] = []
        for g in gpus:
            if g in per_g_equiv:
                scaling["entries_equiv"].append({
                    "gpus": g,
                    **per_g_equiv[g],
                    "equivalent_batch": int(base_batch) * int(g),
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
        fig.savefig(os.path.join(fig_dir, "scaling_time.png"), dpi=300)
        plt.close(fig)
        # Speedup
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(gp, spv, marker='o')
        ax.set_title(f"Speedup vs {g0} GPU(s)")
        ax.set_xlabel("GPUs")
        ax.set_ylabel("Speedup")
        ax.grid(True, linestyle='--', alpha=0.4)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "scaling_speedup.png"), dpi=300)
        plt.close(fig)
        # 2x2 grouped bars: DDP vs 1-GPU-equivalent (if present)
        ae = [float(v) if v is not None else np.nan for v in [e.get('avg_epoch_time_s_mean') for e in scaling['entries']]]
        thr = [float(v) if v is not None else np.nan for v in [e.get('avg_throughput_global_mean') for e in scaling['entries']]]
        fl = [float(v) if v is not None else np.nan for v in [e.get('final_test_loss_mean') for e in scaling['entries']]]
        entries_equiv = scaling.get('entries_equiv', [])
        has_equiv = bool(entries_equiv)
        if has_equiv:
            emap = {e.get('gpus'): e for e in entries_equiv}
            tt_e = [float(emap.get(g, {}).get('total_time_s_mean')) if emap.get(g, {}).get('total_time_s_mean') is not None else np.nan for g in gp]
            ae_e = [float(emap.get(g, {}).get('avg_epoch_time_s_mean')) if emap.get(g, {}).get('avg_epoch_time_s_mean') is not None else np.nan for g in gp]
            thr_e = [float(emap.get(g, {}).get('avg_throughput_global_mean')) if emap.get(g, {}).get('avg_throughput_global_mean') is not None else np.nan for g in gp]
            fl_e = [float(emap.get(g, {}).get('final_test_loss_mean')) if emap.get(g, {}).get('final_test_loss_mean') is not None else np.nan for g in gp]
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        axes = axes.ravel()
        x = np.arange(len(gp))
        w = 0.35
        def _group(ax, ddp_vals, equiv_vals, title, ylabel, annotate_kind):
            b1 = ax.bar(x - w/2, ddp_vals, width=w, label='DDP')
            if has_equiv:
                b2 = ax.bar(x + w/2, equiv_vals, width=w, label='1GPU (b×g)')
            ax.set_xticks(x, [str(g) for g in gp])
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(True, axis='y', linestyle='--', alpha=0.4)
            # annotate relative to DDP baseline at g0
            base = ddp_vals[0] if ddp_vals and np.isfinite(ddp_vals[0]) else None
            if base and base > 0:
                for i, v in enumerate(ddp_vals):
                    if v and np.isfinite(v):
                        if annotate_kind == 'time' or annotate_kind == 'loss':
                            ax.text(x[i]-w/2, v, f"×{base/float(v):.2f}", ha='center', va='bottom', fontsize=8)
                        elif annotate_kind == 'thr':
                            ax.text(x[i]-w/2, v, f"×{float(v)/base:.2f}", ha='center', va='bottom', fontsize=8)
                if has_equiv:
                    for i, v in enumerate(equiv_vals):
                        if v and np.isfinite(v):
                            if annotate_kind == 'time' or annotate_kind == 'loss':
                                ax.text(x[i]+w/2, v, f"×{base/float(v):.2f}", ha='center', va='bottom', fontsize=8)
                            elif annotate_kind == 'thr':
                                ax.text(x[i]+w/2, v, f"×{float(v)/base:.2f}", ha='center', va='bottom', fontsize=8)
            if has_equiv:
                ax.legend()
        _group(axes[0], tt, tt_e if has_equiv else [], 'Total Training Time (mean)', 'Seconds', 'time')
        _group(axes[1], ae, ae_e if has_equiv else [], 'Avg Epoch Time (mean)', 'Seconds', 'time')
        _group(axes[2], thr, thr_e if has_equiv else [], 'Avg Throughput (global)', 'Samples/s', 'thr')
        _group(axes[3], fl, fl_e if has_equiv else [], 'Final Test Loss (mean)', 'Loss', 'loss')
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, 'scaling_bars_2x2.png'), dpi=300)
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
        # Auto Y axis range for RMSE (no fixed limits)
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, "scaling_rmse.png"), dpi=300)
            plt.close(fig)
        # RMSE vs Training Time (scatter with annotations)
        tt_arr = np.asarray(tt, dtype=float)
        mask2 = np.isfinite(tt_arr) & np.isfinite(rm_arr)
        if np.any(mask2):
            fig, ax = plt.subplots(figsize=(6, 4))
            xdat = tt_arr[mask2]
            ydat = rm_arr[mask2]
            gdat = gp_arr[mask2]
            ax.scatter(xdat, ydat, c='tab:blue')
            for x, y, g in zip(xdat, ydat, gdat):
                ax.annotate(str(int(g)), (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
            # Pearson corr and linear regression (with intercept)
            if xdat.size >= 2:
                try:
                    r = float(np.corrcoef(xdat, ydat)[0,1])
                except Exception:
                    r = np.nan
                try:
                    a, b = np.polyfit(xdat, ydat, 1)
                    y_fit = a * xdat + b
                    ss_res = float(np.nansum((ydat - y_fit) ** 2))
                    ss_tot = float(np.nansum((ydat - np.nanmean(ydat)) ** 2) + 1e-12)
                    r2 = 1.0 - ss_res / ss_tot
                    x_line = np.linspace(0.0, float(np.nanmax(xdat)), 100)
                    y_line = a * x_line + b
                    ax.plot(x_line, y_line, color='red', linestyle='--', linewidth=1.0,
                            label=f"Reg: y={a:.3e}x+{b:.3f}, R2={r2:.3f}")
                except Exception:
                    r = r if np.isfinite(r) else np.nan
                title_suf = f" (r={r:.3f})" if np.isfinite(r) else ""
                ax.set_title(f"RMSE vs Training Time{title_suf}")
            else:
                ax.set_title("RMSE vs Training Time")
            ax.set_xlabel("Total time (s)")
            ax.set_ylabel("RMSE (mean)")
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.legend()
            # Ensure X axis starts at 0 (auto Y range)
            ax.set_xlim(left=0.0)
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, "rmse_vs_time.png"), dpi=300)
            plt.close(fig)
        # Also output average losses per g: linear and log
        try:
            colors = plt.cm.tab10.colors
            # Linear
            fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            for idx, e in enumerate(scaling.get('entries', [])):
                g = e.get('gpus')
                trm = e.get('train_loss_curve_mean')
                tsm = e.get('test_loss_curve_mean')
                col = colors[idx % len(colors)]
                if trm is not None:
                    y = np.asarray(trm, dtype=float)
                    axes[0].plot(np.arange(1, y.size+1), y, label=f"g={g}", color=col)
                if tsm is not None:
                    y = np.asarray(tsm, dtype=float)
                    axes[1].plot(np.arange(1, y.size+1), y, label=f"g={g}", color=col)
            axes[0].set_title('Average Train Loss per GPU count (DDP)')
            axes[1].set_title('Average Test Loss per GPU count (DDP)')
            for ax in axes:
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.grid(True, linestyle='--', alpha=0.4)
                ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, 'avg_losses_by_g.png'), dpi=300)
            plt.close(fig)
            # Log
            fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            for idx, e in enumerate(scaling.get('entries', [])):
                g = e.get('gpus')
                trm = e.get('train_loss_curve_mean')
                tsm = e.get('test_loss_curve_mean')
                col = colors[idx % len(colors)]
                if trm is not None:
                    y = np.asarray(trm, dtype=float)
                    axes[0].plot(np.arange(1, y.size+1), y, label=f"g={g}", color=col)
                if tsm is not None:
                    y = np.asarray(tsm, dtype=float)
                    axes[1].plot(np.arange(1, y.size+1), y, label=f"g={g}", color=col)
            axes[0].set_title('Average Train Loss per GPU count (DDP) (log)')
            axes[1].set_title('Average Test Loss per GPU count (DDP) (log)')
            for ax in axes:
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_yscale('log')
                ax.grid(True, linestyle='--', alpha=0.4)
                ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, 'avg_losses_by_g_log.png'), dpi=300)
            plt.close(fig)
        except Exception:
            pass
        print(f"Saved: {os.path.join(fig_dir, 'scaling_time.png')}, scaling_speedup.png, scaling_rmse.png, rmse_vs_time.png, avg_losses_by_g.png, avg_losses_by_g_log.png")
    except Exception as e:
        print(f"[WARN] Could not generate scaling figures: {e}")

    # For each g: aggregate true vs pred across runs and save scatter with regression
    try:
        import matplotlib.pyplot as plt
        for e in scaling["entries"]:
            g = e["gpus"]
            gdir = os.path.join(resudir_base, f"g{g}")
            y_list, p_list = [], []
            for i in range(1, repeats + 1):
                rdir = os.path.join(gdir, f"run{i}")
                yp = os.path.join(rdir, "scaled_preds_ddp.npy")
                yt = os.path.join(rdir, "scaled_y.npy")
                if os.path.exists(yp) and os.path.exists(yt):
                    try:
                        y_true = np.load(yt)
                        y_pred = np.load(yp)
                        # flatten to 1D if needed
                        y_list.append(y_true.reshape(-1))
                        p_list.append(y_pred.reshape(-1))
                    except Exception:
                        pass
            if y_list and p_list:
                y = np.concatenate(y_list)
                p = np.concatenate(p_list)
                # regression
                try:
                    a, b = np.polyfit(y, p, 1)
                    y_fit = a * y + b
                    ss_res = np.sum((p - y_fit) ** 2)
                    ss_tot = np.sum((p - np.mean(p)) ** 2) + 1e-12
                    r2 = 1.0 - ss_res / ss_tot
                except Exception:
                    a = b = r2 = np.nan
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(y, p, s=1, alpha=0.3, label='Data')
                if np.isfinite(a) and np.isfinite(b):
                    x_line = np.linspace(np.nanmin(y), np.nanmax(y), 100)
                    y_line = a * x_line + b
                    ax.plot(x_line, y_line, 'r--', lw=1.0, label=f"Reg: y={a:.3f}x+{b:.3f}, R2={r2:.4f}")
                rmse_g = float(np.sqrt(np.nanmean((p - y) ** 2)))
                ax.set_title(f"True vs Pred (g={g}) RMSE={rmse_g:.3e}")
                ax.set_xlabel("True")
                ax.set_ylabel("Pred")
                ax.grid(True)
                ax.legend()
                fig.tight_layout()
                fig.savefig(os.path.join(fig_dir, f"true_vs_pred_ddp_avg_g{g}.png"), dpi=300)
                plt.close(fig)
    except Exception as e:
        print(f"[WARN] Could not generate per-g true_vs_pred plots: {e}")

    # Avg training/test losses per g in one figure (two subplots) with light error bands
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        colors = plt.cm.tab10.colors
        for idx, e in enumerate(scaling["entries"]):
            g = e["gpus"]
            trm = e.get("train_loss_curve_mean")
            trs = e.get("train_loss_curve_std")
            tsm = e.get("test_loss_curve_mean")
            tss = e.get("test_loss_curve_std")
            col = colors[idx % len(colors)]
            if trm is not None:
                trm_np = np.asarray(trm, dtype=float)
                axes[0].plot(np.arange(1, trm_np.size+1), trm_np, label=f"g={g}", color=col, lw=1.8)
                if trs is not None:
                    trs_np = np.asarray(trs, dtype=float)
                    if trs_np.size == trm_np.size:
                        lo, hi = trm_np - trs_np, trm_np + trs_np
                        axes[0].fill_between(np.arange(1, trm_np.size+1), lo, hi, color=col, alpha=0.12, linewidth=0)
            if tsm is not None:
                tsm_np = np.asarray(tsm, dtype=float)
                axes[1].plot(np.arange(1, tsm_np.size+1), tsm_np, label=f"g={g}", color=col, lw=1.8)
                if tss is not None:
                    tss_np = np.asarray(tss, dtype=float)
                    if tss_np.size == tsm_np.size:
                        lo, hi = tsm_np - tss_np, tsm_np + tss_np
                        axes[1].fill_between(np.arange(1, tsm_np.size+1), lo, hi, color=col, alpha=0.12, linewidth=0)
        axes[0].set_title("Average Training Loss per GPU count")
        axes[1].set_title("Average Test Loss per GPU count")
        for ax in axes:
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(True)
            ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(resudir_base, "avg_losses_by_g.png"), dpi=300)
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] Could not generate avg_losses_by_g.png: {e}")


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
    ps.add_argument("--equiv", action="store_true", help="Also run 1-GPU equivalent (batch=base-batch×g) and aggregate")
    ps.add_argument("--base-batch", type=int, default=119, help="Baseline per-process batch used to build equivalent 1-GPU batch")
    ps.add_argument("--epochs", type=int, default=None, help="Override number of epochs for Example runs")

    # plot_curves: regenerate curves (train/test) from a results directory
    pp = sub.add_parser("plot_curves", help="Plot training/test curves from existing NPYs in a results dir")
    pp.add_argument("--resudir", required=True, help="Results directory containing training_results_mlp_*.npy")
    pp.add_argument("--out", default=None, help="Output PNG path (defaults to <resudir>/curves_train_test_log.png)")
    pp.add_argument("--logy", action="store_true", help="Use logarithmic Y scale")

    # plot_avg_by_g: regenerate the averaged curves per g from scaling_summary.json
    pg = sub.add_parser("plot_avg_by_g", help="Plot averaged train/test curves per GPU count from scaling_summary.json")
    pg.add_argument("--resudir-base", required=True, help="Scaling base directory containing scaling_summary.json")
    pg.add_argument("--logy", action="store_true", help="Use logarithmic Y scale")

    # plot_rmse_vs_time: regenerate the RMSE vs time scatter with regression from scaling_summary.json
    pr = sub.add_parser("plot_rmse_vs_time", help="Plot RMSE vs Training Time (with regression) from scaling_summary.json")
    pr.add_argument("--resudir-base", required=True, help="Scaling base directory containing scaling_summary.json")

    # plot_summary4: composite 2x2 figure (speedup, final loss, test curves (log), rmse_vs_time)
    p4 = sub.add_parser("plot_summary4", help="Create a 2x2 summary figure from scaling_summary.json")
    p4.add_argument("--resudir-base", required=True, help="Scaling base directory containing scaling_summary.json")
    p4.add_argument("--logy", action="store_true", help="Use logarithmic Y for test loss curves")

    args = p.parse_args()

    if args.task == "compare":
        task_compare(args.single_dir, args.ddp_dir, args.repeats, args.out_dir)
    elif args.task == "scale":
        g_list = [int(x) for x in str(args.gpus).split(',') if str(x).strip()]
        task_scale(args.resudir_base, g_list, args.repeats, args.basedir, args.casestr, args.launch,
                   args.ddp_baseline, args.equiv, args.base_batch, args.epochs)
    elif args.task == "plot_curves":
        # Load available single/ddp results and create a two-subplot figure
        res_dir = args.resudir
        out_path = args.out or os.path.join(res_dir, 'curves_train_test_log.png')
        try:
            import matplotlib.pyplot as plt
            # Try to load both modes if present
            paths = {
                'single': os.path.join(res_dir, 'training_results_mlp_single.npy'),
                'ddp': os.path.join(res_dir, 'training_results_mlp_ddp.npy'),
            }
            data = {}
            for k, fp in paths.items():
                if os.path.exists(fp):
                    try:
                        data[k] = np.load(fp, allow_pickle=True).item()
                    except Exception:
                        pass
            if not data:
                raise FileNotFoundError('No training_results_mlp_*.npy found in ' + res_dir)
            fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=False)
            colors = {'single': 'tab:blue', 'ddp': 'tab:orange'}
            # Training losses (per-iteration)
            for mode, d in data.items():
                tr = d.get('train_loss')
                if tr is not None and len(tr) > 0:
                    tr_np = np.asarray(tr, dtype=float)
                    axes[0].plot(np.arange(1, tr_np.size+1), tr_np, label=f"{mode.capitalize()} - Train", color=colors.get(mode, None), lw=1.5)
            # Test losses (per-epoch)
            for mode, d in data.items():
                ts = d.get('test_loss')
                if ts is not None and len(ts) > 0:
                    ts_np = np.asarray(ts, dtype=float)
                    axes[1].plot(np.arange(1, ts_np.size+1), ts_np, label=f"{mode.capitalize()} - Test", color=colors.get(mode, None), lw=1.5)
            for ax, title in zip(axes, ("Training Loss", "Test Loss")):
                if args.logy:
                    ax.set_yscale('log')
                ax.set_title(title)
                ax.set_xlabel('Epoch' if title.startswith('Test') else 'Iteration')
                ax.set_ylabel('Loss')
                ax.grid(True, linestyle='--', alpha=0.4)
                ax.legend()
            fig.tight_layout()
            fig.savefig(out_path, dpi=300)
            plt.close(fig)
            print(f"Saved: {out_path}")
        except Exception as e:
            print(f"[ERR] plot_curves failed: {e}")
    elif args.task == "plot_avg_by_g":
        base = args.resudir_base
        summ = os.path.join(base, 'scaling_summary.json')
        try:
            with open(summ, 'r') as f:
                scaling = json.load(f)
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
            colors = plt.cm.tab10.colors
            for idx, e in enumerate(scaling.get('entries', [])):
                g = e.get('gpus')
                trm = e.get('train_loss_curve_mean')
                tsm = e.get('test_loss_curve_mean')
                col = colors[idx % len(colors)]
                if trm is not None:
                    trm_np = np.asarray(trm, dtype=float)
                    axes[0].plot(np.arange(1, trm_np.size+1), trm_np, label=f"g={g}", color=col, lw=1.8)
                if tsm is not None:
                    tsm_np = np.asarray(tsm, dtype=float)
                    axes[1].plot(np.arange(1, tsm_np.size+1), tsm_np, label=f"g={g}", color=col, lw=1.8)
            for ax, title in zip(axes, ("Average Training Loss per GPU count", "Average Test Loss per GPU count")):
                if args.logy:
                    ax.set_yscale('log')
                ax.set_title(title)
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.grid(True)
                ax.legend()
            fig.tight_layout()
            os.makedirs(os.path.join(base, 'figures'), exist_ok=True)
            outp = os.path.join(base, 'figures', 'avg_losses_by_g_log.png' if args.logy else 'avg_losses_by_g.png')
            fig.savefig(outp, dpi=300)
            plt.close(fig)
            print(f"Saved: {outp}")
        except Exception as e:
            print(f"[ERR] plot_avg_by_g failed: {e}")
    elif args.task == "plot_rmse_vs_time":
        base = args.resudir_base
        summ = os.path.join(base, 'scaling_summary.json')
        try:
            with open(summ, 'r') as f:
                scaling = json.load(f)
            import matplotlib.pyplot as plt
            # Extract arrays
            gp = [e.get('gpus') for e in scaling.get('entries', [])]
            tt = [e.get('total_time_s_mean') for e in scaling.get('entries', [])]
            rm = [e.get('rmse_mean') for e in scaling.get('entries', [])]
            gp_arr = np.asarray(gp, dtype=float)
            xdat = np.asarray(tt, dtype=float)
            ydat = np.asarray(rm, dtype=float)
            mask = np.isfinite(xdat) & np.isfinite(ydat)
            if not np.any(mask):
                raise RuntimeError('No finite data points found')
            x = xdat[mask]
            y = ydat[mask]
            gmask = gp_arr[mask]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(x, y, c='tab:blue')
            for xi, yi, gi in zip(x, y, gmask):
                ax.annotate(str(int(gi)), (xi, yi), textcoords='offset points', xytext=(5, 5), fontsize=8)
            # Regression + R^2
            a, b = np.polyfit(x, y, 1)
            y_fit = a * x + b
            ss_res = float(np.nansum((y - y_fit) ** 2))
            ss_tot = float(np.nansum((y - np.nanmean(y)) ** 2) + 1e-12)
            r2 = 1.0 - ss_res / ss_tot
            x_line = np.linspace(0.0, float(np.nanmax(x)), 100)
            y_line = a * x_line + b
            ax.plot(x_line, y_line, color='red', linestyle='--', linewidth=1.0,
                    label=f"Reg: y={a:.3e}x+{b:.3f}, R2={r2:.3f}")
            ax.set_title('RMSE vs Training Time')
            ax.set_xlabel('Total time (s)')
            ax.set_ylabel('RMSE (mean)')
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.legend()
            ax.set_xlim(left=0.0)
            fig.tight_layout()
            # ensure figures dir exists for subcommand outputs
            os.makedirs(os.path.join(base, 'figures'), exist_ok=True)
            outp = os.path.join(base, 'figures', 'rmse_vs_time.png')
            fig.savefig(outp, dpi=300)
            plt.close(fig)
            print(f"Saved: {outp}")
        except Exception as e:
            print(f"[ERR] plot_rmse_vs_time failed: {e}")
    elif args.task == "plot_summary4":
        base = args.resudir_base
        summ = os.path.join(base, 'scaling_summary.json')
        try:
            with open(summ, 'r') as f:
                scaling = json.load(f)
            import matplotlib.pyplot as plt
            entries = scaling.get('entries', [])
            if not entries:
                raise RuntimeError('No entries in scaling_summary.json')
            gp = [e.get('gpus') for e in entries]
            times = [e.get('total_time_s_mean') for e in entries]
            losses = [e.get('final_test_loss_mean') for e in entries]
            rm = [e.get('rmse_mean') for e in entries]
            # Compute speedup vs first entry (or use stored value if present)
            t0 = times[0] if times and times[0] is not None else None
            if t0 is None or t0 <= 0:
                raise RuntimeError('Invalid baseline time')
            speed = []
            for e, t in zip(entries, times):
                sp = e.get(f"speedup_vs_{gp[0]}")
                if sp is None and (t is not None and t > 0):
                    sp = float(t0) / float(t)
                speed.append(sp)
            # Factors for final loss: baseline_loss / loss
            l0 = losses[0] if losses and losses[0] is not None else None
            loss_factor = []
            for l in losses:
                if l0 is not None and l not in (None, 0):
                    loss_factor.append(float(l0) / float(l))
                else:
                    loss_factor.append(None)
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            ax00, ax01, ax10, ax11 = axes.ravel()
            # 1.1 Time speedup
            ax00.bar([str(x) for x in gp], [float(s) if s is not None else np.nan for s in speed])
            for i, s in enumerate(speed):
                if s is not None and np.isfinite(s):
                    ax00.text(i, s, f"×{s:.2f}", ha='center', va='bottom', fontsize=8)
            ax00.set_title('Time Speedup (vs {} GPU)'.format(gp[0]))
            ax00.set_ylabel('× speedup')
            ax00.grid(True, axis='y', linestyle='--', alpha=0.4)
            # 1.2 Final Test Loss
            vals_loss = [float(l) if l is not None else np.nan for l in losses]
            ax01.bar([str(x) for x in gp], vals_loss)
            for i, (v, fct) in enumerate(zip(vals_loss, loss_factor)):
                if np.isfinite(v) and fct is not None and np.isfinite(fct):
                    ax01.text(i, v, f"×{fct:.2f}", ha='center', va='bottom', fontsize=8)
            ax01.set_title('Final Test Loss (mean)')
            ax01.set_ylabel('Loss')
            ax01.grid(True, axis='y', linestyle='--', alpha=0.4)
            # 2.1 Test loss curves (log optional)
            colors = plt.cm.tab10.colors
            for idx, e in enumerate(entries):
                tsm = e.get('test_loss_curve_mean')
                if tsm is not None:
                    y = np.asarray(tsm, dtype=float)
                    ax10.plot(np.arange(1, y.size+1), y, label=f"g={e.get('gpus')}", color=colors[idx % len(colors)])
            ax10.set_title('Average Test Loss per GPU count')
            ax10.set_xlabel('Epoch')
            ax10.set_ylabel('Loss')
            if args.logy:
                ax10.set_yscale('log')
            ax10.grid(True, linestyle='--', alpha=0.4)
            ax10.legend()
            # 2.2 RMSE vs time with regression
            x = np.asarray(times, dtype=float)
            y = np.asarray(rm, dtype=float)
            m = np.isfinite(x) & np.isfinite(y)
            if np.any(m):
                xm = x[m]
                ym = y[m]
                gm = np.asarray(gp, dtype=float)[m]
                ax11.scatter(xm, ym, c='tab:blue')
                for xi, yi, gi in zip(xm, ym, gm):
                    ax11.annotate(str(int(gi)), (xi, yi), textcoords='offset points', xytext=(5,5), fontsize=8)
                if xm.size >= 2:
                    a, b = np.polyfit(xm, ym, 1)
                    y_fit = a * xm + b
                    ss_res = float(np.nansum((ym - y_fit) ** 2))
                    ss_tot = float(np.nansum((ym - np.nanmean(ym)) ** 2) + 1e-12)
                    r2 = 1.0 - ss_res / ss_tot
                    x_line = np.linspace(0.0, float(np.nanmax(xm)), 100)
                    y_line = a * x_line + b
                    ax11.plot(x_line, y_line, color='red', linestyle='--', linewidth=1.0,
                              label=f"Reg: y={a:.3e}x+{b:.3f}, R2={r2:.3f}")
                ax11.set_xlim(left=0.0)
                ax11.grid(True, linestyle='--', alpha=0.4)
                ax11.set_title('RMSE vs Training Time')
                ax11.set_xlabel('Total time (s)')
                ax11.set_ylabel('RMSE (mean)')
                ax11.legend()
            fig.tight_layout()
            os.makedirs(os.path.join(base, 'figures'), exist_ok=True)
            outp = os.path.join(base, 'figures', 'summary_4in1.png')
            fig.savefig(outp, dpi=300)
            plt.close(fig)
            print(f"Saved: {outp}")
        except Exception as e:
            print(f"[ERR] plot_summary4 failed: {e}")
    else:
        p.error("Unknown task")


if __name__ == "__main__":
    main()
