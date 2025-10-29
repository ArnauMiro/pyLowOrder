#!/usr/bin/env python

import os
import sys
import json
import shlex
import argparse
import subprocess as sp
from typing import List, Dict, Tuple

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE = os.path.join(THIS_DIR, "example_MLP_DLR_airfoil.py")


def _run(cmd: List[str]) -> int:
    print("$", " ".join(shlex.quote(c) for c in cmd), flush=True)
    res = sp.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
    return res.returncode


def launch_example(resudir: str, ddp_on: bool, nproc: int, basedir: str, casestr: str, batch_size: int | None = None) -> int:
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
        if batch_size is not None:
            cmd += ["--batch-size", str(int(batch_size))]
    else:
        cmd = [
            sys.executable, EXAMPLE,
            "--ddp", "off",
            "--resudir", resudir,
            "--basedir", basedir,
            "--casestr", casestr,
        ]
        if batch_size is not None:
            cmd += ["--batch-size", str(int(batch_size))]
    return _run(cmd)


def _load_results(resudir: str, mode: str) -> Dict:
    fname = os.path.join(resudir, f"training_results_mlp_{mode}.npy")
    if not os.path.exists(fname):
        raise FileNotFoundError(fname)
    data = np.load(fname, allow_pickle=True).item()
    et = data.get("epoch_time_s")
    tl = data.get("test_loss")
    tr = data.get("train_loss")
    epochs = int(len(et)) if et is not None else (int(len(tl)) if tl is not None else None)
    total_time = data.get("cr_total_time_s")
    if total_time is None and et is not None:
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
        test_loss_curve=np.asarray(test, dtype=float) if len(test) else None,
        train_loss_curve=np.asarray(train, dtype=float) if len(train) else None,
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
        rdir = os.path.join(base_dir, f"run{i}")
        try:
            runs.append(_load_results(rdir, mode))
        except Exception as e:
            print(f"[WARN] Skipping run {i} at {rdir}: {e}")
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
    # attach train/test loss avg curves (aligned)
    def _avg_curve(key):
        curves = [r.get(key) for r in runs if r.get(key) is not None]
        if not curves:
            return None
        min_len = min(c.shape[0] for c in curves)
        if min_len <= 0:
            return None
        arr = np.stack([c[:min_len] for c in curves], axis=0)
        return arr.mean(axis=0).tolist()
    trm = _avg_curve("train_loss_curve")
    tsm = _avg_curve("test_loss_curve")
    if trm is not None:
        summary["train_loss_curve_mean"] = trm
    if tsm is not None:
        summary["test_loss_curve_mean"] = tsm
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resudir-base", required=True)
    p.add_argument("--gpus", default="1,2,3,4")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--basedir", default="/home/airbus/CETACEO_cp_interp/DATA/DLR_pylom/")
    p.add_argument("--casestr", default="NRL7301")
    p.add_argument("--base-batch", type=int, default=119)
    p.add_argument("--launch", action="store_true")
    args = p.parse_args()

    base = args.resudir_base
    g_list = [int(x) for x in str(args.gpus).split(',') if str(x).strip()]
    os.makedirs(base, exist_ok=True)

    # Run and aggregate
    entries, entries_equiv = [], []
    # Baseline time (from g1 DDP after aggregation)
    t0 = None
    for g in g_list:
        # DDP runs under g{g}
        gdir = os.path.join(base, f"g{g}")
        os.makedirs(gdir, exist_ok=True)
        if args.launch:
            for i in range(1, args.repeats + 1):
                rdir = os.path.join(gdir, f"run{i}")
                print(f"[RUN] DDP g={g} i={i} -> {rdir}")
                rc = launch_example(rdir, ddp_on=True, nproc=g, basedir=args.basedir, casestr=args.casestr, batch_size=None)
                if rc != 0:
                    print(f"[ERR] DDP run failed (g={g}, i={i}) rc={rc}")
                    break
        ddp_sum = aggregate_runs(gdir, "ddp", args.repeats)
        entries.append({"gpus": g, **ddp_sum})
        if g == g_list[0]:
            t0 = ddp_sum.get("total_time_s_mean")

        # Equivalent 1-GPU with batch=b*g under g{g}_equiv
        gdir_e = os.path.join(base, f"g{g}_equiv")
        os.makedirs(gdir_e, exist_ok=True)
        if args.launch:
            bs = int(args.base_batch) * int(g)
            for i in range(1, args.repeats + 1):
                rdir = os.path.join(gdir_e, f"run{i}")
                print(f"[RUN][EQUIV] g={g} i={i} (1-GPU, batch={bs}) -> {rdir}")
                rc = launch_example(rdir, ddp_on=False, nproc=1, basedir=args.basedir, casestr=args.casestr, batch_size=bs)
                if rc != 0:
                    print(f"[ERR] Equiv run failed (g={g}, i={i}) rc={rc}")
                    break
        equiv_sum = aggregate_runs(gdir_e, "single", args.repeats)
        entries_equiv.append({"gpus": g, **equiv_sum, "equivalent_batch": int(args.base_batch) * int(g)})

    # Build summary and figures
    scaling = {
        "repeats": args.repeats,
        "base_batch": args.base_batch,
        "entries": entries,
        "entries_equiv": entries_equiv,
    }
    out_json = os.path.join(base, "scaling_summary_equiv.json")
    with open(out_json, "w") as f:
        json.dump(scaling, f, indent=2)
    print(f"Saved: {out_json}")

    # 2x2 summary figure with grouped bars
    try:
        import matplotlib.pyplot as plt
        gp = [e["gpus"] for e in entries]
        # Time speedup
        t_ddp = [e.get("total_time_s_mean") for e in entries]
        t_eq = [e.get("total_time_s_mean") for e in entries_equiv]
        sp_ddp = [float(t0)/float(x) if (t0 and x) else np.nan for x in t_ddp]
        sp_eq = [float(t0)/float(x) if (t0 and x) else np.nan for x in t_eq]
        # Final test loss
        l_ddp = [e.get("final_test_loss_mean") for e in entries]
        l_eq = [e.get("final_test_loss_mean") for e in entries_equiv]
        l0 = l_ddp[0] if l_ddp and l_ddp[0] else None
        f_ddp = [float(l0)/float(x) if (l0 and x) else np.nan for x in l_ddp]
        f_eq = [float(l0)/float(x) if (l0 and x) else np.nan for x in l_eq]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        ax00, ax01, ax10, ax11 = axes.ravel()
        x = np.arange(len(gp))
        w = 0.35
        # 1.1 Speedup grouped bars
        b1 = ax00.bar(x - w/2, sp_ddp, width=w, label='DDP')
        b2 = ax00.bar(x + w/2, sp_eq, width=w, label='1GPU (b×g)')
        ax00.set_xticks(x, [str(g) for g in gp])
        ax00.set_title('Time Speedup (vs {} GPU)'.format(gp[0]))
        ax00.set_ylabel('× speedup')
        ax00.grid(True, axis='y', linestyle='--', alpha=0.4)
        ax00.legend()
        # annotate
        for bars in (b1, b2):
            for bar in bars:
                h = bar.get_height()
                if np.isfinite(h):
                    ax00.text(bar.get_x()+bar.get_width()/2, h, f"×{h:.2f}", ha='center', va='bottom', fontsize=8)

        # 1.2 Final Test Loss grouped bars
        b1 = ax01.bar(x - w/2, l_ddp, width=w, label='DDP')
        b2 = ax01.bar(x + w/2, l_eq, width=w, label='1GPU (b×g)')
        ax01.set_xticks(x, [str(g) for g in gp])
        ax01.set_title('Final Test Loss (mean)')
        ax01.set_ylabel('Loss')
        ax01.grid(True, axis='y', linestyle='--', alpha=0.4)
        ax01.legend()
        # annotate factors (baseline/loss)
        for i, (v, f) in enumerate(zip(l_ddp, f_ddp)):
            if v and np.isfinite(v) and f and np.isfinite(f):
                ax01.text(x[i]-w/2, v, f"×{f:.2f}", ha='center', va='bottom', fontsize=8)
        for i, (v, f) in enumerate(zip(l_eq, f_eq)):
            if v and np.isfinite(v) and f and np.isfinite(f):
                ax01.text(x[i]+w/2, v, f"×{f:.2f}", ha='center', va='bottom', fontsize=8)

        # 2.1 Test loss curves (DDP averages only, log scale)
        colors = plt.cm.tab10.colors
        for idx, e in enumerate(entries):
            tsm = e.get('test_loss_curve_mean')
            if tsm is not None:
                y = np.asarray(tsm, dtype=float)
                ax10.plot(np.arange(1, y.size+1), y, label=f"g={e.get('gpus')}", color=colors[idx % len(colors)])
        ax10.set_title('Average Test Loss per GPU count (DDP)')
        ax10.set_xlabel('Epoch')
        ax10.set_ylabel('Loss')
        ax10.set_yscale('log')
        ax10.grid(True, linestyle='--', alpha=0.4)
        ax10.legend()

        # 2.2 RMSE vs time (DDP)
        x_t = np.asarray(t_ddp, dtype=float)
        y_r = np.asarray([e.get('rmse_mean') for e in entries], dtype=float)
        m = np.isfinite(x_t) & np.isfinite(y_r)
        if np.any(m):
            xm = x_t[m]
            ym = y_r[m]
            gm = np.asarray(gp, dtype=float)[m]
            ax11.scatter(xm, ym, c='tab:blue')
            for xi, yi, gi in zip(xm, ym, gm):
                ax11.annotate(str(int(gi)), (xi, yi), textcoords='offset points', xytext=(5,5), fontsize=8)
            if xm.size >= 2:
                a, b = np.polyfit(xm, ym, 1)
                x_line = np.linspace(0.0, float(np.nanmax(xm)), 100)
                y_line = a * x_line + b
                # R2
                y_fit = a * xm + b
                ss_res = float(np.nansum((ym - y_fit) ** 2))
                ss_tot = float(np.nansum((ym - np.nanmean(ym)) ** 2) + 1e-12)
                r2 = 1.0 - ss_res / ss_tot
                ax11.plot(x_line, y_line, color='red', linestyle='--', linewidth=1.0,
                          label=f"Reg: y={a:.3e}x+{b:.3f}, R2={r2:.3f}")
        ax11.set_xlim(left=0.0)
        ax11.set_ylim(0.0, 0.10)
        ax11.grid(True, linestyle='--', alpha=0.4)
        ax11.set_title('RMSE vs Training Time (DDP)')
        ax11.set_xlabel('Total time (s)')
        ax11.set_ylabel('RMSE (mean)')
        ax11.legend()

        fig.tight_layout()
        outp = os.path.join(base, 'summary_4in1.png')
        fig.savefig(outp, dpi=300)
        plt.close(fig)
        print(f"Saved: {outp}")
    except Exception as e:
        print(f"[WARN] Could not generate summary_4in1.png: {e}")

    # Recover additional figures we had before (DDP-only versions for continuity)
    try:
        import matplotlib.pyplot as plt
        gp = [e["gpus"] for e in entries]
        # Total time (DDP)
        tt = [e.get("total_time_s_mean") for e in entries]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar([str(x) for x in gp], [float(v) if v is not None else np.nan for v in tt])
        ax.set_title("Total Training Time (mean) [DDP]")
        ax.set_xlabel("GPUs")
        ax.set_ylabel("Seconds")
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)
        fig.tight_layout()
        fig.savefig(os.path.join(base, "scaling_time.png"), dpi=300)
        plt.close(fig)

        # Speedup (DDP)
        t0 = tt[0] if tt and tt[0] else None
        spv = [float(t0)/float(x) if (t0 and x) else np.nan for x in tt]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(gp, spv, marker='o')
        ax.set_title(f"Speedup vs {gp[0]} GPU(s) [DDP]")
        ax.set_xlabel("GPUs")
        ax.set_ylabel("Speedup")
        ax.grid(True, linestyle='--', alpha=0.4)
        fig.tight_layout()
        fig.savefig(os.path.join(base, "scaling_speedup.png"), dpi=300)
        plt.close(fig)

        # RMSE vs GPUs (DDP)
        rm = [e.get("rmse_mean") for e in entries]
        rm_std = [e.get("rmse_std") for e in entries]
        rm_arr = np.asarray([float(v) if v is not None else np.nan for v in rm], dtype=float)
        rm_std_arr = np.asarray([float(v) if v is not None else np.nan for v in rm_std], dtype=float)
        gp_arr = np.asarray(gp, dtype=int)
        mask = np.isfinite(rm_arr)
        if np.any(mask):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.errorbar(gp_arr[mask], rm_arr[mask], yerr=rm_std_arr[mask], fmt='-o', capsize=4)
            ax.set_title("RMSE vs GPUs [DDP]")
            ax.set_xlabel("GPUs")
            ax.set_ylabel("RMSE (mean ± std)")
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.set_ylim(0.0, 0.15)
            fig.tight_layout()
            fig.savefig(os.path.join(base, "scaling_rmse.png"), dpi=300)
            plt.close(fig)

        # RMSE vs Time (DDP) standalone
        x = np.asarray([e.get("total_time_s_mean") for e in entries], dtype=float)
        y = np.asarray([e.get("rmse_mean") for e in entries], dtype=float)
        mask2 = np.isfinite(x) & np.isfinite(y)
        if np.any(mask2):
            xm = x[mask2]
            ym = y[mask2]
            gm = np.asarray(gp, dtype=float)[mask2]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(xm, ym, c='tab:blue')
            for xi, yi, gi in zip(xm, ym, gm):
                ax.annotate(str(int(gi)), (xi, yi), textcoords='offset points', xytext=(5,5), fontsize=8)
            if xm.size >= 2:
                a, b = np.polyfit(xm, ym, 1)
                x_line = np.linspace(0.0, float(np.nanmax(xm)), 100)
                y_line = a * x_line + b
                # R2
                y_fit = a * xm + b
                ss_res = float(np.nansum((ym - y_fit) ** 2))
                ss_tot = float(np.nansum((ym - np.nanmean(ym)) ** 2) + 1e-12)
                r2 = 1.0 - ss_res / ss_tot
                ax.plot(x_line, y_line, color='red', linestyle='--', linewidth=1.0,
                        label=f"Reg: y={a:.3e}x+{b:.3f}, R2={r2:.3f}")
            ax.set_xlim(left=0.0)
            ax.set_ylim(0.0, 0.10)
            ax.set_title('RMSE vs Training Time [DDP]')
            ax.set_xlabel('Total time (s)')
            ax.set_ylabel('RMSE (mean)')
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(base, "rmse_vs_time.png"), dpi=300)
            plt.close(fig)

        # Average losses per g (DDP): two figures requested
        try:
            colors = plt.cm.tab10.colors
            # Linear scale
            fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            for idx, e in enumerate(entries):
                trm = e.get('train_loss_curve_mean')
                tsm = e.get('test_loss_curve_mean')
                if trm is not None:
                    y = np.asarray(trm, dtype=float)
                    axes[0].plot(np.arange(1, y.size+1), y, label=f"g={e.get('gpus')}", color=colors[idx % len(colors)])
                if tsm is not None:
                    y = np.asarray(tsm, dtype=float)
                    axes[1].plot(np.arange(1, y.size+1), y, label=f"g={e.get('gpus')}", color=colors[idx % len(colors)])
            axes[0].set_title('Average Train Loss per GPU count [DDP]')
            axes[1].set_title('Average Test Loss per GPU count [DDP]')
            for ax in axes:
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.grid(True, linestyle='--', alpha=0.4)
                ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(base, "avg_losses_by_g.png"), dpi=300)
            plt.close(fig)

            # Log scale
            fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            for idx, e in enumerate(entries):
                trm = e.get('train_loss_curve_mean')
                tsm = e.get('test_loss_curve_mean')
                if trm is not None:
                    y = np.asarray(trm, dtype=float)
                    axes[0].plot(np.arange(1, y.size+1), y, label=f"g={e.get('gpus')}", color=colors[idx % len(colors)])
                if tsm is not None:
                    y = np.asarray(tsm, dtype=float)
                    axes[1].plot(np.arange(1, y.size+1), y, label=f"g={e.get('gpus')}", color=colors[idx % len(colors)])
            axes[0].set_title('Average Train Loss per GPU count [DDP] (log)')
            axes[1].set_title('Average Test Loss per GPU count [DDP] (log)')
            for ax in axes:
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_yscale('log')
                ax.grid(True, linestyle='--', alpha=0.4)
                ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(base, "avg_losses_by_g_log.png"), dpi=300)
            plt.close(fig)
        except Exception:
            pass

        # Per-g true_vs_pred aggregated: DDP and equivalent 1-GPU
        for e, ee in zip(entries, entries_equiv):
            g = e['gpus']
            # DDP
            gdir = os.path.join(base, f"g{g}")
            y_list, p_list = [], []
            for i in range(1, args.repeats + 1):
                rdir = os.path.join(gdir, f"run{i}")
                yp = os.path.join(rdir, "scaled_preds_ddp.npy")
                yt = os.path.join(rdir, "scaled_y.npy")
                if os.path.exists(yp) and os.path.exists(yt):
                    try:
                        y_true = np.load(yt)
                        y_pred = np.load(yp)
                        y_list.append(y_true.reshape(-1))
                        p_list.append(y_pred.reshape(-1))
                    except Exception:
                        pass
            if y_list and p_list:
                y = np.concatenate(y_list)
                p = np.concatenate(p_list)
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(y, p, s=1, alpha=0.3)
                try:
                    a, b = np.polyfit(y, p, 1)
                    x_line = np.linspace(np.nanmin(y), np.nanmax(y), 100)
                    y_line = a * x_line + b
                    y_fit = a * y + b
                    ss_res = float(np.nansum((p - y_fit) ** 2))
                    ss_tot = float(np.nansum((p - np.nanmean(p)) ** 2) + 1e-12)
                    r2 = 1.0 - ss_res / ss_tot
                    ax.plot(x_line, y_line, color='red', linestyle='--', linewidth=1.0,
                            label=f"Reg: y={a:.3f}x+{b:.3f}, R2={r2:.4f}")
                except Exception:
                    pass
                rmse_g = float(np.sqrt(np.nanmean((p - y) ** 2)))
                ax.set_title(f"True vs Pred DDP (g={g}) RMSE={rmse_g:.3e}")
                ax.set_xlabel("True")
                ax.set_ylabel("Pred")
                ax.grid(True)
                ax.legend()
                fig.tight_layout()
                fig.savefig(os.path.join(gdir, "true_vs_pred_ddp_avg.png"), dpi=300)
                plt.close(fig)
            # Per-g curves (DDP): linear + log stacked
            trm = e.get('train_loss_curve_mean')
            tsm = e.get('test_loss_curve_mean')
            if trm is not None or tsm is not None:
                fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
                if trm is not None:
                    y = np.asarray(trm, dtype=float)
                    axes[0].plot(np.arange(1, y.size+1), y, label='Train', color='tab:blue')
                if tsm is not None:
                    y = np.asarray(tsm, dtype=float)
                    axes[0].plot(np.arange(1, y.size+1), y, label='Test', color='tab:orange')
                axes[0].set_title(f'Train/Test Loss (DDP g={g})')
                axes[0].set_xlabel('Epoch')
                axes[0].set_ylabel('Loss')
                axes[0].grid(True, linestyle='--', alpha=0.4)
                axes[0].legend()
                # Log subplot
                if e.get('train_loss_curve_mean') is not None:
                    y = np.asarray(e.get('train_loss_curve_mean'), dtype=float)
                    axes[1].plot(np.arange(1, y.size+1), y, label='Train', color='tab:blue')
                if e.get('test_loss_curve_mean') is not None:
                    y = np.asarray(e.get('test_loss_curve_mean'), dtype=float)
                    axes[1].plot(np.arange(1, y.size+1), y, label='Test', color='tab:orange')
                axes[1].set_title(f'Train/Test Loss (log) (DDP g={g})')
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('Loss')
                axes[1].set_yscale('log')
                axes[1].grid(True, linestyle='--', alpha=0.4)
                axes[1].legend()
                fig.tight_layout()
                fig.savefig(os.path.join(gdir, "curves_train_test_avg.png"), dpi=300)
                plt.close(fig)
            # Equiv 1-GPU
            gdir_e = os.path.join(base, f"g{g}_equiv")
            y_list, p_list = [], []
            for i in range(1, args.repeats + 1):
                rdir = os.path.join(gdir_e, f"run{i}")
                yp = os.path.join(rdir, "scaled_preds_single.npy")
                yt = os.path.join(rdir, "scaled_y.npy")
                if os.path.exists(yp) and os.path.exists(yt):
                    try:
                        y_true = np.load(yt)
                        y_pred = np.load(yp)
                        y_list.append(y_true.reshape(-1))
                        p_list.append(y_pred.reshape(-1))
                    except Exception:
                        pass
            if y_list and p_list:
                y = np.concatenate(y_list)
                p = np.concatenate(p_list)
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(y, p, s=1, alpha=0.3)
                try:
                    a, b = np.polyfit(y, p, 1)
                    x_line = np.linspace(np.nanmin(y), np.nanmax(y), 100)
                    y_line = a * x_line + b
                    y_fit = a * y + b
                    ss_res = float(np.nansum((p - y_fit) ** 2))
                    ss_tot = float(np.nansum((p - np.nanmean(p)) ** 2) + 1e-12)
                    r2 = 1.0 - ss_res / ss_tot
                    ax.plot(x_line, y_line, color='red', linestyle='--', linewidth=1.0,
                            label=f"Reg: y={a:.3f}x+{b:.3f}, R2={r2:.4f}")
                except Exception:
                    pass
                rmse_g = float(np.sqrt(np.nanmean((p - y) ** 2)))
                ax.set_title(f"True vs Pred 1GPU (b×g) g={g} RMSE={rmse_g:.3e}")
                ax.set_xlabel("True")
                ax.set_ylabel("Pred")
                ax.grid(True)
                ax.legend()
                fig.tight_layout()
                fig.savefig(os.path.join(gdir_e, "true_vs_pred_single_avg.png"), dpi=300)
                plt.close(fig)
            # Per-g curves (equiv 1-GPU): linear + log stacked
            trm_e = ee.get('train_loss_curve_mean')
            tsm_e = ee.get('test_loss_curve_mean')
            if trm_e is not None or tsm_e is not None:
                fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
                if trm_e is not None:
                    y = np.asarray(trm_e, dtype=float)
                    axes[0].plot(np.arange(1, y.size+1), y, label='Train', color='tab:blue')
                if tsm_e is not None:
                    y = np.asarray(tsm_e, dtype=float)
                    axes[0].plot(np.arange(1, y.size+1), y, label='Test', color='tab:orange')
                axes[0].set_title(f'Train/Test Loss (1GPU b×g, g={g})')
                axes[0].set_xlabel('Epoch')
                axes[0].set_ylabel('Loss')
                axes[0].grid(True, linestyle='--', alpha=0.4)
                axes[0].legend()
                # Log subplot
                if trm_e is not None:
                    y = np.asarray(trm_e, dtype=float)
                    axes[1].plot(np.arange(1, y.size+1), y, label='Train', color='tab:blue')
                if tsm_e is not None:
                    y = np.asarray(tsm_e, dtype=float)
                    axes[1].plot(np.arange(1, y.size+1), y, label='Test', color='tab:orange')
                axes[1].set_title(f'Train/Test Loss (log) (1GPU b×g, g={g})')
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('Loss')
                axes[1].set_yscale('log')
                axes[1].grid(True, linestyle='--', alpha=0.4)
                axes[1].legend()
                fig.tight_layout()
                fig.savefig(os.path.join(gdir_e, "curves_train_test_avg.png"), dpi=300)
                plt.close(fig)

    except Exception as e:
        print(f"[WARN] Could not regenerate legacy figures: {e}")


if __name__ == "__main__":
    main()
