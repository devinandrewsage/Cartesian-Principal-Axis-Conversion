"""
batch_principal_axes.py

Run the Cartesian -> principal-axis conversion across many structures in
parallel, and collect QC metadata for the whole run. Supports plain .xyz,
Gaussian .com/.gjf, and PDB input files.
"""

import os
import glob
import time
import sys
import platform
import json
from datetime import datetime
import numpy as np
import pandas as pd
import scipy
import joblib as joblib_module
from joblib import Parallel, delayed
from tqdm import tqdm

from principal_axes import (
    to_principal_axes, ATOMIC_MASSES,
    NEAR_DEGENERATE_THRESHOLD, SIGN_AMBIGUOUS_THRESHOLD,
)
from gaussian_reader import read_gaussian_com, write_gaussian_com
from pdb_reader import read_pdb, write_pdb


def write_run_metadata(out_dir, input_dir, n_jobs_requested, effective_n_jobs, n_files):
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "input_dir": os.path.abspath(input_dir),
        "output_dir": os.path.abspath(out_dir),
        "n_files": n_files,
        "n_jobs_requested": n_jobs_requested,
        "n_jobs_used": effective_n_jobs,
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "pandas_version": pd.__version__,
            "joblib_version": joblib_module.__version__,
        },
        "conventions": {
            "near_degenerate_threshold": NEAR_DEGENERATE_THRESHOLD,
            "near_degenerate_description":
                "Two principal moments flagged if within this fraction of each other.",
            "sign_ambiguous_threshold": SIGN_AMBIGUOUS_THRESHOLD,
            "sign_ambiguous_description":
                "Flagged when the top-2 largest-projection atoms on any axis "
                "are within this relative tolerance -- the sign convention "
                "becomes numerically unstable under rotation in this case.",
            "sign_convention_rule":
                "For each principal axis, the atom with largest |projection| "
                "is forced positive.",
            "reflection_rule":
                "If the resulting rotation matrix has det(V) < 0 (a "
                "reflection, not a proper rotation), axis 2 is flipped to "
                "restore det(V) = +1 and preserve molecular chirality.",
        },
    }
    with open(os.path.join(out_dir, "run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def read_xyz(path):
    with open(path) as f:
        lines = f.readlines()
    n = int(lines[0].strip())
    symbols, coords = [], []
    for line in lines[2:2 + n]:
        parts = line.split()
        symbols.append(parts[0])
        coords.append([float(x) for x in parts[1:4]])
    return symbols, np.array(coords)


def read_structure(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".xyz":
        return read_xyz(path)
    elif ext in (".com", ".gjf"):
        return read_gaussian_com(path)
    elif ext == ".pdb":
        symbols, coords, n_ambiguous = read_pdb(path)
        if n_ambiguous > 0:
            import warnings
            warnings.warn(
                f"{path}: {n_ambiguous} atom(s) had no explicit element "
                f"column and used the name-based fallback heuristic -- "
                f"review these if the file is from an unusual source."
            )
        return symbols, coords
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def write_xyz(path, symbols, coords, comment=""):
    with open(path, "w") as f:
        f.write(f"{len(symbols)}\n{comment}\n")
        for s, (x, y, z) in zip(symbols, coords):
            f.write(f"{s} {x:.6f} {y:.6f} {z:.6f}\n")


def process_one(path, out_dir, output_format="xyz", consolidated=False):
    name = os.path.splitext(os.path.basename(path))[0]
    wall_t0 = time.perf_counter()
    cpu_t0 = time.process_time()
    try:
        symbols, coords = read_structure(path)
        unknown = set(symbols) - set(ATOMIC_MASSES.keys())
        if unknown:
            return {"name": name, "status": f"error: unknown elements {unknown}"}
        masses = np.array([ATOMIC_MASSES[s] for s in symbols])

        new_coords, info = to_principal_axes(coords, masses)

        out_path = None
        if not consolidated:
            if output_format == "com":
                out_path = os.path.join(out_dir, f"{name}_principal.com")
                write_gaussian_com(out_path, symbols, new_coords,
                                    title=f"principal-axis frame of {name}")
            elif output_format == "pdb":
                out_path = os.path.join(out_dir, f"{name}_principal.pdb")
                write_pdb(out_path, symbols, new_coords)
            else:
                out_path = os.path.join(out_dir, f"{name}_principal.xyz")
                write_xyz(out_path, symbols, new_coords, comment=f"principal-axis frame of {name}")

        moments = info["principal_moments"]
        near_degenerate = False
        for i in range(3):
            for j in range(i + 1, 3):
                if moments[j] > 0 and abs(moments[i] - moments[j]) / moments[j] < NEAR_DEGENERATE_THRESHOLD:
                    near_degenerate = True

        wall_time = time.perf_counter() - wall_t0
        cpu_time = time.process_time() - cpu_t0

        result = {
            "name": name,
            "status": "ok",
            "n_atoms": len(symbols),
            "com_x": info["com"][0], "com_y": info["com"][1], "com_z": info["com"][2],
            "I1": moments[0], "I2": moments[1], "I3": moments[2],
            "near_degenerate_moments": near_degenerate,
            "reflection_corrected": info["reflection_corrected"],
            "sign_ambiguous": info["sign_ambiguous"],
            "bbox_area_before": info["bbox_area_before"],
            "bbox_area_after": info["bbox_area_after"],
            "wall_time_sec": wall_time,
            "cpu_time_sec": cpu_time,
            "output_path": out_path,
        }
        if consolidated:
            result["_symbols"] = symbols
            result["_coords"] = new_coords
        return result
    except Exception as e:
        return {"name": name, "status": f"error: {e}"}


def write_consolidated_hdf5(results, out_dir):
    import h5py
    path = os.path.join(out_dir, "structures.h5")
    with h5py.File(path, "w") as f:
        for r in results:
            if r.get("status") != "ok" or "_symbols" not in r:
                continue
            grp = f.create_group(r["name"])
            grp.create_dataset("symbols", data=np.array(r["_symbols"], dtype="S4"))
            grp.create_dataset("coords", data=r["_coords"])
            grp.attrs["n_atoms"] = r["n_atoms"]
    return path


def load_config(config_path):
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def estimate_parallel_overhead(n_jobs):
    n_workers = os.cpu_count() if n_jobs == -1 else abs(n_jobs)
    t0 = time.perf_counter()
    Parallel(n_jobs=n_jobs)(delayed(lambda: None)() for _ in range(n_workers))
    return time.perf_counter() - t0


def generate_summary_report(df, metadata, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok_df = df[df["status"] == "ok"]
    n_ok = len(ok_df)
    n_total = len(df)

    chart_path = os.path.join(out_dir, "summary_chart.png")
    if n_ok > 0:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        counts = [
            ok_df["near_degenerate_moments"].sum(),
            ok_df["reflection_corrected"].sum(),
            ok_df["sign_ambiguous"].sum(),
        ]
        labels = ["Near-degenerate\nmoments", "Reflection\ncorrected", "Sign\nambiguous"]
        axes[0].bar(labels, counts, color=["#4C72B0", "#55A868", "#C44E52"])
        axes[0].set_ylabel("Count")
        axes[0].set_title(f"QC flags ({n_ok} structures)")

        axes[1].hist(ok_df["n_atoms"], bins=min(20, n_ok), color="#4C72B0", edgecolor="white")
        axes[1].set_xlabel("Atoms per structure")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Structure size distribution")

        plt.tight_layout()
        plt.savefig(chart_path, dpi=120)
        plt.close(fig)

    report_lines = [
        "# Batch Run Summary Report",
        "",
        f"**Run timestamp:** {metadata['timestamp']}",
        f"**Input directory:** `{metadata['input_dir']}`",
        f"**Structures found:** {metadata['n_files']}",
        f"**Succeeded:** {n_ok} / {n_total}",
        "",
        "## Environment",
        f"- Python {metadata['environment']['python_version'].split()[0]}",
        f"- numpy {metadata['environment']['numpy_version']}, "
        f"scipy {metadata['environment']['scipy_version']}, "
        f"pandas {metadata['environment']['pandas_version']}, "
        f"joblib {metadata['environment']['joblib_version']}",
        f"- Platform: {metadata['environment']['platform']}",
        "",
        "## Analysis conventions used",
        f"- Near-degenerate threshold: {metadata['conventions']['near_degenerate_threshold']} "
        f"({metadata['conventions']['near_degenerate_description']})",
        f"- Sign-ambiguous threshold: {metadata['conventions']['sign_ambiguous_threshold']} "
        f"({metadata['conventions']['sign_ambiguous_description']})",
        f"- Sign convention: {metadata['conventions']['sign_convention_rule']}",
        f"- Reflection rule: {metadata['conventions']['reflection_rule']}",
        "",
    ]

    if n_ok > 0:
        report_lines += [
            "## QC summary",
            f"- Near-degenerate moments flagged: {ok_df['near_degenerate_moments'].sum()} of {n_ok}",
            f"- Reflection corrections applied: {ok_df['reflection_corrected'].sum()} of {n_ok}",
            f"- Sign-ambiguous (unstable) cases: {ok_df['sign_ambiguous'].sum()} of {n_ok}",
            "",
            "![QC summary chart](summary_chart.png)",
            "",
        ]

    if n_total - n_ok > 0:
        report_lines += ["## Failures", ""]
        for _, row in df[df["status"] != "ok"].iterrows():
            report_lines.append(f"- `{row['name']}`: {row['status']}")
        report_lines.append("")

    report_path = os.path.join(out_dir, "summary_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    return report_path


class TqdmParallel(Parallel):
    """
    joblib.Parallel subclass that drives a tqdm progress bar correctly
    under process-based parallelism. A naive approach (updating a tqdm
    object from inside each worker task) fails, because the pbar object
    can't be pickled and sent across process boundaries. This instead
    hooks print_progress(), which joblib itself calls in the PARENT
    process as tasks complete, regardless of backend.
    """
    def __init__(self, total, desc="Converting structures", *args, **kwargs):
        self._tqdm_total = total
        self._tqdm_desc = desc
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(total=self._tqdm_total, desc=self._tqdm_desc, unit="structure") as self._pbar:
            return super().__call__(*args, **kwargs)

    def print_progress(self):
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def run_batch(input_dir, out_dir, n_jobs=-1, pattern="*", auto_parallel=True,
              output_format="xyz", consolidated=False):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(
        glob.glob(os.path.join(input_dir, f"{pattern}.xyz")) +
        glob.glob(os.path.join(input_dir, f"{pattern}.com")) +
        glob.glob(os.path.join(input_dir, f"{pattern}.gjf")) +
        glob.glob(os.path.join(input_dir, f"{pattern}.pdb"))
    )
    print(f"Found {len(files)} structures.")

    if len(files) == 0:
        print(f"No .xyz/.com/.gjf/.pdb files found in '{input_dir}'. "
              f"Check the path is correct and contains structure files.")
        return pd.DataFrame(columns=["name", "status"])

    effective_n_jobs = n_jobs
    if auto_parallel and len(files) > 0:
        sample_size = min(5, len(files))
        sample_t0 = time.perf_counter()
        for f in files[:sample_size]:
            process_one(f, out_dir, output_format, consolidated)
        sample_time = time.perf_counter() - sample_t0
        avg_per_structure = sample_time / sample_size
        est_serial_total = avg_per_structure * len(files)

        overhead = estimate_parallel_overhead(n_jobs)

        print(f"  Calibration: ~{avg_per_structure*1000:.2f} ms/structure, "
              f"est. serial total = {est_serial_total:.3f}s, "
              f"parallel overhead = {overhead:.3f}s")

        if est_serial_total < overhead:
            print(f"  -> Estimated serial time is faster than parallel overhead. "
                  f"Running on 1 core.")
            effective_n_jobs = 1
        else:
            print(f"  -> Parallel overhead pays off at this batch size. "
                  f"Running with n_jobs={n_jobs}.")

    print(f"\nRunning with n_jobs={effective_n_jobs}...")
    batch_wall_t0 = time.perf_counter()

    results = TqdmParallel(total=len(files), n_jobs=effective_n_jobs)(
        delayed(process_one)(f, out_dir, output_format, consolidated) for f in files
    )
    batch_wall_time = time.perf_counter() - batch_wall_t0

    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in results])
    df.to_csv(os.path.join(out_dir, "batch_report.csv"), index=False)

    if consolidated:
        h5_path = write_consolidated_hdf5(results, out_dir)
        print(f"\nConsolidated structures written to {h5_path}")

    metadata = write_run_metadata(out_dir, input_dir, n_jobs, effective_n_jobs, len(files))
    report_path = generate_summary_report(df, metadata, out_dir)
    print(f"Environment/convention metadata written to "
          f"{os.path.join(out_dir, 'run_metadata.json')}")
    print(f"Summary report written to {report_path}")

    n_ok = (df["status"] == "ok").sum()
    n_err = len(df) - n_ok
    print(f"\nDone: {n_ok} succeeded, {n_err} failed.")
    if n_err:
        print("Failures:")
        print(df.loc[df.status != "ok", ["name", "status"]].to_string(index=False))
    if n_ok:
        n_degenerate = df.loc[df.status == "ok", "near_degenerate_moments"].sum()
        n_reflected = df.loc[df.status == "ok", "reflection_corrected"].sum()
        n_ambiguous = df.loc[df.status == "ok", "sign_ambiguous"].sum()
        print(f"Near-degenerate moments flagged: {n_degenerate} of {n_ok}")
        print(f"Reflection corrections applied:  {n_reflected} of {n_ok}")
        print(f"Sign-ambiguous (unstable) cases: {n_ambiguous} of {n_ok}")

        total_cpu_time = df.loc[df.status == "ok", "cpu_time_sec"].sum()
        print(f"\nTiming summary:")
        print(f"  Batch wall-clock time: {batch_wall_time:.3f} sec")
        if effective_n_jobs == 1:
            print(f"  (Ran on 1 core -- no parallel speedup to report; "
                  f"per-task CPU-time sums are not a meaningful comparison "
                  f"when nothing ran in parallel.)")
        else:
            print(f"  Sum of per-structure CPU time: {total_cpu_time:.3f} sec "
                  f"(what a single-core serial run would take)")
            if batch_wall_time > 0:
                print(f"  Effective parallel speedup:    {total_cpu_time/batch_wall_time:.2f}x")
    return df


def main_cli():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("input_dir", nargs="?", default=None)
    p.add_argument("out_dir", nargs="?", default=None)
    p.add_argument("--n_jobs", type=int, default=None)
    p.add_argument("--no-auto", action="store_true",
                    help="Skip the serial-vs-parallel calibration check and always use --n_jobs as given.")
    p.add_argument("--output-format", choices=["xyz", "com", "pdb"], default=None,
                    help="Write output structures as .xyz (default), Gaussian .com, or .pdb files.")
    p.add_argument("--consolidated", action="store_true",
                    help="Write all results into a single structures.h5 file instead of one file per structure.")
    p.add_argument("--config", default=None,
                    help="YAML config file. CLI arguments override config file values.")
    args = p.parse_args()

    settings = {}
    if args.config:
        settings = load_config(args.config)

    input_dir = args.input_dir or settings.get("input_dir")
    out_dir = args.out_dir or settings.get("out_dir")
    if not input_dir or not out_dir:
        p.error("input_dir and out_dir must be given on the command line or in --config.")

    n_jobs = args.n_jobs if args.n_jobs is not None else settings.get("n_jobs", -1)
    output_format = args.output_format or settings.get("output_format", "xyz")
    auto_parallel = not args.no_auto and settings.get("auto_parallel", True)
    consolidated = args.consolidated or settings.get("consolidated", False)

    run_batch(input_dir, out_dir, n_jobs, auto_parallel=auto_parallel,
              output_format=output_format, consolidated=consolidated)


if __name__ == "__main__":
    main_cli()
