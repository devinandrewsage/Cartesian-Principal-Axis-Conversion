"""
batch_principal_axes.py

Run the Cartesian -> principal-axis conversion across many structures in
parallel, and collect QC metadata for the whole run. Supports plain .xyz
files and Gaussian .com/.gjf input files.
"""

import os
import glob
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from principal_axes import to_principal_axes, ATOMIC_MASSES
from gaussian_reader import read_gaussian_com


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
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def write_xyz(path, symbols, coords, comment=""):
    with open(path, "w") as f:
        f.write(f"{len(symbols)}\n{comment}\n")
        for s, (x, y, z) in zip(symbols, coords):
            f.write(f"{s} {x:.6f} {y:.6f} {z:.6f}\n")


def process_one(path, out_dir):
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

        out_path = os.path.join(out_dir, f"{name}_principal.xyz")
        write_xyz(out_path, symbols, new_coords, comment=f"principal-axis frame of {name}")

        moments = info["principal_moments"]
        near_degenerate = False
        for i in range(3):
            for j in range(i + 1, 3):
                if moments[j] > 0 and abs(moments[i] - moments[j]) / moments[j] < 0.02:
                    near_degenerate = True

        wall_time = time.perf_counter() - wall_t0
        cpu_time = time.process_time() - cpu_t0

        return {
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
    except Exception as e:
        return {"name": name, "status": f"error: {e}"}


def estimate_parallel_overhead(n_jobs):
    n_workers = os.cpu_count() if n_jobs == -1 else abs(n_jobs)
    t0 = time.perf_counter()
    Parallel(n_jobs=n_jobs)(delayed(lambda: None)() for _ in range(n_workers))
    return time.perf_counter() - t0


def run_batch(input_dir, out_dir, n_jobs=-1, pattern="*", auto_parallel=True):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(
        glob.glob(os.path.join(input_dir, f"{pattern}.xyz")) +
        glob.glob(os.path.join(input_dir, f"{pattern}.com")) +
        glob.glob(os.path.join(input_dir, f"{pattern}.gjf"))
    )
    print(f"Found {len(files)} structures.")

    if len(files) == 0:
        print(f"No .xyz/.com/.gjf files found in '{input_dir}'. "
              f"Check the path is correct and contains structure files.")
        return pd.DataFrame(columns=["name", "status"])

    effective_n_jobs = n_jobs
    if auto_parallel and len(files) > 0:
        sample_size = min(5, len(files))
        sample_t0 = time.perf_counter()
        for f in files[:sample_size]:
            process_one(f, out_dir)
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
    results = Parallel(n_jobs=effective_n_jobs, verbose=5)(
        delayed(process_one)(f, out_dir) for f in files
    )
    batch_wall_time = time.perf_counter() - batch_wall_t0

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "batch_report.csv"), index=False)

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


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("input_dir")
    p.add_argument("out_dir")
    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument("--no-auto", action="store_true",
                    help="Skip the serial-vs-parallel calibration check and always use --n_jobs as given.")
    args = p.parse_args()
    run_batch(args.input_dir, args.out_dir, args.n_jobs, auto_parallel=not args.no_auto)
