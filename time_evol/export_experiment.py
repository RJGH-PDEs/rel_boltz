"""Package one time-evolution run into a self-contained, LaTeX-ready folder.

Run from time_evol/ AFTER time_ev.py (writes ../plot/coeff/run_meta.json) and
the plot scripts (plot.py, plot_heatmap.py) have produced their figures. Reads
run_meta.json, copies the figures, and writes a README.md describing the run.

    cd time_evol && python export_experiment.py
"""
import os
import json
import glob
import shutil

META_PATH = '../plot/coeff/run_meta.json'
FIG_ROOT  = '../plot/figures'
OUT_ROOT  = 'experiments'

# (source figure dir, destination subdir, caption template) for the figure sets.
AXIS_PLOTS = [
    ('axis_plots/xi_x.png', 'f along the ξ_x axis'),
    ('axis_plots/xi_y.png', 'f along the ξ_y axis'),
    ('axis_plots/xi_z.png', 'f along the ξ_z axis'),
]
HEATMAP_SETS = [
    ('heatmap_evolution_direct',    'heatmaps_direct',    'Direct f(ξ) slices (xy, xz, yz)'),
    ('heatmap_evolution_asymmetry', 'heatmaps_asymmetry', 'Asymmetry F(u,v)−F(u,−v) slices (xy, xz, yz)'),
]


def copy_glob(src_dir, dst_dir):
    """Copy every .png from src_dir into dst_dir (created if needed). Returns sorted basenames."""
    os.makedirs(dst_dir, exist_ok=True)
    names = []
    for src in glob.glob(os.path.join(src_dir, '*.png')):
        shutil.copy2(src, dst_dir)
        names.append(os.path.basename(src))
    return sorted(names, key=lambda s: int(os.path.splitext(s)[0]) if s[0].isdigit() else s)


def write_readme(meta, out_dir, heatmap_frames):
    case = meta['case']
    lines = []
    lines.append(f"# Experiment: {meta['label']} (`{case}`)\n")
    lines.append(meta['significance'] + "\n")

    lines.append("## Parameters\n")
    lines.append("| parameter | value |")
    lines.append("|---|---|")
    lines.append(f"| initial time t0 | {meta['t0']} |")
    lines.append(f"| time step dt | {meta['dt']} |")
    lines.append(f"| iterations | {meta['num_iterations']} |")
    lines.append(f"| snapshot interval | {meta['save_every']} |")
    lines.append(f"| spectral order n | {meta['n']} |")
    lines.append(f"| n_laguerre | {meta['n_laguerre']} |")
    lines.append(f"| n_lebedev | {meta['n_lebedev']} |")
    lines.append("")

    lines.append("## Initial condition\n")
    lines.append("Coefficient vector f (basis index → (k, l, m) → value); all other entries zero.\n")
    lines.append("| index | (k, l, m) | value |")
    lines.append("|---|---|---|")
    for i, klm, val in meta['ic_entries']:
        k, l, m = klm
        lines.append(f"| {i} | ({k}, {l}, {m}) | {val:.6g} |")
    lines.append("")

    lines.append("## Figures\n")
    for fname, caption in AXIS_PLOTS:
        base = os.path.basename(fname)
        lines.append(f"- `axis_plots/{base}` — {caption}.")
    for _src, dst, caption in HEATMAP_SETS:
        frames = ", ".join(heatmap_frames.get(dst, []))
        lines.append(f"- `{dst}/` — {caption}; frames (iteration): {frames}.")
    lines.append("")

    with open(os.path.join(out_dir, 'README.md'), 'w') as f:
        f.write("\n".join(lines))


def main():
    if not os.path.exists(META_PATH):
        raise SystemExit(f"missing {META_PATH} — run time_ev.py (with save=True) first")
    with open(META_PATH) as f:
        meta = json.load(f)

    case = meta['case']
    out_dir = os.path.join(OUT_ROOT, case)
    os.makedirs(out_dir, exist_ok=True)

    # axis plots
    axis_dst = os.path.join(out_dir, 'axis_plots')
    os.makedirs(axis_dst, exist_ok=True)
    for rel, _caption in AXIS_PLOTS:
        src = os.path.join(FIG_ROOT, rel)
        if os.path.exists(src):
            shutil.copy2(src, axis_dst)
        else:
            print(f"WARNING: missing axis plot {src}")

    # heatmap sets (both views)
    heatmap_frames = {}
    for src_name, dst_name, _caption in HEATMAP_SETS:
        src_dir = os.path.join(FIG_ROOT, src_name)
        dst_dir = os.path.join(out_dir, dst_name)
        if os.path.isdir(src_dir):
            names = copy_glob(src_dir, dst_dir)
            heatmap_frames[dst_name] = [os.path.splitext(n)[0] for n in names]
        else:
            print(f"WARNING: missing heatmap dir {src_dir}")
            heatmap_frames[dst_name] = []

    write_readme(meta, out_dir, heatmap_frames)
    print(f"\npackaged case '{case}' -> {out_dir}/")
    print("  README.md, axis_plots/, heatmaps_direct/, heatmaps_asymmetry/")


if __name__ == '__main__':
    main()
