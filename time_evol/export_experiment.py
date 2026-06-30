"""Write the README for one packaged time-evolution run.

The plot scripts (plot.py, plot_heatmap.py) write their figures directly into
experiments/<case>/ (axis_plots/, heatmaps_direct/, heatmaps_asymmetry/). This
script is the final packaging step: it reads the run metadata, checks the
figures are present, and writes a LaTeX-ready README.md alongside them.

    cd time_evol && python export_experiment.py
"""
import os
import json
import glob

META_PATH = '../plot/coeff/run_meta.json'
OUT_ROOT  = 'experiments'

AXIS_PLOTS = [
    ('xi_x.png', 'f along the ξ_x axis'),
    ('xi_y.png', 'f along the ξ_y axis'),
    ('xi_z.png', 'f along the ξ_z axis'),
]
HEATMAP_SETS = [
    ('heatmaps_direct',    'Direct f(ξ) slices (xy, xz, yz)'),
    ('heatmaps_asymmetry', 'Asymmetry F(u,v)−F(u,−v) slices (xy, xz, yz)'),
]
MOMENTS_FILES = [
    ('moments/moments.csv', 'Collision-invariant moments (mass, momentum, energy) vs time'),
    ('moments/moments.png', 'Conserved moments vs time (flat curves = conservation)'),
]


def frame_numbers(figdir):
    """Sorted snapshot iterations present in a heatmap dir (from <iter>.png names)."""
    names = [os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(figdir, '*.png'))]
    return sorted(names, key=lambda s: int(s) if s.isdigit() else s)


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
        lines.append(f"- `axis_plots/{fname}` — {caption}.")
    for dst, caption in HEATMAP_SETS:
        frames = ", ".join(heatmap_frames.get(dst, []))
        lines.append(f"- `{dst}/` — {caption}; frames (iteration): {frames}.")
    for fname, caption in MOMENTS_FILES:
        lines.append(f"- `{fname}` — {caption}.")
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
    if not os.path.isdir(out_dir):
        raise SystemExit(
            f"missing {out_dir}/ — run plot.py and plot_heatmap.py (from plot/) first "
            "to generate this case's figures."
        )

    # verify the figures the plot scripts should have produced
    axis_dir = os.path.join(out_dir, 'axis_plots')
    for fname, _caption in AXIS_PLOTS:
        if not os.path.exists(os.path.join(axis_dir, fname)):
            print(f"WARNING: missing axis plot {os.path.join(axis_dir, fname)}")

    heatmap_frames = {}
    for dst_name, _caption in HEATMAP_SETS:
        figdir = os.path.join(out_dir, dst_name)
        if os.path.isdir(figdir):
            heatmap_frames[dst_name] = frame_numbers(figdir)
        else:
            print(f"WARNING: missing heatmap dir {figdir}")
            heatmap_frames[dst_name] = []

    for fname, _caption in MOMENTS_FILES:
        if not os.path.exists(os.path.join(out_dir, fname)):
            print(f"WARNING: missing moments output {os.path.join(out_dir, fname)}")

    write_readme(meta, out_dir, heatmap_frames)
    print(f"\nwrote {out_dir}/README.md for case '{case}'")


if __name__ == '__main__':
    main()
