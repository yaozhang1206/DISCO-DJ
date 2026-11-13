"""Plot P(k) files written by run_sim.py (Simulation.power_spectrum), optionally against a linear z=0 table.

  python examples/plot_pk.py runs/*/pk_z0_*.npz --linear Pk_lin_z0.txt --out pk.png
Shot noise is subtracted; curves stop at the particle Nyquist k = pi res / box. With --linear, a ratio panel shows
P / (D^2 P_lin) (D = 1 at z = 0; for IC files the stored 'dplus' is used).
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1", "#d1495b", "#52514e"]
STYLES = ["-", "--", "-.", ":", "-", "--"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--linear", help="linear z=0 P(k) table (k, P)")
    ap.add_argument("--labels", nargs="*", help="legend labels (default: file names)")
    ap.add_argument("--out", default="pk.png")
    a = ap.parse_args()

    lin = np.loadtxt(a.linear) if a.linear else None
    plin = (lambda k: np.exp(np.interp(np.log(k), np.log(lin[:, 0]), np.log(lin[:, 1])))) if a.linear else None
    if a.linear:
        fig, (ax, axr) = plt.subplots(2, 1, figsize=(7.2, 7), sharex=True,
                                      gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.06})
    else:
        fig, ax = plt.subplots(figsize=(7.2, 5)); axr = None
    for i, fn in enumerate(a.files):
        d = np.load(fn)
        k, pk, shot = d["k"], d["pk"], float(d["shot"])
        k_ny = np.pi * float(d["res"]) / float(d["box"])
        m = (k > 0) & (k <= k_ny)
        lab = a.labels[i] if a.labels and i < len(a.labels) else os.path.basename(fn)
        st = dict(color=COLORS[i % len(COLORS)], ls=STYLES[i % len(STYLES)], lw=1.8)
        ax.plot(k[m], pk[m] - shot, label=lab, **st)
        if axr is not None:
            d2 = float(d["dplus"]) ** 2 if "dplus" in d.files else 1.0
            axr.plot(k[m], (pk[m] - shot) / (d2 * plin(k[m])), **st)
    if a.linear:
        kk = np.logspace(np.log10(2e-3), np.log10(5), 300)
        ax.plot(kk, plin(kk), color="#52514e", lw=1.1, ls=":", label="linear (z=0)")
        axr.axhline(1, color="#52514e", lw=0.8, ls=":")
        axr.set_yscale("log"); axr.set_ylabel(r"$P/P_\mathrm{lin}$"); axr.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
    else:
        ax.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylabel(r"$P(k) - 1/\bar n\ \ [(h^{-1}\mathrm{Mpc})^3]$")
    ax.legend(frameon=False, fontsize=8)
    for x in (ax, axr):
        if x is not None:
            x.grid(True, color="#e4e3df", lw=0.6); x.set_axisbelow(True)
    fig.savefig(a.out, dpi=160, bbox_inches="tight")
    print("saved", a.out)


if __name__ == "__main__":
    main()
