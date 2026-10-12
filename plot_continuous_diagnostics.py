#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot continuous PPOTPC gain/phase diagnostics from an output CSV.

Diagnostic panels:
  1. Upstream fitted amplitude through time
  2. Gain through time
  3. Phase shift through time
  4. phi - arccos(Gain) through time

Negative phi - arccos(Gain) means the measured point lies below the xi=0
Bernabé boundary for that gain.
"""

import argparse
import os
from pathlib import Path
from tkinter import Tk, filedialog

import numpy as np
import pandas as pd
import matplotlib

os.environ.setdefault('MPLBACKEND', 'TkAgg')
try:
    matplotlib.use('TkAgg')
except (ImportError, RuntimeError):
    pass

import matplotlib.pyplot as plt


REQUIRED_COLUMNS = {'Time', 'UpAmp', 'Gain', 'Phase'}


def choose_csv_file():
    """Open a file dialog and return the selected CSV path."""
    root = Tk()
    root.withdraw()
    root.lift()
    root.focus_force()
    try:
        filename = filedialog.askopenfilename(
            title="Select continuous PPOTPC output CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
    finally:
        root.destroy()
    return filename


def boundary_margin(gain, phase):
    """Return phi - arccos(gain); negative values are below xi=0 boundary."""
    gain = np.asarray(gain, dtype=float)
    phase = np.abs(np.asarray(phase, dtype=float))
    margin = np.full_like(gain, np.nan, dtype=float)
    valid = np.isfinite(gain) & np.isfinite(phase) & (gain > 0) & (gain < 1)
    margin[valid] = phase[valid] - np.arccos(gain[valid])
    return margin


def plot_diagnostics_csv(csv_file, save=None, show=True):
    """Plot continuous gain/phase diagnostics from a PPOTPC output CSV."""
    csv_file = Path(csv_file)
    data = pd.read_csv(csv_file)

    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(
            f"{csv_file} does not look like a continuous PPOTPC output CSV. "
            f"Missing columns: {sorted(missing)}"
        )

    time = data['Time'].to_numpy(dtype=float)
    up_amp = data['UpAmp'].to_numpy(dtype=float)
    gain = data['Gain'].to_numpy(dtype=float)
    phase = np.abs(data['Phase'].to_numpy(dtype=float))
    margin = boundary_margin(gain, phase)

    below = np.isfinite(margin) & (margin < 0)
    above = np.isfinite(margin) & (margin >= 0)

    fig, axes = plt.subplots(4, 1, num=22, figsize=(11, 9), sharex=True, clear=True)

    axes[0].plot(time, up_amp, color='tab:red', lw=1.0)
    axes[0].set_ylabel('UpAmp')
    axes[0].set_title(f'Continuous diagnostics — {csv_file.name}')

    axes[1].plot(time, gain, color='tab:blue', lw=1.0)
    axes[1].axhline(1.0, color='0.5', ls='--', lw=0.8)
    axes[1].set_ylabel('Gain A')

    axes[2].plot(time, phase, color='tab:purple', lw=1.0)
    axes[2].set_ylabel('|Phase| rad')

    axes[3].axhline(0.0, color='k', lw=0.9)
    axes[3].fill_between(
        time, margin, 0, where=below, interpolate=True,
        color='tab:red', alpha=0.18, label='Below xi=0 boundary',
    )
    axes[3].plot(time[below], margin[below], '.', color='tab:red', ms=4, label='Below xi=0')
    axes[3].plot(time[above], margin[above], '.', color='tab:green', ms=4, label='At/above xi=0')
    axes[3].set_ylabel('phi - arccos(A) rad')
    axes[3].set_xlabel('Time (s)')
    axes[3].legend(loc='best', fontsize=8)

    for ax in axes:
        ax.grid(True, linestyle='--', alpha=0.35)

    n_valid = np.count_nonzero(np.isfinite(margin))
    n_below = np.count_nonzero(below)
    if n_valid:
        fig.text(
            0.01, 0.01,
            f'{n_below}/{n_valid} windows below xi=0 boundary '
            f'({100 * n_below / n_valid:.1f}%).',
            fontsize=9,
        )

    fig.tight_layout(rect=[0, 0.03, 1, 1])

    if save:
        save = Path(save)
        fig.savefig(save, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save}")

    if show:
        plt.show()

    return fig, axes


def main():
    parser = argparse.ArgumentParser(
        description="Plot UpAmp/Gain/Phase and xi=0-boundary diagnostics from a continuous PPOTPC CSV."
    )
    parser.add_argument('csv_file', nargs='?', help="Continuous output CSV to plot")
    parser.add_argument('--save', help="Optional image path to save, e.g. diagnostics.png")
    parser.add_argument('--no-show', action='store_true', help="Do not open the interactive plot window")
    args = parser.parse_args()

    csv_file = args.csv_file or choose_csv_file()
    if not csv_file:
        print("No CSV selected; nothing to plot.")
        return

    plot_diagnostics_csv(csv_file, save=args.save, show=not args.no_show)


if __name__ == '__main__':
    main()
