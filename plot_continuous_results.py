#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot continuous PPOTPC permeability/storage results from an output CSV.

This helper is intentionally separate from PPO_perm_process.py so an already
processed CSV can be replotted without launching the full interactive
processing workflow.
"""

import argparse
import os
from pathlib import Path
from tkinter import Tk, filedialog

import numpy as np
import pandas as pd
import matplotlib

# Keep the same GUI backend used by the processing script in WSL/terminal runs.
os.environ.setdefault('MPLBACKEND', 'TkAgg')
try:
    matplotlib.use('TkAgg')
except (ImportError, RuntimeError):
    # Let matplotlib fall back if Tk is unavailable, e.g. for --save in a
    # non-interactive environment.
    pass

import matplotlib.pyplot as plt


REQUIRED_COLUMNS = {
    'Time',
    'Permeability',
    'delk',
    'Storage Capacity',
    'delbeta',
}


def positive_error_band(values, errors):
    """Return lower/upper arrays for log-scale error shading."""
    values = np.asarray(values, dtype=float)
    errors = np.asarray(errors, dtype=float)
    lower = values - errors
    upper = values + errors
    valid = (
        np.isfinite(values) & np.isfinite(errors) &
        np.isfinite(lower) & np.isfinite(upper) &
        (values > 0) & (errors >= 0) & (lower > 0) & (upper > 0)
    )
    return np.where(valid, lower, np.nan), np.where(valid, upper, np.nan)


def has_positive_values(values):
    """Return True if an array has at least one finite positive value."""
    values = np.asarray(values, dtype=float)
    return bool(np.any(np.isfinite(values) & (values > 0)))


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


def plot_permeability_storage_csv(csv_file, save=None, show=True):
    """Plot permeability/storage from a continuous processing CSV."""
    csv_file = Path(csv_file)
    data = pd.read_csv(csv_file)

    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(
            f"{csv_file} does not look like a continuous PPOTPC output CSV. "
            f"Missing columns: {sorted(missing)}"
        )

    time = data['Time'].to_numpy(dtype=float)
    permeability = data['Permeability'].to_numpy(dtype=float)
    permeability_err = data['delk'].to_numpy(dtype=float)
    storage = data['Storage Capacity'].to_numpy(dtype=float)
    storage_err = data['delbeta'].to_numpy(dtype=float)

    perm_plot = np.where(np.isfinite(permeability) & (permeability > 0), permeability, np.nan)
    storage_plot = np.where(np.isfinite(storage) & (storage > 0), storage, np.nan)
    perm_low, perm_high = positive_error_band(permeability, permeability_err)
    storage_low, storage_high = positive_error_band(storage, storage_err)

    fig, ax_perm = plt.subplots(num=20, figsize=(10, 6), clear=True)
    ax_perm.plot(time, perm_plot, 'r', label='Permeability')
    ax_perm.fill_between(time, perm_low, perm_high, color='r', alpha=0.18, linewidth=0)
    ax_perm.set_xlabel('Time (s)')
    ax_perm.tick_params(axis='y', labelcolor='r')
    if has_positive_values(perm_plot):
        ax_perm.set_yscale('log', base=10)
        ax_perm.set_ylabel('Permeability m² (log10)', color='r')
    else:
        ax_perm.plot(time, permeability, 'r', alpha=0.35)
        ax_perm.set_ylabel('Permeability m²', color='r')
        ax_perm.text(
            0.02, 0.95, 'No positive permeability values for log10 scale',
            transform=ax_perm.transAxes, color='r', va='top', fontsize=9,
        )

    ax_storage = ax_perm.twinx()
    ax_storage.plot(time, storage_plot, 'b', label='Storage')
    ax_storage.fill_between(time, storage_low, storage_high, color='b', alpha=0.15, linewidth=0)
    ax_storage.tick_params(axis='y', labelcolor='b')
    if has_positive_values(storage_plot):
        ax_storage.set_yscale('log', base=10)
        ax_storage.set_ylabel('Storage Pa⁻¹ (log10)', color='b')
    else:
        ax_storage.plot(time, storage, 'b', alpha=0.35)
        ax_storage.set_ylabel('Storage Pa⁻¹', color='b')
        ax_storage.text(
            0.98, 0.95, 'No positive storage values for log10 scale',
            transform=ax_storage.transAxes, color='b', va='top', ha='right', fontsize=9,
        )

    ax_perm.set_title(csv_file.name)
    fig.tight_layout()

    if save:
        save = Path(save)
        fig.savefig(save, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save}")

    if show:
        plt.show()

    return fig, ax_perm, ax_storage


def main():
    parser = argparse.ArgumentParser(
        description="Plot permeability/storage from a continuous PPOTPC output CSV."
    )
    parser.add_argument('csv_file', nargs='?', help="Continuous output CSV to plot")
    parser.add_argument('--save', help="Optional image path to save, e.g. plot.png")
    parser.add_argument('--no-show', action='store_true', help="Do not open the interactive plot window")
    args = parser.parse_args()

    csv_file = args.csv_file or choose_csv_file()
    if not csv_file:
        print("No CSV selected; nothing to plot.")
        return

    plot_permeability_storage_csv(csv_file, save=args.save, show=not args.no_show)


if __name__ == '__main__':
    main()
