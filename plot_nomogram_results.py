#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot a Bernabé nomogram from a PPOTPC output CSV.

This utility is standalone: it does not import PPO_perm_process.py, so an
already processed CSV can be replotted without launching the full interactive
processing workflow.
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


REQUIRED_COLUMNS = {'Gain', 'Phase'}


def bern_complex(eta, xi):
    """Exact Bernabé (2006) forward model: returns gain A and phase phi."""
    if xi <= 0 or eta <= 0:
        raise ValueError("eta and xi must be positive")
    if eta * xi < 1e-30:
        gain = 1.0 / np.sqrt(1.0 + (eta / 2.0) ** 2)
        phase = np.arctan(eta / 2.0)
        return gain, phase

    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        s = (1 + 1j) * np.sqrt(xi / eta)
        val = ((1 + 1j) / np.sqrt(eta * xi) * np.sinh(s) + np.cosh(s)) ** (-1)
        gain = np.abs(val)
        phase = np.angle(val)
    if not (np.isfinite(gain) and np.isfinite(phase)):
        raise ValueError("non-finite Bernabé result")

    if phase > 0:
        phase -= 2 * np.pi
    phase = -phase
    return gain, phase


def draw_nomogram(ax, show_legend=True):
    """Draw Bernabé nomogram curves on ax."""
    xi_scan = np.logspace(-5, 1.5, 3000)
    eta_scan = np.logspace(2, -2, 3000)

    for log_eta in np.arange(-1.6, 0.21, 0.2):
        eta = 10 ** log_eta
        phase_curve, gain_curve, prev_phase = [], [], None
        for xi in xi_scan:
            try:
                gain, phase = bern_complex(eta, xi)
            except ValueError:
                continue
            if not (0.001 < gain < 0.9995 and 0 < phase < np.pi):
                continue
            if prev_phase is not None and abs(phase - prev_phase) > 0.15:
                continue
            phase_curve.append(phase)
            gain_curve.append(gain)
            prev_phase = phase
        if len(phase_curve) > 2:
            order = np.argsort(phase_curve)
            phase_sorted = np.asarray(phase_curve)[order]
            log_gain = np.log10(np.asarray(gain_curve)[order])
            ax.plot(phase_sorted, log_gain, color='red', lw=0.9)
            ax.text(
                phase_sorted[0], log_gain[0], f'{log_eta:.1f}',
                color='red', fontsize=7, va='bottom', ha='right', clip_on=True,
            )

    xi_list = [0.002, 0.004, 0.008, 0.016, 0.032,
               0.064, 0.128, 0.256, 0.512, 1, 2, 4, 8, 16]
    for xi in xi_list:
        phase_curve, gain_curve, prev_phase = [], [], None
        for eta in eta_scan:
            try:
                gain, phase = bern_complex(eta, xi)
            except ValueError:
                continue
            if not (0.001 < gain < 0.9995 and 0 < phase < np.pi):
                continue
            if prev_phase is not None and abs(phase - prev_phase) > 0.15:
                continue
            phase_curve.append(phase)
            gain_curve.append(gain)
            prev_phase = phase
        if len(phase_curve) > 2:
            order = np.argsort(phase_curve)
            phase_sorted = np.asarray(phase_curve)[order]
            log_gain = np.log10(np.asarray(gain_curve)[order])
            ax.plot(phase_sorted, log_gain, color='green', lw=0.9)
            if phase_sorted[-1] > 1.5:
                ax.text(
                    phase_sorted[-1], log_gain[-1], f'  {xi}',
                    color='green', fontsize=7, va='center', ha='left', clip_on=True,
                )
            else:
                ax.text(
                    phase_sorted[len(phase_sorted) // 2], -2.02, f'{xi}',
                    color='green', fontsize=7, va='top', ha='center', clip_on=False,
                )

    ax.set_xlabel('Phase Shift (radians)', fontsize=11)
    ax.set_ylabel('Log10(Gain)', fontsize=11)
    ax.set_title('Nomogram — Bernabé (2006) Red = iso-η   |   Green = iso-ξ', fontsize=11)
    ax.set_xlim([0, 3.9])
    ax.set_ylim([-2.05, 0.05])
    ax.grid(True, linestyle='--', alpha=0.4)

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xlabel('Phase Shift rad', fontsize=11)

    if show_legend:
        import matplotlib.lines as mlines
        measured = mlines.Line2D(
            [], [], color='blue', marker='o', linestyle='None',
            markersize=6, label='Measured (φ, log A) ± errors',
        )
        ax.legend(handles=[measured], loc='lower right', fontsize=8, framealpha=0.8)

    return ax


def choose_csv_file():
    """Open a file dialog and return the selected CSV path."""
    root = Tk()
    root.withdraw()
    root.lift()
    root.focus_force()
    try:
        filename = filedialog.askopenfilename(
            title="Select PPOTPC output CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
    finally:
        root.destroy()
    return filename


def plot_nomogram_csv(csv_file, save=None, show=True, show_errors=True, show_legend=True):
    """Plot measured gain/phase from a PPOTPC CSV on a Bernabé nomogram."""
    csv_file = Path(csv_file)
    data = pd.read_csv(csv_file)

    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(
            f"{csv_file} does not look like a PPOTPC output CSV. "
            f"Missing columns: {sorted(missing)}"
        )

    gain = data['Gain'].to_numpy(dtype=float)
    phase = np.abs(data['Phase'].to_numpy(dtype=float))
    valid = np.isfinite(gain) & np.isfinite(phase) & (gain > 0)
    gain = gain[valid]
    phase = phase[valid]
    log_gain = np.log10(gain)

    fig, ax = plt.subplots(num=21, figsize=(9, 7), clear=True)
    draw_nomogram(ax, show_legend=show_legend)

    if show_errors and {'delA', 'delphi'}.issubset(data.columns):
        del_a = data.loc[valid, 'delA'].to_numpy(dtype=float)
        del_phi = data.loc[valid, 'delphi'].to_numpy(dtype=float)
        log_gain_err = np.abs(del_a / gain / np.log(10))
        err_valid = np.isfinite(log_gain_err) & np.isfinite(del_phi)
        ax.errorbar(
            phase[err_valid], log_gain[err_valid],
            xerr=del_phi[err_valid], yerr=log_gain_err[err_valid],
            fmt='o', color='blue', ecolor='0.45', elinewidth=0.8,
            capsize=2, markersize=4, alpha=0.75, label='_nolegend_',
        )
        if np.any(~err_valid):
            ax.plot(
                phase[~err_valid], log_gain[~err_valid], 'o',
                color='blue', markersize=4, alpha=0.75, label='_nolegend_',
            )
    else:
        ax.plot(phase, log_gain, 'o', color='blue', markersize=4, alpha=0.75, label='_nolegend_')

    ax.set_title(f'{ax.get_title()}\n{csv_file.name}', fontsize=11)
    fig.tight_layout()

    if save:
        save = Path(save)
        fig.savefig(save, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save}")

    if show:
        plt.show()

    return fig, ax


def main():
    parser = argparse.ArgumentParser(
        description="Plot measured PPOTPC gain/phase from an output CSV on a Bernabé nomogram."
    )
    parser.add_argument('csv_file', nargs='?', help="PPOTPC output CSV to plot")
    parser.add_argument('--save', help="Optional image path to save, e.g. nomogram.png")
    parser.add_argument('--no-show', action='store_true', help="Do not open the interactive plot window")
    parser.add_argument('--no-errors', action='store_true', help="Plot points without delA/delphi error bars")
    parser.add_argument('--no-legend', action='store_true', help="Hide the measured-point legend")
    args = parser.parse_args()

    csv_file = args.csv_file or choose_csv_file()
    if not csv_file:
        print("No CSV selected; nothing to plot.")
        return

    plot_nomogram_csv(
        csv_file,
        save=args.save,
        show=not args.no_show,
        show_errors=not args.no_errors,
        show_legend=not args.no_legend,
    )


if __name__ == '__main__':
    main()
