#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone Figure 1-style overview plot for PPOTPC input data.

Plots upstream/downstream pressure and, when available, velocity (`vel`) on a
secondary y-axis.  The script reads the same data-file-specific `.ini` used by
PPO_perm_process.py, then falls back to root `config.ini`.
"""

import argparse
import configparser
import os
from pathlib import Path
from tkinter import Tk, filedialog

import numpy as np
import scipy.io
import matplotlib

os.environ.setdefault('MPLBACKEND', 'TkAgg')
try:
    matplotlib.use('TkAgg')
except (ImportError, RuntimeError):
    pass

import matplotlib.pyplot as plt


def choose_data_file():
    """Open a file dialog and return the selected data path."""
    root = Tk()
    root.withdraw()
    root.lift()
    root.focus_force()
    try:
        filename = filedialog.askopenfilename(
            title="Select PPOTPC input data file",
            filetypes=[
                ("PPOTPC data", "*.dat *.mat"),
                ("MATLAB files", "*.mat"),
                ("DAT files", "*.dat"),
                ("All files", "*.*"),
            ],
        )
    finally:
        root.destroy()
    return filename


def read_config(datafile, explicit_config=None):
    """Read data-file-specific config, root config.ini, or explicit config."""
    datafile = Path(datafile)
    candidates = []
    if explicit_config:
        candidates.append(Path(explicit_config))
    candidates.extend([datafile.with_suffix('.ini'), Path(__file__).resolve().parent / 'config.ini'])

    config_path = next((path for path in candidates if path.exists()), None)
    config = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
    if config_path is not None:
        config.read(config_path)
    return config, config_path


def cfg(config, section, option, default=None, cast=str):
    """Read one config value with a default."""
    if not config.has_option(section, option):
        return default
    if cast is int:
        return config.getint(section, option)
    if cast is float:
        return config.getfloat(section, option)
    value = config.get(section, option)
    return value.strip() if isinstance(value, str) else value


def mat_vector(mat_data, var_name, scale=1.0, required=True):
    """Return one MATLAB variable as a 1-D float vector."""
    if var_name not in mat_data:
        if required:
            available = sorted(k for k in mat_data if not k.startswith('__'))
            raise KeyError(f"Variable '{var_name}' not found. Available variables: {available}")
        return None
    values = np.asarray(mat_data[var_name]).squeeze()
    if values.ndim != 1:
        raise ValueError(f"Variable '{var_name}' must be a vector after squeezing; got {values.shape}.")
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError(f"Variable '{var_name}' is not numeric.")
    return values.astype(float) * scale


def load_overview_data(datafile, config, args):
    """Load time, upstream pressure, downstream pressure, and optional velocity."""
    datafile = Path(datafile)
    file_mode = cfg(config, 'File', 'mode', default=None, cast=str)
    if file_mode is None:
        file_mode = 'mat' if datafile.suffix.lower() == '.mat' else 'dat'
    file_mode = file_mode.strip().lower()
    if file_mode not in ('dat', 'mat'):
        raise ValueError(f"Invalid file mode '{file_mode}'. Use dat or mat.")

    if file_mode == 'dat':
        header_rows = cfg(config, 'File', 'HeaderRows', default=3, cast=int)
        time_col = cfg(config, 'File', 'time_col', default=0, cast=int)
        pup_col = cfg(config, 'File', 'Pup_col', default=1, cast=int)
        pdwn_col = cfg(config, 'File', 'Pdwn_col', default=2, cast=int)
        vel_col = args.vel_col
        if vel_col is None:
            vel_col = cfg(config, 'File', 'vel_col', default=None, cast=int)

        data = np.loadtxt(datafile, delimiter='\t', skiprows=header_rows)
        time = data[:, time_col]
        pup = data[:, pup_col]
        pdwn = data[:, pdwn_col]
        vel = data[:, vel_col] if vel_col is not None else None
        vel_label = f"column {vel_col}" if vel_col is not None else None
        return time, pup, pdwn, vel, vel_label, file_mode

    try:
        mat_data = scipy.io.loadmat(datafile, squeeze_me=True, struct_as_record=False)
    except NotImplementedError as exc:
        raise RuntimeError(
            "This .mat file appears to be MATLAB v7.3/HDF5. scipy.io.loadmat cannot read it."
        ) from exc

    time_var = cfg(config, 'File', 'time_var', default='Time', cast=str)
    pup_var = cfg(config, 'File', 'Pup_var', default='PumpPressure', cast=str)
    pdwn_var = cfg(config, 'File', 'Pdwn_var', default='Pf', cast=str)
    time_scale = cfg(config, 'File', 'time_scale', default=1.0, cast=float)
    pup_scale = cfg(config, 'File', 'Pup_scale', default=1.0, cast=float)
    pdwn_scale = cfg(config, 'File', 'Pdwn_scale', default=1.0, cast=float)

    vel_var = args.vel_var or cfg(config, 'File', 'vel_var', default='vel', cast=str)
    vel_scale = args.vel_scale
    if vel_scale is None:
        vel_scale = cfg(config, 'File', 'vel_scale', default=1.0, cast=float)

    time = mat_vector(mat_data, time_var, time_scale)
    pup = mat_vector(mat_data, pup_var, pup_scale)
    pdwn = mat_vector(mat_data, pdwn_var, pdwn_scale)
    vel = mat_vector(mat_data, vel_var, vel_scale, required=False) if vel_var else None
    vel_label = vel_var if vel is not None else None

    lengths = {len(time), len(pup), len(pdwn)}
    if vel is not None:
        lengths.add(len(vel))
    if len(lengths) != 1:
        raise ValueError(
            "Loaded vectors do not have the same length: "
            f"time={len(time)}, Pup={len(pup)}, Pdwn={len(pdwn)}, "
            f"vel={len(vel) if vel is not None else 'not plotted'}"
        )

    return time, pup, pdwn, vel, vel_label, file_mode


def plot_figure1(datafile, config=None, config_path=None, args=None, save=None, show=True):
    """Create the Figure 1-style overview plot."""
    if args is None:
        args = argparse.Namespace(vel_var=None, vel_col=None, vel_scale=None, title=None)
    if config is None:
        config, config_path = read_config(datafile)

    time, pup, pdwn, vel, vel_label, file_mode = load_overview_data(datafile, config, args)

    mask = np.ones_like(time, dtype=bool)
    if getattr(args, 'start_time', None) is not None:
        mask &= time >= args.start_time
    if getattr(args, 'end_time', None) is not None:
        mask &= time <= args.end_time
    time = time[mask]
    pup = pup[mask]
    pdwn = pdwn[mask]
    if vel is not None:
        vel = vel[mask]
    if len(time) == 0:
        raise ValueError("No data points remain after applying start/end time limits.")

    max_points = getattr(args, 'max_points', None)
    if max_points is not None and max_points > 0 and len(time) > max_points:
        idx = np.unique(np.linspace(0, len(time) - 1, max_points).round().astype(int))
        time = time[idx]
        pup = pup[idx]
        pdwn = pdwn[idx]
        if vel is not None:
            vel = vel[idx]

    fig, ax_pressure = plt.subplots(num=1, figsize=(12, 5), clear=True)
    ax_pressure.plot(time, pup, color='tab:red', lw=0.9, label='Upstream Pressure')
    ax_pressure.plot(time, pdwn, color='tab:blue', lw=0.9, label='Downstream Pressure')
    ax_pressure.set_xlabel('Time (s)')
    ax_pressure.set_ylabel('Pressure (MPa)')
    ax_pressure.grid(True, linestyle='--', alpha=0.35)

    axes = [ax_pressure]
    if vel is not None:
        ax_vel = ax_pressure.twinx()
        ax_vel.plot(time, vel, color='tab:green', lw=0.8, alpha=0.85, label=f'Velocity ({vel_label})')
        ax_vel.set_ylabel('Velocity', color='tab:green')
        ax_vel.tick_params(axis='y', labelcolor='tab:green')
        ax_vel.spines['right'].set_color('tab:green')
        axes.append(ax_vel)
    else:
        print("No velocity variable/column found; plotting pressure only.")

    lines, labels = [], []
    for axis in axes:
        axis_lines, axis_labels = axis.get_legend_handles_labels()
        lines.extend(axis_lines)
        labels.extend(axis_labels)
    ax_pressure.legend(lines, labels, loc='best')

    title = args.title or f"Figure 1 overview — {Path(datafile).name}"
    if config_path is not None:
        title += f"\nconfig: {Path(config_path).name} | mode: {file_mode}"
    ax_pressure.set_title(title)
    fig.tight_layout()

    if save:
        save = Path(save)
        fig.savefig(save, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save}")

    if show:
        plt.show()

    return fig, axes


def main():
    parser = argparse.ArgumentParser(
        description="Standalone Figure 1-style raw data overview plot, optionally including velocity."
    )
    parser.add_argument('datafile', nargs='?', help="Input .dat or .mat file")
    parser.add_argument('--config', help="Optional explicit .ini config path")
    parser.add_argument('--vel-var', help="MATLAB velocity variable to plot; default from config or 'vel'")
    parser.add_argument('--vel-col', type=int, help="DAT velocity column index to plot")
    parser.add_argument('--vel-scale', type=float, help="Velocity scale factor")
    parser.add_argument('--start-time', type=float, help="Optional start time for plotting")
    parser.add_argument('--end-time', type=float, help="Optional end time for plotting")
    parser.add_argument('--max-points', type=int, default=50000, help="Uniformly downsample plot to this many points; <=0 disables")
    parser.add_argument('--title', help="Optional plot title")
    parser.add_argument('--save', help="Optional image path to save")
    parser.add_argument('--no-show', action='store_true', help="Do not open the interactive plot window")
    args = parser.parse_args()

    datafile = args.datafile or choose_data_file()
    if not datafile:
        print("No data file selected; nothing to plot.")
        return

    config, config_path = read_config(datafile, args.config)
    plot_figure1(datafile, config=config, config_path=config_path, args=args,
                 save=args.save, show=not args.no_show)


if __name__ == '__main__':
    main()
