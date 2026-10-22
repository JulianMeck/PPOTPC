#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experimental ObsPy-preprocessed PPOTPC workflow.

This is intentionally a thin launcher, not a fork of PPO_perm_process.py.
All standard PPOTPC loading, ROI selection, fitting, inversion, output, and
plotting logic is imported from PPO_perm_process.py at runtime.  The hooks in
this file only add optional ObsPy preprocessing immediately before sine fitting.
In continuous mode, each moving window/subwindow is preprocessed independently.

Install ObsPy separately, for example:

    mamba install -c conda-forge obspy

Optional .ini section:

    [ObsPy]
    enabled = true
    detrend = auto            # auto, none, or comma-separated steps: demean, linear
    detrend_channels = downstream  # upstream, downstream, both, or none
    auto_linear_threshold = 0.15
    taper_percentage = 0      # 0 disables tapering
    taper_type = hann
    filter_type = lowpass     # default: keep periods longer than 10 s
    freq = 0.1                # Hz, lowpass cutoff; signal periods should be >15 s
    corners = 4
    zerophase = true          # acausal forward/backward filtering, no phase shift
    restore_mean = true       # preserve pressure level for fluid properties
    verbose = true

Run this script exactly as PPO_perm_process.py.  Config/data/output selection is
unchanged.
"""

from __future__ import annotations

import configparser
import sys
from pathlib import Path

import numpy as np

try:
    from obspy import Trace
except ModuleNotFoundError as exc:
    raise SystemExit(
        "ObsPy is required for PPO_perm_process_obspy.py but is not installed.\n"
        "Install it in the permeability environment, for example:\n"
        "  mamba install -c conda-forge obspy"
    ) from exc

import PPO_perm_process as base


DEFAULT_OBSPY_CONFIG = {
    'enabled': True,
    'detrend': 'auto',
    'detrend_channels': 'downstream',
    'auto_linear_threshold': 0.15,
    'taper_percentage': 0.0,
    'taper_type': 'hann',
    'filter_type': 'lowpass',
    'freqmin': None,
    'freqmax': None,
    'freq': 0.1,
    'corners': 4,
    'zerophase': True,
    'restore_mean': True,
    'verbose': True,
}


_ORIGINAL_LOAD_PROCESSING_CONFIG = base.load_processing_config
_ORIGINAL_PROCESS_SINGLE = base.process_single_measurement
_ORIGINAL_PROCESS_CONTINUOUS = base.process_continuous
_ORIGINAL_SIN_FITS_BOOTSTRAP = base.sin_fits_bootstrap
_ORIGINAL_PLOT_BESTFIT_DEBUG = base.plot_bestfit_debug

_ACTIVE_PARAMS = None
_PRINTED_PREPROCESS_SUMMARY = False


def _get_cfg(config, section, option, cast=str, default=None):
    if not config.has_option(section, option):
        return default
    if cast is bool:
        return config.getboolean(section, option)
    if cast is int:
        return config.getint(section, option)
    if cast is float:
        return config.getfloat(section, option)
    value = config.get(section, option)
    return value.strip() if isinstance(value, str) else value


def read_obspy_config(config_file):
    """Read optional [ObsPy]/[Preprocessing] settings from the active .ini."""
    settings = dict(DEFAULT_OBSPY_CONFIG)
    if not config_file:
        return settings

    config_path = Path(config_file)
    if not config_path.exists():
        return settings

    config = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
    config.read(config_path)
    section = next((name for name in ('ObsPy', 'Preprocessing') if config.has_section(name)), None)
    if section is None:
        return settings

    settings.update({
        'enabled': _get_cfg(config, section, 'enabled', bool, settings['enabled']),
        'detrend': _get_cfg(config, section, 'detrend', str, settings['detrend']).strip().lower(),
        'detrend_channels': _get_cfg(config, section, 'detrend_channels', str, settings['detrend_channels']).strip().lower(),
        'auto_linear_threshold': _get_cfg(config, section, 'auto_linear_threshold', float, settings['auto_linear_threshold']),
        'taper_percentage': _get_cfg(config, section, 'taper_percentage', float, settings['taper_percentage']),
        'taper_type': _get_cfg(config, section, 'taper_type', str, settings['taper_type']).strip().lower(),
        'filter_type': _get_cfg(config, section, 'filter_type', str, settings['filter_type']).strip().lower(),
        'freqmin': _get_cfg(config, section, 'freqmin', float, settings['freqmin']),
        'freqmax': _get_cfg(config, section, 'freqmax', float, settings['freqmax']),
        'freq': _get_cfg(config, section, 'freq', float, settings['freq']),
        'corners': _get_cfg(config, section, 'corners', int, settings['corners']),
        'zerophase': _get_cfg(config, section, 'zerophase', bool, settings['zerophase']),
        'restore_mean': _get_cfg(config, section, 'restore_mean', bool, settings['restore_mean']),
        'verbose': _get_cfg(config, section, 'verbose', bool, settings['verbose']),
    })
    validate_obspy_config(settings)
    return settings


def validate_obspy_config(settings):
    settings['detrend_channels'] = normalize_channel_selection(
        settings['detrend_channels'], option_name='detrend_channels'
    )
    if settings['auto_linear_threshold'] < 0:
        raise ValueError("[ObsPy] auto_linear_threshold must be >= 0.")
    if settings['taper_percentage'] < 0 or settings['taper_percentage'] >= 0.5:
        raise ValueError("[ObsPy] taper_percentage must satisfy 0 <= value < 0.5.")

    filter_type = settings['filter_type']
    if filter_type in ('', 'none', 'off', 'false'):
        settings['filter_type'] = 'none'
    elif filter_type == 'bandpass':
        if settings['freqmin'] is None or settings['freqmax'] is None:
            raise ValueError("[ObsPy] bandpass filtering requires freqmin and freqmax in Hz.")
        if settings['freqmin'] <= 0 or settings['freqmax'] <= settings['freqmin']:
            raise ValueError("[ObsPy] bandpass requires 0 < freqmin < freqmax.")
    elif filter_type in ('lowpass', 'highpass'):
        if settings['freq'] is None or settings['freq'] <= 0:
            raise ValueError(f"[ObsPy] {filter_type} filtering requires freq > 0 in Hz.")
    else:
        raise ValueError("[ObsPy] filter_type must be none, bandpass, lowpass, or highpass.")

    detrend = settings['detrend']
    if detrend in ('', 'none', 'off', 'false'):
        settings['detrend'] = 'none'


def describe_obspy_config(settings):
    if not settings['enabled']:
        return "ObsPy preprocessing disabled"
    if settings['detrend'] == 'auto':
        pieces = [f"detrend=auto(threshold={settings['auto_linear_threshold']})"]
    else:
        detrend_steps = detrend_steps_from_setting(settings['detrend'])
        pieces = [f"detrend={' -> '.join(detrend_steps) if detrend_steps else 'none'}"]
    pieces.append(f"detrend_channels={settings['detrend_channels']}")
    if settings['taper_percentage'] > 0:
        pieces.append(f"taper={settings['taper_percentage']} ({settings['taper_type']})")
    if settings['filter_type'] == 'bandpass':
        pieces.append(f"bandpass={settings['freqmin']}–{settings['freqmax']} Hz")
    elif settings['filter_type'] in ('lowpass', 'highpass'):
        pieces.append(f"{settings['filter_type']}={settings['freq']} Hz")
    else:
        pieces.append("filter=none")
    pieces.append(f"restore_mean={settings['restore_mean']}")
    return "ObsPy preprocessing: " + ", ".join(pieces)


def detrend_steps_from_setting(detrend_setting):
    """Return detrend operations from a single setting or comma-separated list."""
    if detrend_setting is None:
        return []
    text = str(detrend_setting).strip().lower()
    if text in ('', 'none', 'off', 'false'):
        return []
    if text == 'auto':
        return ['auto']
    return [step.strip() for step in text.split(',') if step.strip()]


def normalize_channel_selection(value, option_name='channels'):
    """Normalize a preprocessing channel selector."""
    text = str(value).strip().lower()
    aliases = {
        'all': 'both',
        'both': 'both',
        'upstream': 'upstream',
        'up': 'upstream',
        'pup': 'upstream',
        'downstream': 'downstream',
        'down': 'downstream',
        'dwn': 'downstream',
        'pdwn': 'downstream',
        'none': 'none',
        'off': 'none',
        'false': 'none',
    }
    if text not in aliases:
        raise ValueError(
            f"[ObsPy] {option_name} must be upstream, downstream, both, or none."
        )
    return aliases[text]


def channel_matches_selection(trace, selection):
    """Return True if a Trace channel is selected for a preprocessing step."""
    if selection == 'none':
        return False
    if selection == 'both':
        return True

    channel = str(trace.stats.channel).upper()
    if selection == 'upstream':
        return channel == 'PUP'
    if selection == 'downstream':
        return channel == 'PDWN'
    return False


def load_processing_config_with_obspy(datafile, outfile=None):
    params = _ORIGINAL_LOAD_PROCESSING_CONFIG(datafile, outfile)
    params['obspy_preprocess'] = read_obspy_config(params.get('config_file'))
    return params


def median_delta(time):
    """Return median sample interval for ObsPy Trace metadata."""
    time = np.asarray(time, dtype=float)
    if len(time) < 2:
        raise ValueError("ObsPy preprocessing requires at least two samples.")
    dt = np.diff(time)
    if not np.all(np.isfinite(dt)) or np.any(dt <= 0):
        raise ValueError("ObsPy preprocessing requires strictly increasing finite time values.")
    return float(np.median(dt))


def trace_from_window(data, time, channel):
    """Create an ObsPy Trace from one pressure window."""
    values = np.asarray(data, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{channel} window contains non-finite values before ObsPy preprocessing.")
    tr = Trace(data=values.copy())
    tr.stats.delta = median_delta(time)
    tr.stats.channel = channel
    tr.stats.station = 'PPOTPC'
    return tr


def choose_auto_detrend(trace, settings):
    """Choose demean or linear detrending from the amount of fitted drift."""
    data = np.asarray(trace.data, dtype=float)
    if data.size < 3 or np.nanmax(data) == np.nanmin(data):
        return 'demean'

    x = np.arange(data.size, dtype=float)
    slope, _intercept = np.polyfit(x, data, deg=1)
    fitted_drift = abs(slope * (x[-1] - x[0]))

    # Robust amplitude proxy.  Use peak-to-peak fallback for very quiet traces.
    robust_span = float(np.nanpercentile(data, 95) - np.nanpercentile(data, 5))
    if robust_span <= 0:
        robust_span = float(np.nanmax(data) - np.nanmin(data))
    if robust_span <= 0:
        return 'demean'

    if fitted_drift / robust_span >= settings['auto_linear_threshold']:
        return 'linear'
    return 'demean'


def apply_detrend(trace, settings):
    """Apply configured detrending, including the local auto mode."""
    for detrend_type in detrend_steps_from_setting(settings['detrend']):
        if detrend_type == 'auto':
            detrend_type = choose_auto_detrend(trace, settings)
            trace.stats.ppotpc_auto_detrend = detrend_type
        trace.detrend(type=detrend_type)
    return trace


def ensure_filter_below_nyquist(trace, cutoff_hz, label):
    """Raise a clear error if a configured cutoff is incompatible with sampling."""
    nyquist = 0.5 / float(trace.stats.delta)
    if cutoff_hz >= nyquist:
        raise ValueError(
            f"[ObsPy] {label} cutoff {cutoff_hz:g} Hz must be below Nyquist "
            f"{nyquist:g} Hz for this window (dt={trace.stats.delta:g} s)."
        )


def apply_obspy_to_trace(trace, settings):
    """Apply configured ObsPy operations in-place and return the trace."""
    original_mean = float(np.mean(trace.data))

    if channel_matches_selection(trace, settings['detrend_channels']):
        apply_detrend(trace, settings)

    if settings['taper_percentage'] > 0:
        trace.taper(max_percentage=settings['taper_percentage'], type=settings['taper_type'])

    filter_type = settings['filter_type']
    if filter_type == 'bandpass':
        ensure_filter_below_nyquist(trace, settings['freqmax'], 'bandpass freqmax')
        trace.filter(
            'bandpass',
            freqmin=settings['freqmin'],
            freqmax=settings['freqmax'],
            corners=settings['corners'],
            zerophase=settings['zerophase'],
        )
    elif filter_type in ('lowpass', 'highpass'):
        ensure_filter_below_nyquist(trace, settings['freq'], filter_type)
        trace.filter(
            filter_type,
            freq=settings['freq'],
            corners=settings['corners'],
            zerophase=settings['zerophase'],
        )

    if settings['restore_mean']:
        trace.data = trace.data + (original_mean - float(np.mean(trace.data)))

    return trace


def preprocess_pair(up, dwn, time, params, context="window"):
    """Preprocess upstream/downstream arrays with ObsPy, preserving length/time."""
    settings = (params or {}).get('obspy_preprocess', DEFAULT_OBSPY_CONFIG)
    if not settings.get('enabled', True):
        return np.asarray(up, dtype=float), np.asarray(dwn, dtype=float), False

    up_trace = trace_from_window(up, time, 'PUP')
    dwn_trace = trace_from_window(dwn, time, 'PDWN')
    apply_obspy_to_trace(up_trace, settings)
    apply_obspy_to_trace(dwn_trace, settings)

    up_processed = np.asarray(up_trace.data, dtype=float)
    dwn_processed = np.asarray(dwn_trace.data, dtype=float)
    if len(up_processed) != len(up) or len(dwn_processed) != len(dwn):
        raise ValueError(f"ObsPy preprocessing changed data length in {context}; this is not supported.")
    return up_processed, dwn_processed, True


def print_preprocess_summary_once(params):
    global _PRINTED_PREPROCESS_SUMMARY
    settings = (params or {}).get('obspy_preprocess', DEFAULT_OBSPY_CONFIG)
    if settings.get('verbose', True) and not _PRINTED_PREPROCESS_SUMMARY:
        print("\n" + describe_obspy_config(settings))
        print("ObsPy preprocessing is applied to each ROI/window immediately before sine fitting.")
        _PRINTED_PREPROCESS_SUMMARY = True


def set_active_params(params):
    global _ACTIVE_PARAMS, _PRINTED_PREPROCESS_SUMMARY
    _ACTIVE_PARAMS = params
    _PRINTED_PREPROCESS_SUMMARY = False
    if params is not None:
        print_preprocess_summary_once(params)


def sin_fits_bootstrap_obspy(up, dwn, t, Num, Tmin, Tmax):
    """Preprocess one fit window, then delegate to the standard PPOTPC fit."""
    params = _ACTIVE_PARAMS or {}
    up_proc, dwn_proc, applied = preprocess_pair(up, dwn, t, params, context="sine-fit window")
    if applied:
        print_preprocess_summary_once(params)
    return _ORIGINAL_SIN_FITS_BOOTSTRAP(up_proc, dwn_proc, t, Num, Tmin, Tmax)


def plot_bestfit_debug_obspy(time_window, pup_window, pdwn_window, *args, **kwargs):
    """Show debug_bestfit using the same preprocessed window passed to fitting."""
    params = _ACTIVE_PARAMS or {}
    pup_proc, pdwn_proc, _ = preprocess_pair(pup_window, pdwn_window, time_window, params, context="debug_bestfit")
    return _ORIGINAL_PLOT_BESTFIT_DEBUG(time_window, pup_proc, pdwn_proc, *args, **kwargs)


def process_single_measurement_obspy(datafile, outfile, params, first_loop, nomo_ax):
    set_active_params(params)
    try:
        return _ORIGINAL_PROCESS_SINGLE(datafile, outfile, params, first_loop, nomo_ax)
    finally:
        set_active_params(None)


def process_continuous_obspy(datafile, outfile, params, first_loop, nomo_ax):
    set_active_params(params)
    try:
        return _ORIGINAL_PROCESS_CONTINUOUS(datafile, outfile, params, first_loop, nomo_ax)
    finally:
        set_active_params(None)


def install_obspy_hooks():
    base.load_processing_config = load_processing_config_with_obspy
    base.process_single_measurement = process_single_measurement_obspy
    base.process_continuous = process_continuous_obspy
    base.sin_fits_bootstrap = sin_fits_bootstrap_obspy
    base.plot_bestfit_debug = plot_bestfit_debug_obspy


def main():
    install_obspy_hooks()
    print("\n" + "=" * 60)
    print("EXPERIMENTAL MODE: ObsPy preprocessing before sine fits")
    print("=" * 60)
    return base.main()


if __name__ == '__main__':
    sys.exit(main())
