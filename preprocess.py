import numpy as np
import pandas as pd
import sfdmap
from pathlib import Path
from astropy import units as u
from dust_extinction.parameter_averages import F99
from keras.preprocessing.sequence import pad_sequences

np.int = int  # sfdmap compatibility shim

SFD_MAP_PATH = f"{Path(__file__).resolve().parent}/sfddata-master"
FILTERS_EFF = {"u": 3641, "g": 4704, "r": 6155, "i": 7504, "z": 8695, "y": 10056}
RENAME_MAP = {'oid': 'object_id', 'mjd': 'Time (MJD)', 'band_name': 'Filter',
              'psfFlux': 'Flux', 'psfFluxErr': 'Flux_err'}

_sfd = sfdmap.SFDMap(SFD_MAP_PATH)

def clean_and_rename(df):
    df = df[
        (df["psfFlux_flag"] == False) &
        (df["pixelFlags_bad"] == False) &
        (df["pixelFlags_saturated"] == False)
    ].copy()

    df = df[["oid", "mjd", "band_name", "psfFlux", "psfFluxErr", "ra", "dec"]]
    df[['psfFlux', 'psfFluxErr']] *= 1e-3  # nJy -> uJy
    df.rename(columns=RENAME_MAP, inplace=True)
    return df

def get_ebv(ra, dec):
    return _sfd.ebv(ra, dec)

def add_filters(df):
    df = df.copy()
    filter_order = ["z", "r", "y", "i", "g", "u"]
    filter_cols = []
    for i in filter_order:
        name = f"Filter_{i}_flux"
        filter_cols += [name, f"{name}_err"]
        df[name] = 0.0
        df[f"{name}_err"] = 0.0
        mask = df["Filter"] == i
        df.loc[mask, name] = df.loc[mask, "Flux"]
        df.loc[mask, f"{name}_err"] = df.loc[mask, "Flux_err"]
    df.drop(columns=["Filter", "Flux_err", "Flux"], inplace=True)
    return df, filter_cols

def combine_filters(df, filter_cols):
    return (
        df.set_index(['object_id', 'Time (MJD)'])[filter_cols]
        .replace(0, np.nan)
        .groupby(level=[0, 1])
        .first()
        .fillna(0)
        .reset_index()
    )

def _find_dict(filter_cols, filters):
    return {col: wl for col in filter_cols for f, wl in filters.items() if f"Filter_{f}" in col}

def ebv_correct(df, filter_cols, ebv, Rv=3.1):
    df = df.copy()
    eff = _find_dict(filter_cols, FILTERS_EFF)
    law = F99(Rv=Rv)
    for col, wl in eff.items():
        Av = ebv * Rv
        wl_micron = np.full_like(ebv, wl, dtype=float) * 1e-4 * u.micron
        A_lambda = law(wl_micron) * Av
        df[col] = df[col] * 10**(A_lambda / 2.5)
    return df

import numpy as np

def _point_importance(times, values):
    n = len(times)
    importance = np.full(n, np.inf)
    if n < 3:
        return importance
    t0, t1, t2 = times[:-2], times[1:-1], times[2:]
    v0, v1, v2 = values[:-2], values[1:-1], values[2:]
    area = np.abs((t1 - t0) * (v2 - v0) - (t2 - t0) * (v1 - v0)) / 2
    importance[1:-1] = area
    return importance


def cap_long_seq(df, time_col='Time (MJD)', max_len=50,
                  flux_suffix='_flux', err_suffix='_flux_err'):
    df = df.sort_values(time_col).reset_index(drop=True)

    if len(df) <= max_len:
        return df

    flux_cols = [c for c in df.columns if c.endswith(flux_suffix) and not c.endswith(err_suffix)]
    row_importance = np.zeros(len(df))

    for col in flux_cols:
        mask = df[col] != 0
        idx = df.index[mask]
        if mask.sum() < 3:
            row_importance[idx] = np.inf
            continue
        band_times = df.loc[idx, time_col].to_numpy()
        band_values = df.loc[idx, col].to_numpy()
        band_importance = _point_importance(band_times, band_values)
        row_importance[idx] = np.maximum(row_importance[idx], band_importance)

    row_importance[0] = np.inf
    row_importance[-1] = np.inf

    keep_idx = np.sort(np.argsort(row_importance, kind='stable')[-max_len:])
    return df.iloc[keep_idx].reset_index(drop=True)

def to_model_input(df, feature_cols, max_seq_len=100):
    df = df.sort_values('Time (MJD)')
    X_seq = df[feature_cols].values
    return pad_sequences([X_seq], maxlen=max_seq_len, padding='post', dtype='float32')

def to_model_input_with_time(df, feature_cols, max_seq_len=100):
    df = df.sort_values('Time (MJD)')
    X_seq = df[feature_cols].values

    mjd = df['Time (MJD)'].values
    dt = np.zeros(len(mjd))
    if len(mjd) > 1:
        dt[1:] = np.diff(mjd)
        dt[1:] = (dt[1:] - dt[1:].mean()) / (dt[1:].std() + 1e-8)  # normalize
    dt[0] = -1  # sentinel: mark first timestep as "no previous"
    dt = dt.reshape(-1, 1)

    X_padded = pad_sequences([X_seq], maxlen=max_seq_len, padding='post', dtype='float32')
    dt_padded = pad_sequences([dt], maxlen=max_seq_len, padding='post', value=-1, dtype='float32')

    return np.concatenate([X_padded, dt_padded], axis=-1)