#!/usr/bin/env python3
"""
Pilot merge + preprocessing for the Brainrot Predictor (ML4QS).

Joins a Sensor Logger recording (Accelerometer.csv + Gyroscope.csv, 10 Hz, Unix-ns
timestamps in UTC) with the iOS Shortcuts event log (brainrot_tracker.csv, ISO 8601
with timezone offset), bins everything into fixed windows, engineers per-window
features, and assigns the early-warning label: did a TARGET app open during window T+1?

This doubles as the EDA starting point. Run it on one pilot recording to verify the
pipeline before collecting days of data.

Usage:
    python preprocess.py --recording data/2026-06-04_09-13-19 \
                         --events data/brainrot_tracker.csv \
                         --window 180 --out windows.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Substrings (lowercased) that mark a TARGET (doomscroll) app open = the thing we predict.
TARGET_KEYWORDS = ("brainrot", "tiktok", "instagram", "reels")

# Map auxiliary event tags -> coarse category (the four categories used in the report).
# Matched by substring on the lowercased event text. First match wins.
AUX_CATEGORIES = {
    "search": ("search", "gemini", "claude", "chatgpt", "browser", "safari", "chrome"),
    "social": ("social", "whatsapp", "snapchat"),
    "entertainment": ("enter", "netflix", "youtube", "spotify"),
    "other": ("other",),  # generic Other_app_opened bucket -> 'other'
}


# --------------------------------------------------------------------------- #
# Loading                                                                      #
# --------------------------------------------------------------------------- #
def load_sensor(recording_dir: Path) -> pd.DataFrame:
    """Load and merge accelerometer + gyroscope into one 10 Hz frame on a UTC index."""
    frames = []
    for name, prefix in (("Accelerometer.csv", "acc"), ("Gyroscope.csv", "gyr")):
        path = recording_dir / name
        df = pd.read_csv(path)
        # Sensor Logger 'time' is Unix epoch in NANOSECONDS (UTC).
        df["t"] = pd.to_datetime(df["time"], unit="ns", utc=True)
        df = df[["t", "x", "y", "z"]].rename(
            columns={c: f"{prefix}_{c}" for c in ("x", "y", "z")}
        )
        df["magnitude"] = np.sqrt(
            df[f"{prefix}_x"] ** 2 + df[f"{prefix}_y"] ** 2 + df[f"{prefix}_z"] ** 2
        ).rename(f"{prefix}_mag")
        df = df.rename(columns={"magnitude": f"{prefix}_mag"})
        frames.append(df.set_index("t"))

    # Both streams are ~10 Hz but not perfectly co-sampled -> align by nearest within 50 ms.
    acc, gyr = frames
    merged = pd.merge_asof(
        acc.sort_index(),
        gyr.sort_index(),
        left_index=True,
        right_index=True,
        direction="nearest",
        tolerance=pd.Timedelta("50ms"),
    )
    return merged


def classify_event(text: str) -> tuple[str, str]:
    """Return (kind, category) where kind in {'target','aux'}."""
    t = text.lower()
    if any(k in t for k in TARGET_KEYWORDS):
        return "target", "target"
    for cat, keys in AUX_CATEGORIES.items():
        if any(k in t for k in keys):
            return "aux", cat
    return "aux", "other"


def load_events(events_path: Path) -> pd.DataFrame:
    """
    Robust parse of the Shortcuts event log.

    The CSV has an INCONSISTENT column count (some lines carry a stray empty field).
    We therefore split each non-empty line on the FIRST comma only: everything before
    is the ISO timestamp, everything after is the event text.
    """
    rows = []
    for raw in Path(events_path).read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        ts_str, _, rest = line.partition(",")
        text = rest.replace(",", " ").strip()  # collapse the stray empty field
        ts = pd.to_datetime(ts_str.strip(), utc=True)  # ISO 8601 w/ offset -> UTC
        kind, category = classify_event(text)
        rows.append({"t": ts, "text": text, "kind": kind, "category": category})
    ev = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
    return ev


# --------------------------------------------------------------------------- #
# Windowing + features                                                         #
# --------------------------------------------------------------------------- #
def build_windows(
    sensor: pd.DataFrame, events: pd.DataFrame, window_s: int, still_thresh: float
) -> pd.DataFrame:
    """Aggregate into fixed windows and attach the T+1 lookahead label."""
    start = sensor.index.min()
    end = sensor.index.max()
    edges = pd.date_range(start, end, freq=f"{window_s}s", tz="UTC")

    aux_cats = list(AUX_CATEGORIES.keys())
    records = []
    last_target_t: pd.Timestamp | None = None

    for w_start, w_end in zip(edges[:-1], edges[1:]):
        seg = sensor.loc[w_start:w_end]
        ev_in = events[(events["t"] >= w_start) & (events["t"] < w_end)]
        target_in = ev_in[ev_in["kind"] == "target"]

        # --- kinematic features ---
        if len(seg):
            gyr_var = float(seg["gyr_mag"].var(ddof=0))
            gyr_max = float(seg["gyr_mag"].max())
            # stillness: fraction of samples below a static gyro threshold * window length
            still_frac = float((seg["gyr_mag"] < still_thresh).mean())
            stillness_s = still_frac * window_s
            n_samples = len(seg)
        else:
            gyr_var = gyr_max = stillness_s = np.nan
            n_samples = 0

        # --- system / interaction features ---
        aux_in = ev_in[ev_in["kind"] == "aux"]
        cat_counts = {f"{c}_count": int((aux_in["category"] == c).sum()) for c in aux_cats}
        other_apps_opened = int(len(aux_in))
        # switch frequency: transitions between distinct categories within the window
        cats_seq = aux_in.sort_values("t")["category"].tolist()
        app_switches = sum(1 for a, b in zip(cats_seq, cats_seq[1:]) if a != b)

        # --- contextual / temporal ---
        local = w_start.tz_convert("Europe/Amsterdam")
        secs = local.hour * 3600 + local.minute * 60 + local.second
        tod_sin = np.sin(2 * np.pi * secs / 86400)
        tod_cos = np.cos(2 * np.pi * secs / 86400)
        if last_target_t is not None:
            time_since_target = (w_start - last_target_t).total_seconds()
        else:
            time_since_target = np.nan

        records.append(
            {
                "window_start": w_start,
                "n_samples": n_samples,
                "coverage": n_samples / (window_s * 10),  # expected 10 Hz
                "gyro_variance": gyr_var,
                "gyro_max_vector": gyr_max,
                "phone_stillness_duration": stillness_s,
                "other_apps_opened_count": other_apps_opened,
                "app_switch_frequency": app_switches,
                **cat_counts,
                "time_of_day_sin": tod_sin,
                "time_of_day_cos": tod_cos,
                "time_since_last_target_open": time_since_target,
                "_target_opens_this_window": int(len(target_in)),
            }
        )

        # update refractory timer AFTER computing this window's feature
        if len(target_in):
            last_target_t = target_in["t"].max()

    df = pd.DataFrame(records)
    # Early-warning label: 1 if a target opens in the NEXT window (T+1).
    df["label"] = (df["_target_opens_this_window"].shift(-1).fillna(0) > 0).astype(int)
    df = df.iloc[:-1]  # last window has no T+1 -> drop
    return df


# --------------------------------------------------------------------------- #
# Reporting                                                                    #
# --------------------------------------------------------------------------- #
def summarise(sensor: pd.DataFrame, events: pd.DataFrame, windows: pd.DataFrame) -> None:
    dur = (sensor.index.max() - sensor.index.min()).total_seconds()
    print("=" * 64)
    print("PILOT MERGE SUMMARY")
    print("=" * 64)
    print(f"sensor samples      : {len(sensor):,}")
    print(f"recording duration  : {dur/60:.1f} min "
          f"(~{len(sensor)/dur:.1f} Hz effective)")
    print(f"sensor span (UTC)   : {sensor.index.min()}  ->  {sensor.index.max()}")
    print(f"events parsed       : {len(events)}  "
          f"(target={int((events['kind']=='target').sum())}, "
          f"aux={int((events['kind']=='aux').sum())})")
    print("event categories    :",
          events["category"].value_counts().to_dict())
    print("-" * 64)
    pos = int(windows["label"].sum())
    print(f"windows             : {len(windows)}  "
          f"(positives={pos}, ratio={pos/max(len(windows),1):.1%})")
    miss = (windows["coverage"] < 0.9).sum()
    print(f"low-coverage windows: {miss}  (<90% of expected samples)")
    print("-" * 64)
    print("feature preview (first 5 windows):")
    cols = ["window_start", "gyro_variance", "phone_stillness_duration",
            "other_apps_opened_count", "time_since_last_target_open", "label"]
    with pd.option_context("display.width", 120, "display.max_columns", None):
        print(windows[cols].head().to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recording", type=Path,
                    default=Path("data/2026-06-04_09-13-19"),
                    help="Sensor Logger recording folder")
    ap.add_argument("--events", type=Path,
                    default=Path("data/brainrot_tracker.csv"),
                    help="Shortcuts event log CSV")
    ap.add_argument("--window", type=int, default=180,
                    help="window length in seconds (default 180 = 3 min)")
    ap.add_argument("--still-thresh", type=float, default=0.05,
                    help="gyro magnitude below which the phone counts as 'still'")
    ap.add_argument("--out", type=Path, default=Path("windows.csv"))
    args = ap.parse_args()

    sensor = load_sensor(args.recording)
    events = load_events(args.events)
    windows = build_windows(sensor, events, args.window, args.still_thresh)
    summarise(sensor, events, windows)
    windows.drop(columns=["_target_opens_this_window"]).to_csv(args.out, index=False)
    print("-" * 64)
    print(f"wrote {args.out}  ({len(windows)} rows)")


if __name__ == "__main__":
    main()
