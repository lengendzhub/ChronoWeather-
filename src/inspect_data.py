"""Phase A: profile the raw Global Weather Repository CSV.

Why this module exists
----------------------
We must not guess the schema, date range, row count, units, or data quality of
the Kaggle file. This module inspects the *real* file and reports facts before
any cleaning or modelling decision is made. It never modifies the raw CSV.

Usage
-----
    python src/inspect_data.py
    python src/inspect_data.py "D:/path/to/GlobalWeatherRepository.csv"

The printed profile is also written to ``reports/data_profile.txt``.
"""

from __future__ import annotations

import io
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.load_data import ensure_raw_csv, sha256_file  # noqa: E402

DEFAULT_CSV = PROJECT_ROOT / "data" / "GlobalWeatherRepository.csv"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "data_profile.txt"

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 60)
pd.set_option("display.max_rows", 200)

# Column-name hints. They are only hints: every match is verified by parsing
# or by looking at the actual values in the report.
TIME_HINTS = ("last_updated", "date", "time", "timestamp")

# Minimum complete pairs needed before trusting a correlation-based check.
MIN_CORR_ROWS = 30

LOCATION_PATTERNS = {
    "city": r"(^|_)(city|location|station)($|_)",
    "country": r"(^|_)country($|_)",
    "region": r"(^|_)(region|state|province)($|_)",
    "continent": r"(^|_)continent($|_)",
    "latitude": r"(^|_)lat(itude)?($|_)",
    "longitude": r"(^|_)(lon|longitude|lng)($|_)",
}

# Exact linear conversions we may find between duplicate unit columns.
KNOWN_CONVERSIONS = [
    (1.8, "F = 1.8 * C + 32 (Celsius vs Fahrenheit)"),
    (0.5555556, "C = (F - 32) / 1.8 (Fahrenheit vs Celsius)"),
    (1.609344, "km = 1.609344 * miles"),
    (0.6213712, "miles = 0.6213712 * km"),
    (25.4, "mm = 25.4 * inches"),
    (0.0393701, "inches = 0.0393701 * mm"),
    (33.86389, "hPa = 33.86389 * inHg"),
    (0.0295300, "inHg = 0.0295300 * hPa"),
    (3.280840, "ft = 3.280840 * m"),
    (0.304800, "m = 0.304800 * ft"),
    (3.6, "km/h = 3.6 * m/s"),
    (0.2777778, "m/s = 0.2777778 * km/h"),
    (2.236936, "mph = 2.236936 * m/s"),
    (0.447040, "m/s = 0.447040 * mph"),
    (1.943844, "knots = 1.943844 * m/s"),
    (0.514444, "m/s = 0.514444 * knots"),
    (10.0, "mm = 10 * cm"),
    (0.1, "cm = 0.1 * mm"),
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_raw(path: Path) -> pd.DataFrame:
    """Read the CSV exactly as stored, without date parsing or type coercion.

    We read once with ``low_memory=False`` so pandas does not change dtypes
    between chunks. Parsing dates is deliberately left to the inspection step,
    because guessing the format can silently destroy rows.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"CSV not found: {path}\n"
            "Download GlobalWeatherRepository.csv from Kaggle and place it in "
            "the data/ folder. See data/README.md."
        )
    return pd.read_csv(path, low_memory=False)


# ---------------------------------------------------------------------------
# Basic schema tables
# ---------------------------------------------------------------------------
def column_overview(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: dtype, missingness, cardinality, first value."""
    rows = []
    for col in df.columns:
        s = df[col]
        non_null = int(s.notna().sum())
        example = ""
        if non_null:
            example = str(s.dropna().iloc[0])[:40]
        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "non_null": non_null,
                "nulls": int(s.isna().sum()),
                "null_%": round(100 * s.isna().mean(), 2),
                "n_unique": int(s.nunique(dropna=True)),
                "example": example,
            }
        )
    return pd.DataFrame(rows)


def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    """Missing values per column, sorted by size."""
    out = pd.DataFrame({"missing": df.isna().sum()})
    out["pct"] = (100 * out["missing"] / len(df)).round(2)
    out = out[out["missing"] > 0].sort_values("missing", ascending=False)
    return out


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Distribution summary for every numeric column.

    Includes 1% and 99% quantiles because they reveal impossible extremes
    without being as sensitive as min/max.
    """
    num = df.select_dtypes(include=np.number)
    if num.empty:
        return pd.DataFrame()
    desc = num.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
    desc["negative"] = (num < 0).sum().astype(int)
    desc["zeros"] = (num == 0).sum().astype(int)
    return desc.round(3)


def constant_columns(df: pd.DataFrame) -> list[tuple[str, object]]:
    """Columns with zero or one distinct value (including NaN).

    A constant column carries no information for EDA or modelling.
    """
    counts = df.nunique(dropna=False)
    result = []
    for col in df.columns:
        if counts[col] <= 1:
            value = df[col].dropna().iloc[0] if df[col].notna().any() else "all NaN"
            result.append((col, value))
    return result


def duplicate_summary(
    df: pd.DataFrame,
    time_col: str | None = None,
    entity_cols: str | list[str] | None = None,
) -> dict[str, int]:
    """Count exact duplicate rows and duplicate keys.

    Entity + time is the key we expect for one weather reading per place per
    timestamp. ``entity_cols`` should include both country and city, because
    city names such as "San Jose" repeat across countries.
    """
    out = {"exact_duplicate_rows": int(df.duplicated().sum())}
    if time_col is not None:
        out["duplicate_timestamp_rows"] = int(
            df.duplicated(subset=[time_col], keep="first").sum()
        )
    if entity_cols is not None:
        if isinstance(entity_cols, str):
            entity_cols = [entity_cols]
        if time_col is not None and entity_cols:
            out["duplicate_entity_timestamp_rows"] = int(
                df.duplicated(subset=entity_cols + [time_col], keep="first").sum()
            )
    return out


# ---------------------------------------------------------------------------
# Time handling
# ---------------------------------------------------------------------------
def parse_datetimes(series: pd.Series) -> pd.Series:
    """Parse a column to datetimes; unparseable values become NaT.

    ``format="mixed"`` lets pandas infer per-row formats, which is safer for an
    unknown CSV than forcing one format. If the column mixes time zones, the
    fallback parses as UTC and drops the timezone label (keeping wall time).
    """
    try:
        parsed = pd.to_datetime(series, errors="coerce", format="mixed")
        # Mixed timezone offsets produce an object dtype in some pandas
        # versions. Re-parse as UTC and drop the label so `.dt` always works.
        if not pd.api.types.is_datetime64_any_dtype(parsed):
            parsed = pd.to_datetime(series, errors="coerce", utc=True)
            parsed = parsed.dt.tz_localize(None)
        return parsed
    except (ValueError, TypeError):
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
        return parsed.dt.tz_localize(None)


def find_time_candidates(df: pd.DataFrame) -> list[str]:
    """Columns whose names suggest they contain dates or timestamps."""
    found = []
    for col in df.columns:
        name = col.lower()
        # "timezone" contains "time" but is not a timestamp column.
        if "timezone" in name or name == "tz":
            continue
        # Epoch columns store seconds since 1970; parsing them as nanoseconds
        # would silently produce 1970 timestamps. The brief asks for the
        # human-readable `last_updated` column, so epoch is not a candidate.
        if name.endswith("_epoch") or name == "epoch":
            continue
        if any(hint in name for hint in TIME_HINTS):
            found.append(col)
    return found


def time_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Parse each time-like column and report coverage and quality."""
    rows = []
    for col in find_time_candidates(df):
        parsed = parse_datetimes(df[col])
        valid = parsed.notna()
        if valid.any():
            vmin, vmax = parsed[valid].min(), parsed[valid].max()
            span_days = (vmax - vmin).days
        else:
            vmin = vmax = pd.NaT
            span_days = np.nan
        rows.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "parse_%": round(100 * valid.mean(), 2),
                "n_unique": int(parsed[valid].nunique()),
                "min": vmin,
                "max": vmax,
                "span_days": span_days,
            }
        )
    return pd.DataFrame(rows)


def pick_time_column(df: pd.DataFrame) -> str | None:
    """Choose the best time index: prefer ``last_updated``, else highest coverage."""
    candidates = find_time_candidates(df)

    # The brief explicitly asks for `last_updated`. Use the exact column
    # whenever it is usable, before considering any other date-like column.
    exact = [c for c in candidates if c.lower() == "last_updated"]
    preferred = exact + [c for c in candidates if c not in exact and "last_updated" in c.lower()]
    for col in preferred:
        parsed = parse_datetimes(df[col])
        if parsed.notna().mean() >= 0.95 and parsed.nunique() > 1:
            return col

    best, best_rate = None, 0.0
    for col in candidates:
        parsed = parse_datetimes(df[col])
        rate = parsed.notna().mean()
        if rate > best_rate and parsed.nunique() > 1:
            best, best_rate = col, rate
    # Require most values to parse before trusting the column.
    return best if best_rate >= 0.95 else None


def coverage_details(df: pd.DataFrame, time_col: str) -> list[str]:
    """Human-readable facts about the time index (gap and cadence checks)."""
    parsed = parse_datetimes(df[time_col])
    valid = parsed.dropna()
    if valid.empty:
        return ["No parseable timestamps."]

    lines = []
    lines.append(f"Parsed values: {len(valid):,} / {len(df):,}")
    lines.append(f"Range: {valid.min()}  ->  {valid.max()}")
    lines.append(f"Span: {(valid.max() - valid.min()).days} days")
    lines.append(f"Unique timestamps: {valid.nunique():,}")

    per_day = valid.dt.floor("D").value_counts().sort_index()
    lines.append(
        "Rows per calendar day: "
        f"min={per_day.min():,}, median={per_day.median():.0f}, "
        f"max={per_day.max():,}, days={len(per_day):,}"
    )

    all_days = pd.date_range(per_day.index.min(), per_day.index.max(), freq="D")
    missing_days = all_days.difference(per_day.index)
    lines.append(f"Calendar days with zero rows: {len(missing_days):,}")

    top_ts = valid.value_counts().head(5)
    lines.append(
        "Most common timestamps: "
        + ", ".join(f"{ts} ({n:,} rows)" for ts, n in top_ts.items())
    )

    # Cadence for a few cities shows whether readings are daily or sub-daily.
    location_cols = detect_location_columns(df)
    entity = location_cols["city"][0] if location_cols["city"] else None
    if entity is not None:
        for value in df[entity].dropna().astype(str).unique()[:3]:
            sub = df[df[entity].astype(str) == value]
            t = parse_datetimes(sub[time_col]).dropna().sort_values()
            if len(t) < 3:
                continue
            gaps_days = t.diff().dropna().dt.total_seconds() / 86400
            common = gaps_days.round(2).value_counts().head(3)
            lines.append(
                f"Cadence for {value!r} ({len(t)} rows): "
                + ", ".join(f"{gap:g} d x {n}" for gap, n in common.items())
            )
    return lines


# ---------------------------------------------------------------------------
# Location columns
# ---------------------------------------------------------------------------
def detect_location_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """Find columns that likely describe where a reading came from."""
    found: dict[str, list[str]] = {role: [] for role in LOCATION_PATTERNS}
    for col in df.columns:
        name = col.lower()
        for role, pattern in LOCATION_PATTERNS.items():
            if re.search(pattern, name):
                found[role].append(col)
    return found


def location_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Cardinality and top values for location-like columns."""
    rows = []
    for role, cols in detect_location_columns(df).items():
        for col in cols:
            s = df[col].dropna()
            row = {
                "role": role,
                "column": col,
                "non_null": int(len(s)),
                "n_unique": int(s.nunique()),
                "min": "",
                "max": "",
                "top_values": "",
            }
            if pd.api.types.is_numeric_dtype(s):
                row["min"] = round(float(s.min()), 4)
                row["max"] = round(float(s.max()), 4)
            else:
                row["top_values"] = ", ".join(map(str, s.value_counts().head(5).index))
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Redundancy and plausibility checks
# ---------------------------------------------------------------------------
def _match_conversion(slope: float, intercept: float) -> str:
    """Label a fitted slope with a known unit conversion when it is close."""
    for factor, label in KNOWN_CONVERSIONS:
        if abs(slope - factor) / abs(factor) < 0.01:
            return label
    if abs(slope - 1.0) < 0.01 and abs(intercept) < 0.01:
        return "possible duplicate feature (same scale)"
    return ""


def unit_duplicate_pairs(df: pd.DataFrame, max_pairs: int = 30) -> pd.DataFrame:
    """Find numeric column pairs that are almost perfectly linearly related.

    Two columns with |r| close to 1 and a non-1 slope are likely the same
    measurement in different units (for example Celsius vs Fahrenheit).
    Such pairs are redundant for modelling.
    """
    num = df.select_dtypes(include=np.number)
    cols = [c for c in num.columns if num[c].notna().sum() >= MIN_CORR_ROWS]
    results = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            pair = num[[a, b]].dropna()
            if len(pair) < MIN_CORR_ROWS:
                continue
            x = pair[a].to_numpy(dtype=float)
            y = pair[b].to_numpy(dtype=float)
            if x.std() == 0 or y.std() == 0:
                continue
            r = float(np.corrcoef(x, y)[0, 1])
            if not np.isfinite(r) or abs(r) < 0.9999:
                continue
            slope = float(np.cov(x, y, ddof=0)[0, 1] / np.var(x, ddof=0))
            intercept = float(y.mean() - slope * x.mean())
            results.append(
                {
                    "column_a": a,
                    "column_b": b,
                    "n": len(pair),
                    "r": round(r, 6),
                    "slope": round(slope, 6),
                    "intercept": round(intercept, 4),
                    "note": _match_conversion(slope, intercept),
                }
            )
    out = pd.DataFrame(results)
    if out.empty:
        return out
    return (
        out.assign(abs_r=out["r"].abs())
        .sort_values(["abs_r", "n"], ascending=False)
        .drop(columns="abs_r")
        .head(max_pairs)
        .reset_index(drop=True)
    )


def plausibility_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Heuristic checks for values that are physically impossible.

    These rules only fire for columns whose names match known weather
    variables, and every rule is reported explicitly so it can be reviewed.
    """
    flags: list[dict[str, object]] = []

    def add(col: str, mask: pd.Series, rule: str) -> None:
        bad = int(mask.sum())
        if bad:
            flags.append(
                {
                    "column": col,
                    "rule": rule,
                    "bad_rows": bad,
                    "examples": ", ".join(map(str, df.loc[mask, col].head(3).tolist())),
                }
            )

    for col in df.select_dtypes(include=np.number).columns:
        name = col.lower()
        s = df[col]

        if "humidity" in name:
            add(col, (s < 0) | (s > 100), "humidity outside 0-100%")
        if "cloud" in name:
            add(col, (s < 0) | (s > 100), "cloud cover outside 0-100%")
        if "precip" in name:
            add(col, s < 0, "negative precipitation")
        if "wind" in name:
            add(col, s < 0, "negative wind value")
        if "visibility" in name:
            add(col, s < 0, "negative visibility")
        if "uv" in name:
            add(col, (s < 0) | (s > 20), "UV index outside 0-20")
        if "pressure" in name or "preassure" in name:
            if "inhg" in name or "inches" in name or name.endswith("_in"):
                add(col, (s < 20) | (s > 35), "pressure outside 20-35 inHg")
            else:
                add(col, (s < 800) | (s > 1100), "pressure outside 800-1100 hPa")
        if "temp" in name:
            if "fahrenheit" in name or name.endswith("_f"):
                add(col, (s < -130) | (s > 140), "temperature outside -130..140 F")
            elif "celsius" in name or name.endswith("_c"):
                add(col, (s < -90) | (s > 60), "temperature outside -90..60 C")
            else:
                add(col, (s < -130) | (s > 140), "temperature outside -130..140 (units unknown)")
        if name.startswith("air_quality"):
            add(col, s < 0, "negative air-quality value")
        if "latitude" in name:
            add(col, (s < -90) | (s > 90), "latitude outside -90..90")
        if "longitude" in name:
            add(col, (s < -180) | (s > 180), "longitude outside -180..180")

    out = pd.DataFrame(flags, columns=["column", "rule", "bad_rows", "examples"])
    return out


# ---------------------------------------------------------------------------
# Full text profile
# ---------------------------------------------------------------------------
def _section(title: str) -> str:
    return "\n" + "=" * 80 + f"\n{title}\n" + "=" * 80


def build_profile(df: pd.DataFrame, csv_path: Path) -> str:
    """Assemble the complete Phase A profile as one string."""
    lines: list[str] = []

    def add(text: object = "") -> None:
        lines.append(str(text))

    size_mb = csv_path.stat().st_size / 1024**2
    add(_section("1. FILE, SHAPE, MEMORY"))
    add(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    add(f"File: {csv_path}")
    add(f"File size: {size_mb:.2f} MB")
    add(f"SHA-256: {sha256_file(csv_path)}")
    add(f"Rows: {len(df):,} | Columns: {df.shape[1]}")
    add(f"Memory (deep): {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    buf = io.StringIO()
    df.info(buf=buf)
    add(buf.getvalue().rstrip())

    add(_section("2. COLUMN OVERVIEW"))
    add(column_overview(df).to_string(index=False))

    add(_section("3. FIRST 5 ROWS (transposed)"))
    add(df.head(5).T.to_string())

    add(_section("4. MISSING VALUES"))
    miss = missing_table(df)
    add(miss.to_string() if not miss.empty else "No missing values found.")
    add(f"Rows with at least one missing value: {int(df.isna().any(axis=1).sum()):,}")

    add(_section("5. DUPLICATES"))
    time_col = pick_time_column(df)
    loc = detect_location_columns(df)
    entity_cols: list[str] = []
    if loc["country"]:
        entity_cols.append(loc["country"][0])
    if loc["city"]:
        entity_cols.append(loc["city"][0])
    add(f"Chosen time column: {time_col!r}")
    add(f"Chosen entity columns: {entity_cols!r}")
    if loc["city"] and loc["country"]:
        groups = df.groupby([loc["country"][0], loc["city"][0]], dropna=False).ngroups
        add(f"Unique country+city groups: {groups:,}")
        add(f"Unique city names: {df[loc['city'][0]].nunique():,}")
    add(
        "Note: many rows sharing one timestamp is expected when each city has "
        "one reading; the entity+timestamp key is the important duplicate check."
    )
    dup = duplicate_summary(df, time_col=time_col, entity_cols=entity_cols)
    for key, value in dup.items():
        add(f"{key}: {value:,}")
    if dup.get("exact_duplicate_rows", 0):
        add("\nExample duplicated rows (first 5):")
        add(df[df.duplicated(keep=False)].head(5).to_string())

    add(_section("6. TIME COLUMN CANDIDATES"))
    tc = time_coverage(df)
    add(tc.to_string(index=False) if not tc.empty else "No time-like columns found.")

    add(_section("7. TIME COVERAGE DETAILS"))
    if time_col is None:
        add("No usable time column (>=95% parseable).")
    else:
        add(f"Time column: {time_col}")
        add("\n".join(coverage_details(df, time_col)))

    add(_section("8. LOCATION COLUMN CANDIDATES"))
    ls = location_summary(df)
    add(ls.to_string(index=False) if not ls.empty else "No location-like columns found.")

    add(_section("9. NUMERIC SUMMARY"))
    ns = numeric_summary(df)
    add(ns.to_string() if not ns.empty else "No numeric columns found.")

    add(_section("10. CONSTANT COLUMNS"))
    const = constant_columns(df)
    if const:
        for col, value in const:
            add(f"{col}: {value!r}")
    else:
        add("None.")

    add(_section("11. UNIT-DUPLICATE / NEAR-DUPLICATE NUMERIC PAIRS"))
    pairs = unit_duplicate_pairs(df)
    add(
        pairs.to_string(index=False)
        if not pairs.empty
        else "No near-perfect linear pairs found (|r| >= 0.9999)."
    )

    add(_section("12. PLAUSIBILITY FLAGS (heuristic)"))
    flags = plausibility_flags(df)
    add(
        flags.to_string(index=False)
        if not flags.empty
        else "No heuristic violations found. This does not replace manual review."
    )

    add(_section("END OF PROFILE"))
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    try:
        # A CLI path wins; otherwise reuse the local CSV or download it once.
        csv_path = Path(argv[1]) if len(argv) > 1 else ensure_raw_csv()
        df = load_raw(csv_path)
    except (FileNotFoundError, RuntimeError) as exc:
        print(exc)
        return 1

    report = build_profile(df, csv_path)
    print(report)

    DEFAULT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_REPORT.write_text(report, encoding="utf-8")
    print(f"\nProfile saved to: {DEFAULT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
