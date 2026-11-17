"""gta6-twitch-ccu-forecast-01 entry: launch-shape forecaster.

Forecasts hours 2..24 of a game's Twitch launch day, anchored on the hour-1
print, by fitting the release-day shape from the past launches shipped in the
case file (the comparables).

Model (fitted at run time, from the case's own comparables):
  For each usable comparable, take its release-day hourly viewers, fill gaps
  by linear interpolation in log space, and form log-ratios r_h = ln(v_h/v_1)
  for h = 2..24.  A big launch behaves differently from a small one: the
  hour-1 print of a mega launch is already the spike, so its day grows little
  (median r_19 ~ +0.2), while a small category explodes off a quiet hour 1
  (median r_19 ~ +5).  So each hour's log-ratio is regressed on log hour-1
  viewers and on the unlock type (rolling midnight vs simultaneous), with
  three regressors [1, log p1, rolling]:
      r_h = b0_h + b1_h * log p1 + b2_h * rolling + eps_h
  fitted by least squares.  Big targets (p1 >= 20000) are fitted on big
  comparables only; small targets on all of them (log p1 capped at 11.5).
  The forecast quantiles are the fitted value plus a blend of the empirical
  residual quantiles and a Gaussian spread from the residual spread:
      q_tau(h) = p1 * exp(pred_h + 0.5*Q_tau(resid_h) + 0.5*z_tau*sd_h)
  If fitting fails (too few comparables, singular design), the model falls
  back to the empirical ratio quantiles, then to persistence.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only: no network, no pip packages.
Outside the grader, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in for
/input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports. This entry is a single file, but the line
# keeps the option open and matches the starter's layout.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
# The spread of an hour's move for the persistence fallback, in natural-log
# units per square-root hour.
SIGMA = 0.15
# A target at or above this hour-1 print is fitted on big comparables only.
BIG_P1 = 20000.0
# Cap on log hour-1 print when fitting on all comparables (small targets).
LOGP1_CAP = 11.5
# Minimum usable comparables for a regression fit.
MIN_TRAIN = 8
# Minimum observed release-day hours for a comparable to be usable.
MIN_DAY_HOURS = 12


def parse_ts(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100 rounds it.

    A whole number comes back as an int, so json.dumps writes 1240 where JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def quantile(sorted_vals: list[float], tau: float) -> float:
    """The tau-quantile of a sorted list, linearly interpolated."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def solve_linear(matrix: list[list[float]], vec: list[float]) -> list[float]:
    """Solve a small dense linear system by Gaussian elimination with partial pivoting.

    Raises ValueError when a pivot is (near) zero, i.e. the system is singular.
    """
    n = len(vec)
    rows = [row[:] + [vec[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(rows[r][col]))
        if abs(rows[piv][col]) < 1e-12:
            raise ValueError("singular matrix")
        rows[col], rows[piv] = rows[piv], rows[col]
        for r in range(col + 1, n):
            factor = rows[r][col] / rows[col][col]
            for c in range(col, n + 1):
                rows[r][c] -= factor * rows[col][c]
    out = [0.0] * n
    for i in range(n - 1, -1, -1):
        out[i] = (rows[i][n] - sum(rows[i][j] * out[j] for j in range(i + 1, n))) / rows[i][i]
    return out


def least_squares(design: list[list[float]], target: list[float]) -> list[float]:
    """Least-squares coefficients for design rows and target values."""
    n = len(design[0])
    normal = [[sum(row[i] * row[j] for row in design) for j in range(n)] for i in range(n)]
    rhs = [sum(row[i] * y for row, y in zip(design, target)) for i in range(n)]
    return solve_linear(normal, rhs)


def rolling_flag(text) -> float:
    """1.0 when the unlock text describes a rolling midnight launch, else 0.0."""
    lowered = (text or "").lower()
    if "rolling" in lowered or "midnight local" in lowered:
        return 1.0
    return 0.0


def day_log_curve(comp: dict) -> dict[int, float] | None:
    """A comparable's release-day log viewers for hours 1..24, gaps interpolated.

    Returns None when the release day has no real hour-1 point or fewer than
    MIN_DAY_HOURS observed hours (the target game's own pre-release history
    has no release day yet, so it is excluded this way).
    """
    day_start = parse_ts(comp["day_start"])
    observed: dict[int, float] = {}
    for point in comp.get("hourly", []):
        when = point.get("from") or point.get("at")
        if when is None:
            continue
        hour = int((parse_ts(when) - day_start).total_seconds() // 3600) + 1
        if 1 <= hour <= 24:
            viewers = point.get("viewers")
            if viewers is not None:
                observed[hour] = math.log(max(1.0, float(viewers)))
    if 1 not in observed or len(observed) < MIN_DAY_HOURS:
        return None
    known_hours = sorted(observed)
    filled: dict[int, float] = {}
    for hour in range(1, 25):
        if hour in observed:
            filled[hour] = observed[hour]
            continue
        below = [h for h in known_hours if h < hour]
        above = [h for h in known_hours if h > hour]
        if below and above:
            a, b = max(below), min(above)
            filled[hour] = observed[a] + (observed[b] - observed[a]) * (hour - a) / (b - a)
        elif below:
            filled[hour] = observed[max(below)]
        else:
            filled[hour] = observed[min(above)]
    return filled


def fit_shape(train: list[tuple[float, float, dict[int, float]]], logp1: float, roll: float) -> dict[int, list[float]] | None:
    """Per-hour log-ratio offsets from a regression on the comparables.

    train holds (log hour-1 print, rolling flag, day log curve) per comparable.
    Returns {hour: [5 offsets]} or None when the fit cannot be made.
    """
    big = [row for row in train if math.exp(row[0]) >= BIG_P1]
    if math.exp(logp1) >= BIG_P1 and len(big) >= MIN_TRAIN:
        sel = big
        cap = None
    else:
        sel = train
        cap = LOGP1_CAP
    if len(sel) < MIN_TRAIN:
        return None
    xs = [row[0] for row in sel]
    low, high = min(xs), max(xs)
    x = logp1
    if cap is not None:
        x = min(x, cap)
    x = min(max(x, low), high)
    shape: dict[int, list[float]] = {}
    for hour in range(2, 25):
        ratios = [curve[hour] - curve[1] for _, _, curve in sel]
        try:
            design = [[1.0, row[0], row[1]] for row in sel]
            beta = least_squares(design, ratios)
            pred = beta[0] + beta[1] * x + beta[2] * roll
            resid = sorted(
                y - (beta[0] + beta[1] * row[0] + beta[2] * row[1])
                for y, row in zip(ratios, sel)
            )
        except ValueError:
            try:
                design = [[1.0, row[0]] for row in sel]
                beta = least_squares(design, ratios)
                pred = beta[0] + beta[1] * x
                resid = sorted(y - (beta[0] + beta[1] * row[0]) for y, row in zip(ratios, sel))
            except ValueError:
                return None
        mean = sum(resid) / len(resid)
        var = sum((r - mean) ** 2 for r in resid) / len(resid)
        sd = max(math.sqrt(var), 0.10)
        shape[hour] = [
            pred + 0.5 * quantile(resid, tau) + 0.5 * z * sd
            for tau, z in zip(LEVELS, Z)
        ]
    return shape


def empirical_shape(train: list[tuple[float, float, dict[int, float]]]) -> dict[int, list[float]] | None:
    """Per-hour log-ratio offsets from the comparables' raw ratio quantiles."""
    if not train:
        return None
    shape: dict[int, list[float]] = {}
    for hour in range(2, 25):
        vals = sorted(curve[hour] - curve[1] for _, _, curve in train)
        shape[hour] = [quantile(vals, tau) for tau in LEVELS]
    return shape


def persistence_shape() -> dict[int, list[float]]:
    """The starter's persistence offsets: a flat median with widening spread."""
    return {hour: [z * SIGMA * math.sqrt(hour - 1) for z in Z] for hour in range(2, 25)}


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    logp1 = math.log(p1)
    roll = rolling_flag(case.get("notes", ""))
    train: list[tuple[float, float, dict[int, float]]] = []
    for comp in case.get("comparables", []):
        curve = day_log_curve(comp)
        if curve is None:
            continue
        train.append((curve[1], rolling_flag(comp.get("unlock", "")), curve))
    shape = fit_shape(train, logp1, roll)
    if shape is None:
        shape = empirical_shape(train)
    if shape is None:
        shape = persistence_shape()
    hours = []
    for row in case["forecast"]:
        hour = row["hour"]
        quantiles = []
        prev = 0.0
        for offset in shape[hour]:
            value = round2(p1 * math.exp(offset))
            value = max(value, 0.01)
            value = min(value, 1e9)
            if value < prev:
                value = prev
            quantiles.append(value)
            prev = value
        hours.append({"hour": hour, "q": quantiles})
    return {"hours": hours}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
