"""
Synthetic behavioural dataset: sleep timing, social jetlag and glycaemic markers
(Meeting RG6 notes).

Idea: every row is built in *layers*. Behaviour is drawn first with uniform /
randint style random numbers (as the notes ask), then every physiological value
is computed from those behaviours with explicit formulas + noise. That keeps the
mathematical relationships intact ("conditional synthetic coding") so a model
trained on the file has something real to learn.

Run:  python generate_synthetic_data.py --rows 2000 --seed 42
"""
import argparse
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def hhmm(hours):
    """Decimal clock hours (may exceed 24) -> 'HH:MM' string."""
    hours = np.asarray(hours) % 24
    h = np.floor(hours).astype(int)
    m = np.round((hours - h) * 60).astype(int)
    h = np.where(m == 60, (h + 1) % 24, h)
    m = np.where(m == 60, 0, m)
    return np.array([f"{a:02d}:{b:02d}" for a, b in zip(h, m)])


# ----------------------------------------------------------------------------
# generator
# ----------------------------------------------------------------------------
def generate(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    u = rng.uniform           # "random.uniform"
    ri = rng.integers         # "random.randint"

    # ---- demographics ------------------------------------------------------
    age = ri(18, 26, n)                                   # 18-25 inclusive
    gender = rng.choice(["Male", "Female", "Other"], n, p=[0.49, 0.49, 0.02])
    is_f = (gender == "Female").astype(float)
    is_m = (gender == "Male").astype(float)

    # ---- sleep: weekdays (free days = weekend) ------------------------------
    onset_wd = u(22.0, 26.0, n)            # 10 PM .. 2 AM (26 = 2 AM next day)
    dur_wd = u(5.0, 9.5, n)                # hours slept on weekdays
    wake_wd = onset_wd + dur_wd
    mid_wd = onset_wd + dur_wd / 2

    # weekend: ~15% keep a steady schedule, the rest shift later + sleep in
    steady = rng.random(n) < 0.15
    delay = np.where(steady, u(0.0, 0.5, n), u(0.5, 4.0, n))
    sleep_debt_wd = np.clip(8 - dur_wd, 0, None)
    extra = np.clip(0.5 * sleep_debt_wd + u(-0.5, 1.0, n), -0.5, 3.0)
    onset_we = onset_wd + delay
    dur_we = np.clip(dur_wd + extra, 4.0, 12.0)
    wake_we = onset_we + dur_we
    mid_we = onset_we + dur_we / 2

    # ---- social jetlag (the core variable) ----------------------------------
    # midpoint shift is signed; social jetlag is its absolute value (float hours)
    midpoint_shift = mid_we - mid_wd
    sjl = np.abs(midpoint_shift)

    avg_sleep = (5 * dur_wd + 2 * dur_we) / 7
    sleep_debt = np.clip(8 - avg_sleep, 0, None)

    sleep_cat = np.where(avg_sleep < 7.5, "<8", np.where(avg_sleep <= 8.5, "8", ">8"))
    # reference windows from the notes: 10PM-6AM (midpoint 2AM) vs 4AM-12PM (8AM)
    timing_window = np.where(mid_wd < 5.0, "10PM-6AM", "4AM-12PM")

    # ---- behaviours ----------------------------------------------------------
    # blue light: screen minutes inside the 2 h before bed (0-120), later bedtime -> more
    blue = np.clip(u(0, 90, n) + 8.0 * np.clip(onset_wd - 23.0, 0, None) + rng.normal(0, 8, n), 0, 120)
    carb_idx = rng.beta(2.2, 2.2, n)                       # 0 = carbs first, 1 = perfect
    fasting = np.clip(rng.normal(12.5, 2.0, n) - 0.35 * np.clip(onset_wd - 24, 0, None), 7, 20)
    bmi = np.clip(rng.normal(23.3, 3.6, n) + 0.4 * is_m, 16, 40)
    outdoor = np.clip(u(0.2, 4.5, n) - 0.008 * blue + rng.normal(0, 0.3, n), 0, 6)
    stress_raw = u(1, 6, n) + 0.55 * sjl + 0.7 * sleep_debt + rng.normal(0, 0.7, n)
    stress = np.clip(np.round(stress_raw), 1, 10).astype(int)   # integer 1-10

    # ---- disrupted sleep architecture (0-1 composite) ------------------------
    disruption = np.clip(
        0.35 * np.clip(sjl / 4, 0, 1)
        + 0.25 * np.clip(sleep_debt / 3, 0, 1)
        + 0.25 * blue / 120
        + 0.15 * stress / 10
        + rng.normal(0, 0.04, n),
        0, 1)

    # ---- cortisol (morning, ug/dL): sleep disruption + stress + gender --------
    cortisol = np.clip(
        9.5 + 7.0 * disruption + 0.40 * (stress - 5)
        + 0.9 * is_f * disruption * 2          # assumed stronger HPA response in women
        + rng.normal(0, 1.6, n),
        4, 32)
    cortisol_spike = (cortisol >= 18).astype(int)

    # ---- insulin resistance (HOMA-IR style, log-normal) ----------------------
    ln_ir = (np.log(1.3)
             + 0.055 * (bmi - 23)
             + 0.10 * sjl
             + 0.025 * (cortisol - 12)
             - 0.25 * (carb_idx - 0.5)
             - 0.04 * outdoor
             - 0.02 * (fasting - 12.5)
             + 0.10 * is_m
             + rng.normal(0, 0.18, n))
    ir = np.clip(np.exp(ln_ir), 0.3, 12)

    # ---- HbA1c (%) ------------------------------------------------------------
    hba1c = (4.66 + 1.5 * np.log(ir / 1.3)
             + 0.12 * sjl + 0.02 * (cortisol - 12)
             + rng.normal(0, 0.18, n))
    hba1c = np.clip(hba1c, 4.2, 9.0)
    status = np.where(hba1c < 5.7, "Normal",
                      np.where(hba1c < 6.5, "Prediabetes (5.7-6.4)", "Diabetes (>=6.5)"))

    df = pd.DataFrame({
        "participant_id": [f"P{i:05d}" for i in range(1, n + 1)],
        "age": age,
        "gender": gender,
        "weekday_sleep_onset": hhmm(onset_wd),
        "weekday_wake_time": hhmm(wake_wd),
        "weekend_sleep_onset": hhmm(onset_we),
        "weekend_wake_time": hhmm(wake_we),
        "weekday_sleep_hrs": dur_wd.round(2),
        "weekend_sleep_hrs": dur_we.round(2),
        "avg_sleep_hrs": avg_sleep.round(2),
        "sleep_duration_category": sleep_cat,
        "sleep_timing_window": timing_window,
        "weekday_midpoint_h": (mid_wd % 24).round(2),
        "weekend_midpoint_h": (mid_we % 24).round(2),
        "midpoint_shift_hrs": midpoint_shift.round(2),
        "social_jetlag_hrs": sjl.round(2),
        "blue_light_exposure_min": blue.round(0).astype(int),
        "carb_sequencing_index": carb_idx.round(2),
        "fasting_window_hrs": fasting.round(1),
        "perceived_stress_score": stress,
        "bmi": bmi.round(1),
        "outdoor_activity_hrs": outdoor.round(2),
        "sleep_disruption_index": disruption.round(3),
        "cortisol_ugdl": cortisol.round(1),
        "cortisol_spike": cortisol_spike,
        "insulin_resistance_score": ir.round(2),
        "hba1c_pct": hba1c.round(2),
        "glycemic_status": status,
    })
    return df


# ----------------------------------------------------------------------------
# validation: make sure the relationships did not collapse
# ----------------------------------------------------------------------------
def validate(df):
    c = df.select_dtypes("number").corr()
    checks = {
        "social_jetlag_hrs -> hba1c_pct > 0.20": c.loc["social_jetlag_hrs", "hba1c_pct"] > 0.20,
        "social_jetlag_hrs -> cortisol > 0.20": c.loc["social_jetlag_hrs", "cortisol_ugdl"] > 0.20,
        "bmi -> insulin_resistance > 0.20": c.loc["bmi", "insulin_resistance_score"] > 0.20,
        "insulin_resistance -> hba1c > 0.40": c.loc["insulin_resistance_score", "hba1c_pct"] > 0.40,
        "carb_sequencing -> insulin_resistance < 0": c.loc["carb_sequencing_index", "insulin_resistance_score"] < 0,
        "sleep_disruption -> cortisol > 0.40": c.loc["sleep_disruption_index", "cortisol_ugdl"] > 0.40,
        "no column is constant": (df.select_dtypes("number").nunique() > 1).all(),
    }
    for k, v in checks.items():
        print(("PASS  " if v else "FAIL  ") + k)
    assert all(checks.values()), "A relationship collapsed - adjust coefficients."


# ----------------------------------------------------------------------------
# Excel writer (Data + Data Dictionary + Validation with live formulas)
# ----------------------------------------------------------------------------
DICT = [
    ("participant_id", "ID", "Unique synthetic ID"),
    ("age", "Context", "Integer 18-25 (randint). Notes: social jetlag is a primary driver of early dysglycemia in this group"),
    ("gender", "Feature (iii)", "Male / Female / Other. Affects cortisol response (women) and insulin resistance (men) in the formulas"),
    ("weekday_sleep_onset", "Sleep timing", "Clock time falling asleep on work/school days (22:00-02:00)"),
    ("weekday_wake_time", "Sleep timing", "Onset + weekday_sleep_hrs"),
    ("weekend_sleep_onset", "Sleep timing", "Weekday onset + a random delay (steady sleepers ~0-0.5 h, others 0.5-4 h)"),
    ("weekend_wake_time", "Sleep timing", "Weekend onset + weekend_sleep_hrs"),
    ("weekday_sleep_hrs", "Sleep duration", "uniform(5, 9.5)"),
    ("weekend_sleep_hrs", "Sleep duration", "Weekday hours + catch-up sleep proportional to sleep debt"),
    ("avg_sleep_hrs", "Sleep duration", "(5 x weekday + 2 x weekend) / 7"),
    ("sleep_duration_category", "Sleep duration", "'<8' if avg < 7.5, '8' if 7.5-8.5, '>8' if > 8.5 (the notes' 3 groups; +/-0.5 h tolerance is my assumption)"),
    ("sleep_timing_window", "Sleep timing", "'10PM-6AM' or '4AM-12PM': whichever reference window the weekday sleep midpoint is closest to"),
    ("weekday_midpoint_h", "Derived", "Weekday sleep midpoint, clock hours"),
    ("weekend_midpoint_h", "Derived", "Weekend sleep midpoint, clock hours"),
    ("midpoint_shift_hrs", "Feature (i, tracking midpoint shift)", "Signed: weekend midpoint - weekday midpoint. Notes example: 12AM-8AM vs 4AM-12PM -> 4 h"),
    ("social_jetlag_hrs", "Feature (i)", "Float. Absolute value of midpoint_shift_hrs"),
    ("blue_light_exposure_min", "Feature (ii)", "Screen minutes in the 2 h before the primary sleep window (0-120). Higher with later bedtimes"),
    ("carb_sequencing_index", "Feature (iv)", "0-1 (beta distribution). 1 = protein eaten before carbs (perfect), 0 = carbs first"),
    ("fasting_window_hrs", "Feature (v)", "Longest daily gap between meals (7-20 h). Shorter in very late sleepers"),
    ("perceived_stress_score", "Feature (vi)", "Integer 1-10 daily subjective score; rises with social jetlag and sleep debt"),
    ("bmi", "Feature (viii)", "Normal(23.3, 3.6) clipped 16-40, slightly higher in males"),
    ("outdoor_activity_hrs", "Feature (ix)", "Hours/day outdoors (0-6); lower with heavy late-night screen use"),
    ("sleep_disruption_index", "Derived link", "0-1 composite of social jetlag, sleep debt, blue light, stress = 'disrupted sleep architecture' proxy"),
    ("cortisol_ugdl", "Target / link", "Morning cortisol, ug/dL. Driven by sleep disruption, stress, gender"),
    ("cortisol_spike", "Target / link", "1 if cortisol >= 18 ug/dL"),
    ("insulin_resistance_score", "Feature (x)", "HOMA-IR style, log-normal. Up with BMI, social jetlag, cortisol; down with protein-first meals, outdoor time"),
    ("hba1c_pct", "Target", "HbA1c %. Driven by insulin resistance, social jetlag, cortisol + noise"),
    ("glycemic_status", "Target", "Normal <5.7 | Prediabetes 5.7-6.4 | Diabetes >=6.5"),
]


def write_excel(df, path):
    wb = Workbook()
    arial = Font(name="Arial", size=10)
    bold = Font(name="Arial", size=10, bold=True, color="FFFFFF")
    hdr_fill = PatternFill("solid", start_color="1F4E78")

    # --- Data ---
    ws = wb.active
    ws.title = "Data"
    ws.append(list(df.columns))
    for row in df.itertuples(index=False):
        ws.append(list(row))
    for cell in ws[1]:
        cell.font, cell.fill = bold, hdr_fill
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
    for r in ws.iter_rows(min_row=2):
        for cell in r:
            cell.font = arial
    for i, col in enumerate(df.columns, 1):
        ws.column_dimensions[get_column_letter(i)].width = max(12, min(24, len(col) + 3))
    ws.freeze_panes = "B2"
    ws.auto_filter.ref = ws.dimensions

    # --- Data Dictionary ---
    wd = wb.create_sheet("Data Dictionary")
    wd.append(["Column", "Role", "How it is generated / meaning"])
    for r in DICT:
        wd.append(list(r))
    for cell in wd[1]:
        cell.font, cell.fill = bold, hdr_fill
    for r in wd.iter_rows(min_row=2):
        for cell in r:
            cell.font = arial
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    wd.column_dimensions["A"].width = 28
    wd.column_dimensions["B"].width = 32
    wd.column_dimensions["C"].width = 95

    # --- Validation (live formulas pointing at Data) ---
    wv = wb.create_sheet("Validation")
    n = len(df)
    col = {c: get_column_letter(i + 1) for i, c in enumerate(df.columns)}
    rng_ = lambda c: f"Data!${col[c]}$2:${col[c]}${n + 1}"
    wv["A1"] = "Summary statistics and correlations (live formulas on the Data sheet)"
    wv["A1"].font = Font(name="Arial", size=11, bold=True)
    heads = ["Variable", "Mean", "Min", "Max", "Corr with HbA1c", "Corr with cortisol", "Corr with insulin resistance"]
    wv.append([])
    wv.append(heads)
    for cell in wv[3]:
        cell.font, cell.fill = bold, hdr_fill
        cell.alignment = Alignment(wrap_text=True, horizontal="center")
    num_vars = ["age", "weekday_sleep_hrs", "weekend_sleep_hrs", "avg_sleep_hrs", "midpoint_shift_hrs",
                "social_jetlag_hrs", "blue_light_exposure_min", "carb_sequencing_index", "fasting_window_hrs",
                "perceived_stress_score", "bmi", "outdoor_activity_hrs", "sleep_disruption_index",
                "cortisol_ugdl", "insulin_resistance_score", "hba1c_pct"]
    for i, v in enumerate(num_vars):
        r = 4 + i
        wv.cell(r, 1, v)
        wv.cell(r, 2, f"=AVERAGE({rng_(v)})")
        wv.cell(r, 3, f"=MIN({rng_(v)})")
        wv.cell(r, 4, f"=MAX({rng_(v)})")
        wv.cell(r, 5, f"=CORREL({rng_(v)},{rng_('hba1c_pct')})")
        wv.cell(r, 6, f"=CORREL({rng_(v)},{rng_('cortisol_ugdl')})")
        wv.cell(r, 7, f"=CORREL({rng_(v)},{rng_('insulin_resistance_score')})")
        for c in range(1, 8):
            wv.cell(r, c).font = arial
            if c > 1:
                wv.cell(r, c).number_format = "0.00"
    r0 = 4 + len(num_vars) + 2
    wv.cell(r0, 1, "HbA1c group").font = bold
    wv.cell(r0, 2, "Rows").font = bold
    wv.cell(r0, 3, "Share").font = bold
    for c in (1, 2, 3):
        wv.cell(r0, c).fill = hdr_fill
    groups = ["Normal", "Prediabetes (5.7-6.4)", "Diabetes (>=6.5)"]
    for j, g in enumerate(groups):
        r = r0 + 1 + j
        wv.cell(r, 1, g).font = arial
        wv.cell(r, 2, f'=COUNTIF({rng_("glycemic_status")},A{r})').font = arial
        wv.cell(r, 3, f"=B{r}/COUNTA({rng_('participant_id')})").font = arial
        wv.cell(r, 3).number_format = "0.0%"
    wv.cell(r0 + 5, 1, "Note: all coefficients are assumptions chosen to give plausible, learnable effects; "
                       "they are not clinical estimates.").font = Font(name="Arial", size=9, italic=True)
    wv.column_dimensions["A"].width = 30
    for c in "BCDEFG":
        wv.column_dimensions[c].width = 16
    wv.row_dimensions[3].height = 32

    wb.save(path)


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="/mnt/user-data/outputs/synthetic_glycemia_data")
    a = ap.parse_args()

    data = generate(a.rows, a.seed)
    validate(data)
    data.to_csv(a.out + ".csv", index=False)
    write_excel(data, a.out + ".xlsx")
    print(data["glycemic_status"].value_counts(normalize=True).round(3))
    print(f"Saved {len(data)} rows -> {a.out}.csv / .xlsx")
