"""Convert synthetic_glycemia_data.csv into the format bensparx_nested_cv.py expects:
   columns `class` (0/1) and `id`, everything else numeric."""
import sys
import pandas as pd

src = sys.argv[1] if len(sys.argv) > 1 else "synthetic_glycemia_data.csv"
dst = sys.argv[2] if len(sys.argv) > 2 else "BenSParX_synthetic.csv"

df = pd.read_csv(src)
out = pd.DataFrame()
out["id"] = df["participant_id"]
out["class"] = (df["hba1c_pct"] >= 5.7).astype(int)          # 1 = dysglycemia (HbA1c >= 5.7)

# numeric features kept as-is
keep = ["age", "weekday_sleep_hrs", "weekend_sleep_hrs", "avg_sleep_hrs",
        "weekday_midpoint_h", "weekend_midpoint_h", "midpoint_shift_hrs",
        "social_jetlag_hrs", "blue_light_exposure_min", "carb_sequencing_index",
        "fasting_window_hrs", "perceived_stress_score", "bmi",
        "outdoor_activity_hrs", "sleep_disruption_index", "cortisol_ugdl",
        "cortisol_spike", "insulin_resistance_score"]
for c in keep:
    out[c] = df[c]

# text -> numbers
out["gender_male"] = (df["gender"] == "Male").astype(int)
out["gender_female"] = (df["gender"] == "Female").astype(int)
out["gender_other"] = (df["gender"] == "Other").astype(int)
out["sleep_window_late"] = (df["sleep_timing_window"] == "4AM-12PM").astype(int)
out["sleep_duration_cat"] = df["sleep_duration_category"].map({"<8": 0, "8": 1, ">8": 2})

# NOT included on purpose (would leak the label): hba1c_pct, glycemic_status.
# HH:MM time strings are dropped (their information is in the midpoint columns).
assert out.drop(columns=["id"]).select_dtypes(exclude="number").empty
out.to_csv(dst, index=False)
print(f"wrote {dst}: {out.shape[0]} rows, {out.shape[1]} cols, class balance {out['class'].mean():.2f}")
