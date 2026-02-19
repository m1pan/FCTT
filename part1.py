"""
PART 1 — Thévenin Battery Model (OCV(SOC) + R) with Coulomb Counting SOC

Topline overview (what this script does):
1) Load and plot the RAW SOC-OCV map (from SOC_OCV.csv / .tsv export).
2) Load and plot the RAW Battery_Testing_Data (time, current, voltage).
3) Build interpolation functions:
      - OCV_from_SOC(z)
      - SOC_from_OCV(v)
4) Determine initial SOC (z0) from the last rest voltage before the current profile starts.
5) Simulate a simple Thévenin model:
      V_pred(t) = OCV(z(t)) - I(t)*R
      z(t) updated by coulomb counting.
6) Compare TWO candidate resistances:
      - R from datasheet DCIR (e.g., ~0.022 Ω)
      - R from your curve-based estimate (you will edit/replace)
   and compute error metrics (RMSE/MAE/max|error|) to decide which fits better.
7) Produce report-ready plots:
      - Measured vs predicted voltage (both R values)
      - Error vs time (both R values)
      - Zoomed region + current (optional)

You only need to edit:
- file paths (SOC_OCV and Battery_Testing_Data)
- Q_Ah (capacity)
- R_DCIROhm (datasheet DCIR)
- R_curveOhm (your curve-based Ohm’s law estimate from discharge curves at same SOC)

Notes:
- Your Battery_Testing_Data appears to use +Current = CHARGE (we saw V rise when I became +1A).
  This script flips sign so that in the model: +I = DISCHARGE (standard convention).
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# 1) Data structure for OCV map
# -----------------------------
@dataclass(frozen=True)
class OCVMap:
    soc: np.ndarray  # SOC in 0..1, strictly increasing
    ocv: np.ndarray  # volts, aligned with soc


# -------------------------------------------------------
# 2) Load SOC-OCV file + prepare for interpolation/invert
# -------------------------------------------------------
def load_soc_ocv_map(path: str, sep: str = "\t") -> OCVMap:
    """
    Loads SOC_OCV data from a 2-column file:
      col0: SOC (often % and descending from ~99.999999)
      col1: ECell/V (volts)

    Why we do each step:
    - read file: get raw SOC and OCV points
    - drop NaNs: avoid interpolation errors
    - normalise SOC to 0..1: convenient and consistent with SOC math
    - sort ascending SOC: np.interp requires increasing x
    - remove duplicate SOC values: prevents weird interpolation edge cases
    """
    df = pd.read_csv(path, sep=sep)

    soc_raw = df.iloc[:, 0].to_numpy(dtype=float)
    ocv_raw = df.iloc[:, 1].to_numpy(dtype=float)

    mask = np.isfinite(soc_raw) & np.isfinite(ocv_raw)
    soc_raw, ocv_raw = soc_raw[mask], ocv_raw[mask]

    # Convert SOC from percent to fraction if needed
    soc = soc_raw.copy()
    if soc.max() > 1.5:
        soc = soc / 100.0

    # Sort SOC ascending for interpolation
    idx = np.argsort(soc)
    soc = soc[idx]
    ocv = ocv_raw[idx]

    # Remove duplicate SOC points
    soc_unique, unique_idx = np.unique(soc, return_index=True)
    ocv_unique = ocv[unique_idx]

    return OCVMap(soc=soc_unique, ocv=ocv_unique)


def plot_raw_soc_ocv(path: str, sep: str = "\t"):
    """
    Plot the SOC-OCV map exactly as stored in the file (RAW), so you can include it in your report.
    """
    df = pd.read_csv(path, sep=sep)
    x = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
    y = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]

    plt.figure()
    plt.plot(x, y)
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    plt.title("RAW SOC-OCV data (as stored in file)")
    plt.grid(True)
    plt.show()


def ocv_from_soc(z: float, ocv_map: OCVMap) -> float:
    """
    OCV(z): returns open-circuit voltage at SOC=z using linear interpolation.
    Why interpolation: keeps the real nonlinear OCV curve shape, avoids forcing a linear fit.
    """
    z = float(np.clip(z, 0.0, 1.0))
    return float(np.interp(z, ocv_map.soc, ocv_map.ocv))


def soc_from_ocv(v: float, ocv_map: OCVMap) -> float:
    """
    SOC(v): invert OCV map to estimate SOC from a voltage v.
    Why needed: to estimate initial SOC from the measured rest voltage before the drive cycle starts.

    We sort by OCV and interpolate SOC in that sorted space.
    Works well if OCV is mostly monotonic with SOC (your curve looks monotonic).
    """
    idx = np.argsort(ocv_map.ocv)
    ocv_sorted = ocv_map.ocv[idx]
    soc_sorted = ocv_map.soc[idx]
    v = float(np.clip(v, ocv_sorted[0], ocv_sorted[-1]))
    return float(np.interp(v, ocv_sorted, soc_sorted))


# -----------------------------------------
# 3) Load Battery_Testing_Data (raw + clean)
# -----------------------------------------
def load_battery_testing_data(path: str, positive_current_is_charge: bool = True):
    """
    Loads Battery_Testing_Data CSV with columns:
      Time (s), Current (mA), Voltage (V), Temperature

    Why steps:
    - drop empty rows (like ",,,") which break numeric conversion
    - convert to numeric (coerce) to safely handle occasional odd formatting
    - convert mA->A for physics equations
    - optionally flip sign so model uses +I = discharge
    """
    df = pd.read_csv(path)
    df = df.dropna(how="all")

    # Coerce numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep rows with at least time/current/voltage
    df = df.dropna(subset=[df.columns[0], df.columns[1], df.columns[2]])

    t = df.iloc[:, 0].to_numpy(dtype=float)
    I_mA = df.iloc[:, 1].to_numpy(dtype=float)
    V = df.iloc[:, 2].to_numpy(dtype=float)
    T = df.iloc[:, 3].to_numpy(dtype=float) if df.shape[1] > 3 else None

    I = I_mA / 1000.0  # mA -> A

    # Convert to model convention (+ = discharge)
    if positive_current_is_charge:
        I = -I

    return t, I, V, T


def plot_raw_battery_data(t, I, V, T=None):
    """
    Plot RAW battery test signals (for report / sanity checks):
    - Voltage vs time
    - Current vs time
    - Temperature vs time (if available)
    """
    plt.figure()
    plt.plot(t, V)
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.title("RAW Battery_Testing_Data: Voltage vs time")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(t, I)
    plt.xlabel("Time (s)")
    plt.ylabel("Current (A)  (+ = discharge)")
    plt.title("RAW Battery_Testing_Data: Current vs time")
    plt.grid(True)
    plt.show()

    if T is not None:
        plt.figure()
        plt.plot(t, T)
        plt.xlabel("Time (s)")
        plt.ylabel("Temperature (°C)")
        plt.title("RAW Battery_Testing_Data: Temperature vs time")
        plt.grid(True)
        plt.show()


# ----------------------------------------------------
# 4) Find initial SOC from last rest point before start
# ----------------------------------------------------
def find_profile_start(I: np.ndarray, threshold_A: float = 0.02, min_consecutive: int = 3) -> int:
    """
    Find when the profile starts: first time |I| stays above threshold for a few samples.

    Why: initial SOC should be inferred at rest (I≈0), right before the profile begins,
    because V≈OCV there.
    """
    above = np.abs(I) > threshold_A
    count = 0
    for k, flag in enumerate(above):
        count = count + 1 if flag else 0
        if count >= min_consecutive:
            return k - min_consecutive + 1
    return 0


def estimate_initial_soc(t, I, V, ocv_map: OCVMap, R_ohm: float = 0.0):
    """
    Estimate initial SOC z0 from last rest sample before profile starts.

    If I at that point is ~0:
        OCV(z0) ≈ V_rest
    If not exactly 0:
        OCV(z0) ≈ V_rest + I_rest*R

    Returns z0, k0 (start index), k_init (rest index)
    """
    k0 = find_profile_start(I)
    k_init = max(k0 - 1, 0)
    ocv_target = V[k_init] + I[k_init] * R_ohm
    z0 = soc_from_ocv(ocv_target, ocv_map)
    z0 = float(np.clip(z0, 0.0, 1.0))
    return z0, k0, k_init


# ------------------------------
# 5) Thévenin model simulation
# ------------------------------
def simulate_thevenin(t, I, ocv_map: OCVMap, z0: float, Q_Ah: float, R_ohm: float):
    """
    Simulate a simple Thévenin model:
      z[k+1] = z[k] - I[k]*dt / (Q_Ah*3600)
      V_pred[k] = OCV(z[k]) - I[k]*R

    Assumes +I = discharge.

    Returns: SOC array z, predicted voltage array V_pred
    """
    Q_C = Q_Ah * 3600.0
    n = len(t)
    z = np.zeros(n, dtype=float)
    V_pred = np.zeros(n, dtype=float)

    z[0] = float(np.clip(z0, 0.0, 1.0))

    for k in range(n - 1):
        dt = t[k + 1] - t[k]
        V_pred[k] = ocv_from_soc(z[k], ocv_map) - I[k] * R_ohm
        z[k + 1] = z[k] - (I[k] * dt) / Q_C
        z[k + 1] = float(np.clip(z[k + 1], 0.0, 1.0))

    V_pred[-1] = ocv_from_soc(z[-1], ocv_map) - I[-1] * R_ohm
    return z, V_pred


# ------------------------------
# 6) Error analysis utilities
# ------------------------------
def error_metrics(V_meas, V_pred):
    """
    Compute standard error metrics for model comparison.
    """
    e = V_meas - V_pred
    rmse = float(np.sqrt(np.mean(e ** 2)))
    mae = float(np.mean(np.abs(e)))
    max_abs = float(np.max(np.abs(e)))
    return rmse, mae, max_abs


def plot_comparison(t, V_meas, V_pred_a, V_pred_b, label_a="Model A", label_b="Model B"):
    """
    Plot measured voltage vs TWO model predictions on the same axes.
    Useful to compare two R values (e.g., DCIR-based vs curve-based).
    """
    plt.figure()
    plt.plot(t, V_meas, label="Measured")
    plt.plot(t, V_pred_a, label=label_a)
    plt.plot(t, V_pred_b, label=label_b)
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.title("Measured vs Predicted Voltage (two R candidates)")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_errors(t, V_meas, V_pred_a, V_pred_b, label_a="Model A", label_b="Model B"):
    """
    Plot error traces for two model predictions.
    """
    e_a = V_meas - V_pred_a
    e_b = V_meas - V_pred_b

    plt.figure()
    plt.plot(t, e_a, label=f"Error: {label_a}")
    plt.plot(t, e_b, label=f"Error: {label_b}")
    plt.xlabel("Time (s)")
    plt.ylabel("Error (V)  (V_meas - V_pred)")
    plt.title("Voltage Prediction Error (two R candidates)")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_zoom_window(t, I, V_meas, V_pred, t1, t2, title_prefix=""):
    """
    Zoomed window plots (voltage + current) to explain spikes/transients in the report.
    """
    mask = (t >= t1) & (t <= t2)

    plt.figure()
    plt.plot(t[mask], V_meas[mask], label="Measured")
    plt.plot(t[mask], V_pred[mask], label="Thevenin model")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.title(f"{title_prefix}Voltage zoom: {t1}-{t2} s")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(t[mask], I[mask])
    plt.xlabel("Time (s)")
    plt.ylabel("Current (A) (+ = discharge)")
    plt.title(f"{title_prefix}Current zoom: {t1}-{t2} s")
    plt.grid(True)
    plt.show()


# ------------------------------
# 7) MAIN — run Part 1 end-to-end
# ------------------------------
def main():
    # -----------------------------
    # File paths (edit as needed)
    # -----------------------------
    # Your SOC-OCV file (tab-separated in your case)
    SOC_OCV_PATH = "Project 3 data/SOC_OCV.csv"
    SOC_OCV_SEP = "\t"                           # Keep "\t" if your SOC_OCV uses tabs
    # Your Battery_Testing_Data file (comma-separated)
    BATTERY_TEST_PATH = "Project 3 data/Battery_Testing_Data.csv"

    # ------------------------------------------
    # Example model parameters (edit as needed)
    # ------------------------------------------
    # Capacity from datasheet tables (Samsung INR18650-25R ~2.5 Ah)
    Q_Ah = 2.5

    # Candidate resistance 1: datasheet DCIR mean (22.15 mΩ -> 0.02215 Ω)
    R_DCIROhm = 0.02215

    # Candidate resistance 2: curve-based estimate (EDIT THIS once you read from discharge curves)
    # Example placeholder: using avg-V slope gave ~0.019 Ω
    V1 = 3.35
    V2 = 3.55
    V3 = 3.6
    I3 = 2.5
    I2 = 5
    I1 = 15

    R_curveOhm = (V2 - V3) / (I3 - I2)  # Ohm's law: R = ΔV/ΔI

    # Your dataset: +Current in file corresponds to CHARGE, so set True to flip sign
    POSITIVE_CURRENT_IS_CHARGE = True

    # Optional: Evaluate discharge-only to avoid charge/regen segments dominating error
    EVALUATE_DISCHARGE_ONLY = True  # set True if you want metrics on I>0 only

    # -----------------------------
    # A) Plot RAW inputs for report
    # -----------------------------
    plot_raw_soc_ocv(SOC_OCV_PATH, sep=SOC_OCV_SEP)

    ocv_map = load_soc_ocv_map(SOC_OCV_PATH, sep=SOC_OCV_SEP)

    # Battery test data
    t, I, V_meas, T = load_battery_testing_data(BATTERY_TEST_PATH,
                                                positive_current_is_charge=POSITIVE_CURRENT_IS_CHARGE)

    plot_raw_battery_data(t, I, V_meas, T)

    print("Current min/max after sign convention (+ = discharge):",
          float(I.min()), float(I.max()))

    # --------------------------------------------
    # B) Estimate initial SOC from rest voltage
    # --------------------------------------------
    # For initial SOC, using R≈0 is fine if the rest current is ~0.
    z0, k0, k_init = estimate_initial_soc(t, I, V_meas, ocv_map, R_ohm=0.0)

    print(f"Profile start index k0={k0}, time={t[k0]:.1f}s")
    print(f"Init (rest) index k_init={k_init}, time={t[k_init]:.1f}s")
    print(
        f"Rest point: V={V_meas[k_init]:.4f} V, I={I[k_init]:.4f} A -> z0={z0:.4f} ({z0*100:.2f}%)")

    # --------------------------------------------
    # C) Simulate model with two R candidates
    # --------------------------------------------
    z_a, V_pred_a = simulate_thevenin(t, I, ocv_map, z0, Q_Ah, R_DCIROhm)
    z_b, V_pred_b = simulate_thevenin(t, I, ocv_map, z0, Q_Ah, R_curveOhm)

    print("SOC range (DCIR R):", float(z_a.min()), float(z_a.max()))
    print("SOC range (Curve R):", float(z_b.min()), float(z_b.max()))

    # --------------------------------------------
    # D) Error analysis (overall or discharge-only)
    # --------------------------------------------
    if EVALUATE_DISCHARGE_ONLY:
        mask = I > 0
        t_eval = t[mask]
        V_eval = V_meas[mask]
        Va = V_pred_a[mask]
        Vb = V_pred_b[mask]
        print("Evaluating DISCHARGE only (I>0). Points:", int(mask.sum()))
    else:
        t_eval = t
        V_eval = V_meas
        Va = V_pred_a
        Vb = V_pred_b
        print("Evaluating ALL points (includes charge if present).")

    rmse_a, mae_a, max_a = error_metrics(V_eval, Va)
    rmse_b, mae_b, max_b = error_metrics(V_eval, Vb)

    print(
        f"\nModel A (DCIR)  R={R_DCIROhm:.5f} Ω: RMSE={rmse_a:.4f} V, MAE={mae_a:.4f} V, max|e|={max_a:.4f} V")
    print(
        f"Model B (Curve) R={R_curveOhm:.5f} Ω: RMSE={rmse_b:.4f} V, MAE={mae_b:.4f} V, max|e|={max_b:.4f} V")

    better = "DCIR" if rmse_a < rmse_b else "Curve-based"
    print(f"\nLower RMSE: {better} resistance estimate.")

    # --------------------------------------------
    # E) Report plots: measured vs predicted + errors
    # --------------------------------------------
    plot_comparison(t_eval, V_eval, Va, Vb,
                    label_a=f"Thevenin (R={R_DCIROhm:.4f}Ω)",
                    label_b=f"Thevenin (R={R_curveOhm:.4f}Ω)")

    plot_errors(t_eval, V_eval, Va, Vb,
                label_a=f"R={R_DCIROhm:.4f}Ω",
                label_b=f"R={R_curveOhm:.4f}Ω")

    # --------------------------------------------
    # F) Optional zoom window for explaining spikes
    # --------------------------------------------
    # Edit these times based on where you see large errors/spikes
    # plot_zoom_window(t, I, V_meas, V_pred_a, 18000, 23500, title_prefix="(DCIR) ")
    # plot_zoom_window(t, I, V_meas, V_pred_b, 18000, 23500, title_prefix="(Curve) ")


if __name__ == "__main__":
    main()
