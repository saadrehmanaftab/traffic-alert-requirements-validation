"""Requirements based validation of a simple tau based traffic alerting function.

Generic study with ASSUMED inputs. It is not a model of any certified collision avoidance system.
See README.md for requirements, assumptions and limits.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)

# ---------------- assumed requirements ----------------
REQ_LEAD = 25.0        # s, R1: alert at least this long before closest approach for threats
THREAT_MISS = 150.0    # m, an encounter is a threat if the true miss distance is below this
SAFE_MISS = 600.0      # m, an encounter is non threatening if the true miss distance is above this
MAX_NUISANCE = 0.05    # R2: at most 5% of non threatening encounters may alert
CREW_DELAY = 5.0       # s, assumed crew recognition and response time (human factors)
MIN_MARGIN = 15.0      # s, R3: time left after crew response must be at least this

# ---------------- assumed encounter and sensor model ----------------
N = 4000
DT = 1.0               # s, surveillance update interval
RANGE_SIGMA = 30.0     # m, range measurement noise
T_CPA = 90.0           # s, time of closest approach after start
WINDOW = 10            # tracker window in updates (least squares velocity estimate)


def encounters(n):
    speed = rng.uniform(80.0, 250.0, n)            # m/s closing speed
    miss = rng.uniform(0.0, 1500.0, n)             # m, true miss distance at closest approach
    return speed, miss


def first_alert_time(speed, miss, tau_th, hmd, window=None):
    """Surveillance gives the intruder's relative position (x along track, y cross track) with noise."""
    t = np.arange(0.0, T_CPA + 30.0, DT)
    x = speed[:, None] * (T_CPA - t[None, :])
    y = np.repeat(miss[:, None], t.size, axis=1)
    xm = x + rng.normal(0.0, RANGE_SIGMA, x.shape)
    ym = y + rng.normal(0.0, RANGE_SIGMA, y.shape)
    # velocity estimate: least squares slope over the last 5 updates (simple tracker)
    w = window or WINDOW
    tt = np.arange(w) - (w - 1) / 2.0
    def slope(a):
        out = np.full(a.shape, np.nan)
        for k in range(w - 1, a.shape[1]):
            out[:, k] = (a[:, k - w + 1:k + 1] * tt).sum(axis=1) / (tt**2).sum() / DT
        return out
    vx, vy = slope(xm), slope(ym)
    r = np.hypot(xm, ym)
    v2 = vx**2 + vy**2
    rv = xm * vx + ym * vy                      # negative when closing
    tau = np.where(rv < 0, -rv / np.maximum(v2, 1e-6), np.inf)   # time to closest approach
    pred_miss = np.abs(xm * vy - ym * vx) / np.sqrt(np.maximum(v2, 1e-6))
    alert = (tau < tau_th) & (pred_miss < hmd)
    alert = np.nan_to_num(alert, nan=0).astype(bool)
    first = np.where(alert.any(axis=1), alert.argmax(axis=1), -1)
    lead = np.where(first >= 0, T_CPA - t[np.clip(first, 0, None)], -np.inf)
    return lead


def evaluate(tau_th, hmd, window=WINDOW):
    speed, miss = encounters(N)
    lead = first_alert_time(speed, miss, tau_th, hmd, window)
    thr = miss < THREAT_MISS
    safe = miss > SAFE_MISS
    r1 = np.mean(lead[thr] >= REQ_LEAD)
    nuisance = np.mean(np.isfinite(lead[safe]) & (lead[safe] > 0))
    margin = np.percentile(lead[thr], 5) - CREW_DELAY
    return dict(tracker_window=window, tau_threshold_s=tau_th, hmd_threshold_m=hmd, threat_alerted_in_time_pct=100 * r1,
                nuisance_alert_pct=100 * nuisance, p5_margin_after_crew_s=margin,
                R1_pass=r1 >= 0.99, R2_pass=nuisance <= MAX_NUISANCE, R3_pass=margin >= MIN_MARGIN)


if __name__ == "__main__":
    rows = [evaluate(tau, hmd, w) for w in (5, 10) for tau in (20, 25, 30, 35, 40, 45) for hmd in (150, 225, 300)]
    df = pd.DataFrame(rows)
    df["all_pass"] = df.R1_pass & df.R2_pass & df.R3_pass
    df.to_csv("results.csv", index=False)
    print(df.round(2).to_string(index=False))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for hmd, g in df[df.tracker_window == WINDOW].groupby("hmd_threshold_m"):
        ax[0].plot(g.tau_threshold_s, g.threat_alerted_in_time_pct, "o-", label="HMD %d m" % hmd)
        ax[1].plot(g.tau_threshold_s, g.nuisance_alert_pct, "o-", label="HMD %d m" % hmd)
    ax[0].axhline(99, ls="--", c="k"); ax[0].set_title("R1: threats alerted at least 25 s ahead (%)")
    ax[1].axhline(5, ls="--", c="k"); ax[1].set_title("R2: nuisance alerts (%)")
    for a in ax:
        a.set_xlabel("tau threshold (s)"); a.grid(alpha=0.3); a.legend()
    fig.tight_layout(); fig.savefig("alert_validation.png", dpi=130)
