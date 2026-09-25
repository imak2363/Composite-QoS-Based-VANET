"""
Rebuild modeling_table_expanded.csv from the raw expanded simulation dataset.

Source: /mnt/user-data/uploads/final_qos_expanded_9720.csv
        9,720 rows = 3,240 network conditions x 3 protocols (AODV, DSDV, AOMDV)

For each of the 3,240 conditions (grouped by Nodes, Speed_kmh, Flows,
Packet_Rate_pps, Mobility_Seed):
  1. Min-max normalize the five QoS metrics across the condition's 3 protocol
     rows (lower-better metrics inverted before/after normalization so that
     higher normalized value always means better).
  2. Composite QoS score = weighted sum using Table 1 weights:
        PDR                       0.30
        End-to-End Delay (inv)    0.25
        Jitter (inv)              0.20
        Throughput                0.15
        Normalized Routing Load (inv) 0.10
  3. Best_Protocol_Composite = argmax composite score within the condition.
  4. Best_Protocol_PDR = argmax raw PDR_Percent within the condition
     (single-metric baseline label).
  5. Build the 9-feature row (4 raw + 5 engineered, per the user-confirmed
     feature-engineering code) for that condition, attaching both labels.

Output: /mnt/user-data/outputs/modeling_table_expanded.csv
        one row per condition (3,240 rows), 9 features + 2 label columns
        (+ the composite/PDR scores per protocol kept as diagnostic columns).
"""
import pandas as pd
import numpy as np
import os

SRC = "/mnt/user-data/uploads/final_qos_expanded_9720.csv"
OUT_DIR = "/mnt/user-data/outputs"
OUT_PATH = os.path.join(OUT_DIR, "modeling_table_expanded.csv")

GROUP_COLS = ["Nodes", "Speed_kmh", "Flows", "Packet_Rate_pps", "Mobility_Seed"]

# metric -> (column name, higher_is_better, composite weight)
METRICS = {
    "PDR":       ("PDR_Percent",             True,  0.30),
    "Delay":     ("Avg_End_to_End_Delay_ms", False, 0.25),
    "Jitter":    ("Avg_Jitter_ms",           False, 0.20),
    "Throughput":("Throughput_kbps",         True,  0.15),
    "RoutingLoad":("Normalized_Routing_Load",False, 0.10),
}


def minmax_normalize_group(sub: pd.DataFrame) -> pd.DataFrame:
    """Given the 3 protocol rows for one condition, return a DataFrame of
    normalized (higher-is-better) scores per metric, plus the composite
    score and both winner labels."""
    out = pd.DataFrame(index=sub.index)
    for name, (col, higher_better, weight) in METRICS.items():
        vals = sub[col].astype(float)
        vmin, vmax = vals.min(), vals.max()
        if vmax == vmin:
            # No discrimination among protocols on this metric for this
            # condition; treat all three as equally (best) on this axis.
            norm = pd.Series(1.0, index=sub.index)
        else:
            if higher_better:
                norm = (vals - vmin) / (vmax - vmin)
            else:
                norm = (vmax - vals) / (vmax - vmin)
        out[f"norm_{name}"] = norm

    out["Composite_Score"] = sum(
        out[f"norm_{name}"] * weight for name, (_, _, weight) in METRICS.items()
    )
    out["PDR_Score"] = sub["PDR_Percent"].astype(float)
    out["Protocol"] = sub["Protocol"].values
    return out


def main():
    df = pd.read_csv(SRC)

    # sanity checks mirroring what was already confirmed interactively
    assert df.groupby(GROUP_COLS).size().eq(3).all(), "expected exactly 3 protocol rows per condition"
    assert set(df["Protocol"].unique()) == {"AODV", "DSDV", "AOMDV"}
    assert (df["QoS_Validation"] == "PASS").all()
    assert df["Packet_Size_Bytes"].nunique() == 1 and df["Packet_Size_Bytes"].iloc[0] == 512

    records = []
    for cond_key, sub in df.groupby(GROUP_COLS, sort=False):
        scored = minmax_normalize_group(sub)

        best_composite_row = scored.loc[scored["Composite_Score"].idxmax()]
        best_pdr_row = scored.loc[scored["PDR_Score"].idxmax()]

        nodes, speed, flows, rate, seed = cond_key

        offered_load = (flows * rate * 512 * 8) / 1000.0
        flows_per_node = flows / nodes
        rate_x_flows = rate * flows
        nodes_x_speed = nodes * speed
        speed_per_flow = speed / flows

        records.append({
            "Nodes": nodes,
            "Speed_kmh": speed,
            "Flows": flows,
            "Packet_Rate_pps": rate,
            "Mobility_Seed": seed,
            "Offered_Load_kbps": offered_load,
            "Flows_per_Node": flows_per_node,
            "Rate_x_Flows": rate_x_flows,
            "Nodes_x_Speed": nodes_x_speed,
            "Speed_per_Flow": speed_per_flow,
            "Best_Protocol_Composite": best_composite_row["Protocol"],
            "Best_Protocol_PDR": best_pdr_row["Protocol"],
            "AODV_Composite_Score": scored.loc[scored["Protocol"] == "AODV", "Composite_Score"].iloc[0],
            "DSDV_Composite_Score": scored.loc[scored["Protocol"] == "DSDV", "Composite_Score"].iloc[0],
            "AOMDV_Composite_Score": scored.loc[scored["Protocol"] == "AOMDV", "Composite_Score"].iloc[0],
            "AODV_PDR_Percent": scored.loc[scored["Protocol"] == "AODV", "PDR_Score"].iloc[0],
            "DSDV_PDR_Percent": scored.loc[scored["Protocol"] == "DSDV", "PDR_Score"].iloc[0],
            "AOMDV_PDR_Percent": scored.loc[scored["Protocol"] == "AOMDV", "PDR_Score"].iloc[0],
        })

    modeling = pd.DataFrame.from_records(records)
    assert len(modeling) == 3240

    os.makedirs(OUT_DIR, exist_ok=True)
    modeling.to_csv(OUT_PATH, index=False)

    print("Saved:", OUT_PATH, modeling.shape)
    print()
    print("Best_Protocol_Composite distribution:")
    print(modeling["Best_Protocol_Composite"].value_counts())
    print()
    print("Best_Protocol_PDR distribution:")
    print(modeling["Best_Protocol_PDR"].value_counts())
    print()
    agree = (modeling["Best_Protocol_Composite"] == modeling["Best_Protocol_PDR"]).mean()
    print(f"Agreement between composite and PDR-only labels: {agree*100:.2f}%")


if __name__ == "__main__":
    main()
