#!/usr/bin/env python3

import argparse
import csv
import importlib.util
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

OLD_RUNNER = ROOT / "04_NS2/run_final_batch.py"
OLD_QOS = ROOT / "06_RESULTS/final_qos_progress.csv"

NEW_MANIFEST = ROOT / "04_NS2/expanded_new_experiment_manifest.csv"
NEW_PROGRESS = ROOT / "06_RESULTS/expanded_new_qos_progress.csv"
NEW_FAILURES = ROOT / "06_RESULTS/expanded_new_failed_instances.csv"

FINAL_COMBINED = ROOT / "06_RESULTS/final_qos_expanded_9720.csv"


# ============================================================
# EXPANDED FACTORIAL DESIGN
# ============================================================

NODES = [50, 100, 150]
SPEEDS = [20, 30, 40, 50]

FLOWS = [
    5, 7, 10, 12, 15, 18, 20, 22, 25
]

PPS_LEVELS = [
    2, 3, 4, 5, 6, 8
]

MOBILITY_SEEDS = [1, 2, 3, 4, 5]

# These combinations already exist in the locked 2700-run study.
OLD_FLOWS = {5, 10, 15, 20, 25}
OLD_PPS = {2, 4, 8}

OLD_INSTANCE_COUNT = 900
OLD_RUN_COUNT = 2700

FINAL_UNIQUE_CONDITIONS = 648
FINAL_INSTANCE_COUNT = 3240
FINAL_RUN_COUNT = 9720

NEW_UNIQUE_CONDITIONS = 468
NEW_INSTANCE_COUNT = 2340
NEW_RUN_COUNT = 7020

DEFAULT_WORKERS = 2


# ============================================================
# LOAD ORIGINAL TESTED RUNNER
# ============================================================

spec = importlib.util.spec_from_file_location(
    "locked_final_runner",
    OLD_RUNNER
)

base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

# Faster process-completion polling for expanded batch.
# Simulation parameters/results are unchanged.
base.CHECK_INTERVAL_SECONDS = 1

PROTOCOLS = base.PROTOCOLS
RUN_ORDER = base.RUN_ORDER
MAX_ATTEMPTS = base.MAX_ATTEMPTS


# ============================================================
# BASIC CSV HELPERS
# ============================================================

def load_csv(path):
    if not path.exists():
        return []

    with path.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as f:
        return list(csv.DictReader(f))


def atomic_write(path, rows):
    if not rows:
        path.unlink(missing_ok=True)
        return

    temp = path.with_suffix(path.suffix + ".tmp")

    with temp.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)

    temp.replace(path)


def iid_number(iid):
    return int(iid.split("_")[1])


def sort_qos(rows):
    return sorted(
        rows,
        key=lambda r: (
            iid_number(r["Instance_ID"]),
            PROTOCOLS.index(r["Protocol"])
        )
    )


# ============================================================
# VERIFY LOCKED OLD DATA
# ============================================================

def locked_old_rows():

    rows = load_csv(OLD_QOS)

    complete, clean = base.clean_progress(rows)

    if len(rows) != OLD_RUN_COUNT:
        raise RuntimeError(
            f"Locked old QoS must contain {OLD_RUN_COUNT} rows; "
            f"found {len(rows)}"
        )

    if len(clean) != OLD_RUN_COUNT:
        raise RuntimeError(
            "Locked old QoS contains incomplete/invalid triplets."
        )

    if len(complete) != OLD_INSTANCE_COUNT:
        raise RuntimeError(
            f"Expected {OLD_INSTANCE_COUNT} old instances; "
            f"found {len(complete)}"
        )

    return sort_qos(clean)


# ============================================================
# BUILD ONLY THE NEW CONDITIONS
# ============================================================

def build_new_instances():

    conditions = []

    for nodes in NODES:
        for speed in SPEEDS:
            for flows in FLOWS:
                for pps in PPS_LEVELS:

                    # Already represented in locked experiment.
                    if (
                        flows in OLD_FLOWS
                        and pps in OLD_PPS
                    ):
                        continue

                    conditions.append(
                        (nodes, speed, flows, pps)
                    )

    if len(conditions) != NEW_UNIQUE_CONDITIONS:
        raise RuntimeError(
            f"Expected {NEW_UNIQUE_CONDITIONS} new conditions; "
            f"found {len(conditions)}"
        )

    instances = {}
    counter = OLD_INSTANCE_COUNT + 1

    for nodes, speed, flows, pps in conditions:
        for mobility_seed in MOBILITY_SEEDS:

            iid = f"INST_{counter:04d}"

            instances[iid] = {
                "Instance_ID": iid,
                "Nodes": nodes,
                "Speed_kmh": speed,
                "Flows": flows,
                "Packet_Rate_pps": pps,
                "Mobility_Seed": mobility_seed,
                "Simulation_Seed": 100000 + counter
            }

            counter += 1

    if len(instances) != NEW_INSTANCE_COUNT:
        raise RuntimeError(
            f"Expected {NEW_INSTANCE_COUNT} new instances; "
            f"found {len(instances)}"
        )

    if counter - 1 != FINAL_INSTANCE_COUNT:
        raise RuntimeError(
            "Final instance numbering audit failed."
        )

    return instances


# ============================================================
# WRITE REPRODUCIBLE NEW MANIFEST
# ============================================================

def write_manifest(instances):

    rows = []

    for iid in sorted(
        instances,
        key=iid_number
    ):
        x = instances[iid]

        for protocol in PROTOCOLS:
            rows.append({
                "Instance_ID": iid,
                "Protocol": protocol,
                "Nodes": x["Nodes"],
                "Speed_kmh": x["Speed_kmh"],
                "Flows": x["Flows"],
                "Packet_Rate_pps":
                    x["Packet_Rate_pps"],
                "Mobility_Seed":
                    x["Mobility_Seed"],
                "Simulation_Seed":
                    x["Simulation_Seed"],
                "Source":
                    "EXPANDED_REAL_NS2_SIMULATION"
            })

    if len(rows) != NEW_RUN_COUNT:
        raise RuntimeError(
            f"Expanded manifest expected {NEW_RUN_COUNT} rows; "
            f"found {len(rows)}"
        )

    atomic_write(
        NEW_MANIFEST,
        rows
    )


# ============================================================
# CLEAN NEW CHECKPOINT
# ============================================================

def clean_new_progress():

    rows = load_csv(NEW_PROGRESS)

    groups = defaultdict(list)

    for row in rows:
        groups[row["Instance_ID"]].append(row)

    complete = set()
    clean = []

    for iid, group in groups.items():

        if (
            len(group) == 3
            and {
                r["Protocol"]
                for r in group
            } == set(PROTOCOLS)
            and all(
                r.get("QoS_Validation") == "PASS"
                for r in group
            )
        ):
            complete.add(iid)
            clean.extend(group)

    clean = sort_qos(clean)

    if clean != rows:
        atomic_write(
            NEW_PROGRESS,
            clean
        )

    return complete, clean


# ============================================================
# FAILURE CHECKPOINT
# ============================================================

def load_failures():

    result = {}

    for row in load_csv(NEW_FAILURES):
        result[row["Instance_ID"]] = row

    return result


def save_failures(failures):

    rows = list(failures.values())

    rows.sort(
        key=lambda r:
            iid_number(r["Instance_ID"])
    )

    atomic_write(
        NEW_FAILURES,
        rows
    )


# ============================================================
# ONE COMPLETE MATCHED INSTANCE
# ============================================================

def run_instance(iid, x, retry_round):

    nodes = int(x["Nodes"])
    speed = int(x["Speed_kmh"])
    flows = int(x["Flows"])
    pps = int(x["Packet_Rate_pps"])
    mobility_seed = int(x["Mobility_Seed"])

    # One fixed deterministic seed for this instance.
    sim_seed = int(x["Simulation_Seed"])

    last_error = ""

    for attempt in range(
        1,
        MAX_ATTEMPTS + 1
    ):

        attempt_rows = []
        attempt_traces = []
        attempt_pass = True

        # Keep DSDV -> AODV -> AOMDV execution policy
        # from the locked original runner.
        for protocol in RUN_ORDER:

            ok, row, error = base.run_protocol(
                iid=iid,
                protocol=protocol,
                nodes=nodes,
                speed=speed,
                flows=flows,
                pps=pps,
                mobility_seed=mobility_seed,
                sim_seed=sim_seed,
                retry_round=retry_round,
                attempt=attempt
            )

            trace = base.trace_path(
                protocol,
                nodes,
                speed,
                flows,
                pps,
                mobility_seed
            )

            attempt_traces.append(trace)

            if not ok:
                last_error = (
                    f"{protocol}: {error}"
                )
                attempt_pass = False
                break

            attempt_rows.append(row)

        if (
            attempt_pass
            and len(attempt_rows) == 3
        ):

            attempt_rows.sort(
                key=lambda r:
                    PROTOCOLS.index(
                        r["Protocol"]
                    )
            )

            for trace in attempt_traces:
                trace.unlink(
                    missing_ok=True
                )

            return {
                "ok": True,
                "iid": iid,
                "rows": attempt_rows,
                "retry_round": retry_round,
                "attempt": attempt,
                "error": ""
            }

        # Failed attempt: discard every trace generated
        # by this triplet and retry with SAME sim seed.
        for trace in attempt_traces:
            trace.unlink(
                missing_ok=True
            )

    return {
        "ok": False,
        "iid": iid,
        "rows": [],
        "retry_round": retry_round,
        "attempt": MAX_ATTEMPTS,
        "error": last_error
    }


# ============================================================
# FINAL COMBINED DATASET CREATION
# ============================================================

def finalize_if_complete(
    new_complete,
    new_clean
):

    if (
        len(new_complete) != NEW_INSTANCE_COUNT
        or len(new_clean) != NEW_RUN_COUNT
    ):
        return False

    old = locked_old_rows()

    combined = sort_qos(
        old + new_clean
    )

    if len(combined) != FINAL_RUN_COUNT:
        raise RuntimeError(
            "Combined 9720-row audit failed."
        )

    groups = defaultdict(list)

    for row in combined:
        groups[row["Instance_ID"]].append(row)

    if len(groups) != FINAL_INSTANCE_COUNT:
        raise RuntimeError(
            "Combined 3240-instance audit failed."
        )

    for iid, group in groups.items():

        if len(group) != 3:
            raise RuntimeError(
                f"{iid}: incomplete protocol triplet"
            )

        if {
            r["Protocol"]
            for r in group
        } != set(PROTOCOLS):
            raise RuntimeError(
                f"{iid}: invalid protocol set"
            )

        if not all(
            r.get("QoS_Validation") == "PASS"
            for r in group
        ):
            raise RuntimeError(
                f"{iid}: QoS validation failure"
            )

    atomic_write(
        FINAL_COMBINED,
        combined
    )

    return True


# ============================================================
# STATUS
# ============================================================

def show_status(instances):

    complete, clean = clean_new_progress()
    failures = load_failures()

    new_done = len(complete)
    new_rows = len(clean)

    print("=" * 78)
    print("EXPANDED VANET DATASET STATUS")
    print("=" * 78)

    print(
        f"Target unique conditions : "
        f"{FINAL_UNIQUE_CONDITIONS}"
    )

    print(
        f"Old locked conditions    : 180"
    )

    print(
        f"New conditions           : "
        f"{NEW_UNIQUE_CONDITIONS}"
    )

    print(
        f"Old locked runs          : "
        f"{OLD_RUN_COUNT}"
    )

    print(
        f"New completed instances  : "
        f"{new_done} / {NEW_INSTANCE_COUNT}"
    )

    print(
        f"New completed runs       : "
        f"{new_rows} / {NEW_RUN_COUNT}"
    )

    print(
        f"New remaining instances  : "
        f"{NEW_INSTANCE_COUNT - new_done}"
    )

    print(
        f"Pending failures         : "
        f"{len(failures)}"
    )

    print(
        f"New progress             : "
        f"{new_done / NEW_INSTANCE_COUNT * 100:.2f}%"
    )

    print(
        f"Overall completed runs   : "
        f"{OLD_RUN_COUNT + new_rows} / "
        f"{FINAL_RUN_COUNT}"
    )

    print(
        f"Manifest                 : "
        f"{NEW_MANIFEST}"
    )

    print(
        f"New checkpoint           : "
        f"{NEW_PROGRESS}"
    )

    if FINAL_COMBINED.exists():
        print(
            f"FINAL COMBINED           : "
            f"{FINAL_COMBINED}"
        )

    print("=" * 78)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS
    )

    parser.add_argument(
        "--max-instances",
        type=int,
        default=None
    )

    parser.add_argument(
        "--instance",
        type=str,
        default=None
    )

    parser.add_argument(
        "--status",
        action="store_true"
    )

    args = parser.parse_args()

    if args.workers < 1:
        raise SystemExit(
            "--workers must be >= 1"
        )

    # Laptop safety: explicitly limit this project
    # to at most 2 parallel instances.
    if args.workers > 2:
        raise SystemExit(
            "This laptop profile is locked to maximum 2 workers."
        )

    # Never begin expanded work unless old 2700-run
    # research result passes integrity check.
    locked_old_rows()

    instances = build_new_instances()

    write_manifest(instances)

    if args.status:
        show_status(instances)
        return

    complete, final_rows = clean_new_progress()
    failures = load_failures()

    all_ids = sorted(
        instances,
        key=iid_number
    )

    if args.instance is not None:

        target = args.instance.upper()

        if target not in instances:
            raise SystemExit(
                f"Unknown expanded instance: {target}"
            )

        if target in complete:
            print(
                f"{target} is already complete."
            )
            return

        remaining = [target]

    else:

        # Retry previously deferred items first.
        retry_pending = [
            iid for iid in all_ids
            if iid not in complete
            and iid in failures
        ]

        fresh = [
            iid for iid in all_ids
            if iid not in complete
            and iid not in failures
        ]

        remaining = (
            retry_pending + fresh
        )

    if args.max_instances is not None:

        if args.max_instances < 1:
            raise SystemExit(
                "--max-instances must be >= 1"
            )

        remaining = remaining[
            :args.max_instances
        ]

    print("=" * 78)
    print("EXPANDED VANET PARALLEL BATCH")
    print("=" * 78)

    print(
        f"Workers                 : "
        f"{args.workers}"
    )

    print(
        f"Existing locked runs    : "
        f"{OLD_RUN_COUNT}"
    )

    print(
        f"New completed instances : "
        f"{len(complete)} / {NEW_INSTANCE_COUNT}"
    )

    print(
        f"Instances this launch   : "
        f"{len(remaining)}"
    )

    print(
        "Same-seed retries       : "
        f"{MAX_ATTEMPTS}"
    )

    print("=" * 78)

    if not remaining:

        finalized = finalize_if_complete(
            complete,
            final_rows
        )

        show_status(instances)

        if finalized:
            print(
                "\nEXPANDED DATASET COMPLETE : PASS"
            )

        return

    submitted = {}

    try:

        with ThreadPoolExecutor(
            max_workers=args.workers
        ) as executor:

            for iid in remaining:

                previous_round = int(
                    failures.get(
                        iid,
                        {}
                    ).get(
                        "Retry_Round",
                        0
                    )
                )

                retry_round = (
                    previous_round + 1
                )

                future = executor.submit(
                    run_instance,
                    iid,
                    instances[iid],
                    retry_round
                )

                submitted[future] = iid

            finished_this_launch = 0

            for future in as_completed(
                submitted
            ):

                iid = submitted[future]

                try:
                    result = future.result()

                except Exception as exc:

                    x = instances[iid]

                    previous_round = int(
                        failures.get(
                            iid,
                            {}
                        ).get(
                            "Retry_Round",
                            0
                        )
                    )

                    failures[iid] = {
                        "Instance_ID": iid,
                        "Nodes": x["Nodes"],
                        "Speed_kmh": x["Speed_kmh"],
                        "Flows": x["Flows"],
                        "Packet_Rate_pps":
                            x["Packet_Rate_pps"],
                        "Mobility_Seed":
                            x["Mobility_Seed"],
                        "Retry_Round":
                            previous_round + 1,
                        "Attempts_This_Round":
                            MAX_ATTEMPTS,
                        "Last_Simulation_Seed":
                            x["Simulation_Seed"],
                        "Last_Error":
                            f"WORKER_EXCEPTION: {exc}"
                    }

                    save_failures(
                        failures
                    )

                    print(
                        f"[FAIL] {iid} | {exc}"
                    )

                    continue

                if result["ok"]:

                    final_rows.extend(
                        result["rows"]
                    )

                    final_rows = sort_qos(
                        final_rows
                    )

                    atomic_write(
                        NEW_PROGRESS,
                        final_rows
                    )

                    complete.add(iid)

                    failures.pop(
                        iid,
                        None
                    )

                    save_failures(
                        failures
                    )

                    finished_this_launch += 1

                    x = instances[iid]

                    print(
                        f"[PASS] {iid} | "
                        f"{x['Nodes']} nodes | "
                        f"{x['Speed_kmh']} km/h | "
                        f"{x['Flows']} flows | "
                        f"{x['Packet_Rate_pps']} pps | "
                        f"seed {x['Mobility_Seed']} | "
                        f"{len(complete)}/"
                        f"{NEW_INSTANCE_COUNT}"
                    )

                else:

                    x = instances[iid]

                    failures[iid] = {
                        "Instance_ID": iid,
                        "Nodes": x["Nodes"],
                        "Speed_kmh": x["Speed_kmh"],
                        "Flows": x["Flows"],
                        "Packet_Rate_pps":
                            x["Packet_Rate_pps"],
                        "Mobility_Seed":
                            x["Mobility_Seed"],
                        "Retry_Round":
                            result["retry_round"],
                        "Attempts_This_Round":
                            MAX_ATTEMPTS,
                        "Last_Simulation_Seed":
                            x["Simulation_Seed"],
                        "Last_Error":
                            result["error"]
                    }

                    save_failures(
                        failures
                    )

                    print(
                        f"[DEFERRED] {iid} | "
                        f"{result['error']}"
                    )

    except KeyboardInterrupt:

        print()
        print(
            "USER STOP REQUESTED — completed "
            "triplets are already checkpointed."
        )

    complete, final_rows = clean_new_progress()

    finalized = finalize_if_complete(
        complete,
        final_rows
    )

    print()
    show_status(instances)

    if finalized:
        print()
        print("=" * 78)
        print("EXPANDED DATASET COMPLETE : PASS")
        print(
            f"FINAL RAW QoS : "
            f"{FINAL_COMBINED}"
        )
        print("=" * 78)


if __name__ == "__main__":
    main()
