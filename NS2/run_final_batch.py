#!/usr/bin/env python3

import argparse
import csv
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

MANIFEST = ROOT / "04_NS2/final_experiment_manifest.csv"
NS_RUNNER = ROOT / "04_NS2/run_vanet_scenario.tcl"
QOS_PARSER = ROOT / "06_RESULTS/extract_qos.py"

NS_BINARY = (
    Path.home()
    / "ns-allinone-2.35"
    / "ns-2.35"
    / "ns"
)

PROGRESS = ROOT / "06_RESULTS/final_qos_progress.csv"
FAILURES = ROOT / "06_RESULTS/final_failed_instances.csv"

TRACE_DIR = ROOT / "05_TRACES"

# Final output order
PROTOCOLS = ["AODV", "DSDV", "AOMDV"]

# Run DSDV first because it is the legacy implementation
# that showed the pathological stall. This avoids wasting
# AODV/AOMDV time before detecting a bad attempt.
RUN_ORDER = ["DSDV", "AODV", "AOMDV"]

MAX_ATTEMPTS = 3

CHECK_INTERVAL_SECONDS = 5
STALL_SECONDS = 180
TRACE_START_TIMEOUT_SECONDS = 180


def load_csv(path):
    if not path.exists():
        return []

    with path.open(
        newline="",
        encoding="utf-8"
    ) as f:
        return list(csv.DictReader(f))


def atomic_write(path, rows):

    if not rows:
        path.unlink(missing_ok=True)
        return

    temp = path.with_suffix(".tmp")

    with temp.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=rows[0].keys()
        )

        w.writeheader()
        w.writerows(rows)

    temp.replace(path)


def trace_path(protocol, nodes, speed, flows, pps, seed):

    return TRACE_DIR / (
        f"trace_{nodes:03d}veh_"
        f"{speed:02d}kmh_"
        f"{flows:02d}flows_"
        f"{pps:02d}pps_"
        f"seed{seed:02d}_{protocol}.tr"
    )


def manifest_groups():

    rows = load_csv(MANIFEST)

    if len(rows) != 2700:
        raise RuntimeError(
            f"Manifest must contain 2700 rows; found {len(rows)}"
        )

    groups = defaultdict(list)

    for r in rows:
        groups[r["Instance_ID"]].append(r)

    ids = sorted(
        groups,
        key=lambda x: int(x.split("_")[1])
    )

    if len(ids) != 900:
        raise RuntimeError(
            f"Expected 900 instances; found {len(ids)}"
        )

    for iid in ids:

        g = groups[iid]

        if len(g) != 3:
            raise RuntimeError(
                f"{iid}: protocol triplet missing"
            )

        if {x["Protocol"] for x in g} != set(PROTOCOLS):
            raise RuntimeError(
                f"{iid}: invalid protocol triplet"
            )

    return groups, ids


def clean_progress(rows):

    groups = defaultdict(list)

    for r in rows:
        groups[r["Instance_ID"]].append(r)

    complete = set()
    clean = []

    for iid, g in groups.items():

        if (
            len(g) == 3
            and {x["Protocol"] for x in g}
                == set(PROTOCOLS)
            and all(
                x.get("QoS_Validation") == "PASS"
                for x in g
            )
        ):

            complete.add(iid)
            clean.extend(g)

    clean.sort(
        key=lambda r: (
            int(r["Instance_ID"].split("_")[1]),
            PROTOCOLS.index(r["Protocol"])
        )
    )

    return complete, clean


def failure_map():

    result = {}

    for row in load_csv(FAILURES):
        result[row["Instance_ID"]] = row

    return result


def save_failures(fmap):

    rows = list(fmap.values())

    rows.sort(
        key=lambda r:
            int(r["Instance_ID"].split("_")[1])
    )

    atomic_write(
        FAILURES,
        rows
    )


def simulation_seed(iid, retry_round, attempt):

    # FINAL POLICY:
    # exactly one deterministic NS RNG seed per matched instance.
    # Retry round/attempt never changes experimental conditions.
    num = int(iid.split("_")[1])

    return 100000 + num


def run_protocol(
    iid,
    protocol,
    nodes,
    speed,
    flows,
    pps,
    mobility_seed,
    sim_seed,
    retry_round,
    attempt
):

    trace = trace_path(
        protocol,
        nodes,
        speed,
        flows,
        pps,
        mobility_seed
    )

    temp_qos = Path(
        f"/tmp/{iid}_{protocol}_{sim_seed}_qos.csv"
    )

    ns_log = Path(
        f"/tmp/{iid}_{protocol}_{sim_seed}_ns.log"
    )

    trace.unlink(missing_ok=True)
    temp_qos.unlink(missing_ok=True)
    ns_log.unlink(missing_ok=True)

    cmd = [
        str(NS_BINARY),
        str(NS_RUNNER),
        protocol,
        str(nodes),
        str(speed),
        str(flows),
        str(pps),
        str(mobility_seed),
        str(sim_seed)
    ]

    start = time.monotonic()
    last_growth = start
    last_size = -1

    stalled = False
    reason = ""

    with ns_log.open(
        "w",
        encoding="utf-8"
    ) as log:

        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True
        )

        while proc.poll() is None:

            now = time.monotonic()

            if trace.exists():

                try:
                    size = trace.stat().st_size
                except FileNotFoundError:
                    size = -1

                if size != last_size:
                    last_size = size
                    last_growth = now

                elif (
                    size > 0
                    and now - last_growth >= STALL_SECONDS
                ):

                    stalled = True
                    reason = (
                        f"trace did not grow for "
                        f"{STALL_SECONDS} seconds"
                    )
                    break

            elif (
                now - start
                >= TRACE_START_TIMEOUT_SECONDS
            ):

                stalled = True
                reason = (
                    "trace was not created within "
                    f"{TRACE_START_TIMEOUT_SECONDS} seconds"
                )
                break

            time.sleep(
                CHECK_INTERVAL_SECONDS
            )

        if stalled:

            proc.terminate()

            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    output = ""

    if ns_log.exists():
        output = ns_log.read_text(
            encoding="utf-8",
            errors="replace"
        )

    ns_log.unlink(
        missing_ok=True
    )

    if stalled:

        trace.unlink(
            missing_ok=True
        )

        return (
            False,
            None,
            f"STALL: {reason}"
        )

    if (
        proc.returncode != 0
        or "NS2_RUN_PASS" not in output
        or not trace.exists()
        or trace.stat().st_size == 0
    ):

        trace.unlink(
            missing_ok=True
        )

        return (
            False,
            None,
            "NS-2 failed: "
            + output[-1000:].replace("\n", " ")
        )

    qos_cmd = [
        sys.executable,
        str(QOS_PARSER),
        str(trace),
        protocol,
        str(nodes),
        str(speed),
        str(flows),
        str(pps),
        str(mobility_seed),
        str(temp_qos)
    ]

    q = subprocess.run(
        qos_cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    if (
        q.returncode != 0
        or not temp_qos.exists()
    ):

        trace.unlink(missing_ok=True)
        temp_qos.unlink(missing_ok=True)

        return (
            False,
            None,
            "QoS extraction failed: "
            + q.stdout[-1000:].replace("\n", " ")
        )

    qos_rows = load_csv(temp_qos)

    temp_qos.unlink(
        missing_ok=True
    )

    if (
        len(qos_rows) != 1
        or qos_rows[0].get("QoS_Validation")
            != "PASS"
    ):

        trace.unlink(
            missing_ok=True
        )

        return (
            False,
            None,
            "QoS validation failed"
        )

    row = {
        "Instance_ID": iid,
        "Simulation_Seed": sim_seed,
        "Retry_Round": retry_round,
        "Triplet_Attempt": attempt,
        **qos_rows[0]
    }

    return (
        True,
        row,
        ""
    )


def show_status():

    completed, clean = clean_progress(
        load_csv(PROGRESS)
    )

    failures = failure_map()

    print("=" * 74)
    print("FINAL VANET EXPERIMENT PROGRESS")
    print("=" * 74)

    print(
        f"Completed instances : "
        f"{len(completed)} / 900"
    )

    print(
        f"Completed runs      : "
        f"{len(clean)} / 2700"
    )

    print(
        f"Pending failures    : "
        f"{len(failures)}"
    )

    print(
        f"Remaining instances : "
        f"{900 - len(completed)}"
    )

    print(
        f"Progress            : "
        f"{len(completed) / 900 * 100:.2f}%"
    )

    if completed:
        highest = max(
            completed,
            key=lambda x:
                int(x.split("_")[1])
        )

        print(
            f"Highest completed   : {highest}"
        )

    print(
        f"Progress CSV        : {PROGRESS}"
    )

    print(
        f"Failure queue       : {FAILURES}"
    )

    print("=" * 74)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-instances",
        type=int,
        default=None
    )

    parser.add_argument(
        "--status",
        action="store_true"
    )

    parser.add_argument(
        "--instance",
        type=str,
        default=None,
        help="Run one specific instance, e.g. INST_0075"
    )

    args = parser.parse_args()

    groups, ids = manifest_groups()

    progress = load_csv(PROGRESS)

    completed, clean = clean_progress(
        progress
    )

    if clean != progress:
        atomic_write(
            PROGRESS,
            clean
        )

    if args.status:
        show_status()
        return

    failures = failure_map()

    # New/untried work first.
    fresh = [
        iid for iid in ids
        if iid not in completed
        and iid not in failures
    ]

    # Previous failures are retried after fresh work.
    retry_pending = [
        iid for iid in ids
        if iid not in completed
        and iid in failures
    ]

    remaining = retry_pending + fresh

    if args.instance is not None:

        target = args.instance.upper()

        if target not in ids:
            raise SystemExit(
                f"Unknown instance: {target}"
            )

        if target in completed:
            print(
                f"{target} is already complete."
            )
            return

        remaining = [target]

    if args.max_instances is not None:

        if args.max_instances < 1:
            raise SystemExit(
                "--max-instances must be >= 1"
            )

        remaining = remaining[
            :args.max_instances
        ]

    print("=" * 78)
    print("FINAL ROBUST VANET BATCH")
    print("=" * 78)
    print(
        f"Completed          : "
        f"{len(completed)} / 900"
    )
    print(
        f"PASS rows          : "
        f"{len(clean)} / 2700"
    )
    print(
        f"Pending failures   : "
        f"{len(failures)}"
    )
    print(
        f"Instances this run : "
        f"{len(remaining)}"
    )
    print(
        f"Triplet retries    : "
        f"{MAX_ATTEMPTS} per round"
    )
    print("=" * 78)

    final_rows = clean[:]

    try:

        for pos, iid in enumerate(
            remaining,
            start=1
        ):

            r = groups[iid][0]

            nodes = int(r["Nodes"])
            speed = int(r["Speed_kmh"])
            flows = int(r["Flows"])
            pps = int(r["Packet_Rate_pps"])
            mobility_seed = int(
                r["Mobility_Seed"]
            )

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

            print()
            print(
                f"[{pos}/{len(remaining)}] "
                f"{iid} | "
                f"{nodes} nodes | "
                f"{speed} km/h | "
                f"{flows} flows | "
                f"{pps} pps | "
                f"mobility seed {mobility_seed}"
            )

            instance_pass = False
            last_error = ""
            last_sim_seed = ""

            for attempt in range(
                1,
                MAX_ATTEMPTS + 1
            ):

                sim_seed = simulation_seed(
                    iid,
                    retry_round,
                    attempt
                )

                last_sim_seed = sim_seed

                print(
                    f"  Attempt {attempt}/"
                    f"{MAX_ATTEMPTS}"
                    f" | NS seed {sim_seed}"
                )

                attempt_rows = []
                attempt_traces = []

                attempt_pass = True

                for protocol in RUN_ORDER:

                    print(
                        f"    {protocol:<5}: ",
                        end="",
                        flush=True
                    )

                    (
                        ok,
                        row,
                        error
                    ) = run_protocol(
                        iid,
                        protocol,
                        nodes,
                        speed,
                        flows,
                        pps,
                        mobility_seed,
                        sim_seed,
                        retry_round,
                        attempt
                    )

                    tr = trace_path(
                        protocol,
                        nodes,
                        speed,
                        flows,
                        pps,
                        mobility_seed
                    )

                    attempt_traces.append(
                        tr
                    )

                    if not ok:

                        print("FAIL")

                        last_error = (
                            f"{protocol}: {error}"
                        )

                        attempt_pass = False
                        break

                    attempt_rows.append(
                        row
                    )

                    print(
                        "PASS"
                        f" | PDR="
                        f"{float(row['PDR_Percent']):.2f}%"
                    )

                if attempt_pass and len(attempt_rows) == 3:

                    # Sort final triplet consistently.
                    attempt_rows.sort(
                        key=lambda x:
                            PROTOCOLS.index(
                                x["Protocol"]
                            )
                    )

                    final_rows.extend(
                        attempt_rows
                    )

                    atomic_write(
                        PROGRESS,
                        final_rows
                    )

                    for tr in attempt_traces:
                        tr.unlink(
                            missing_ok=True
                        )

                    failures.pop(
                        iid,
                        None
                    )

                    save_failures(
                        failures
                    )

                    print(
                        f"  {iid} TRIPLET : PASS"
                    )

                    print(
                        f"  CHECKPOINT    : "
                        f"{len(final_rows)} / 2700 rows"
                    )

                    instance_pass = True
                    break

                # Failed attempt: discard everything.
                for tr in attempt_traces:
                    tr.unlink(
                        missing_ok=True
                    )

                print(
                    "    Attempt discarded; "
                    "retrying complete triplet."
                )

            if not instance_pass:

                failures[iid] = {
                    "Instance_ID": iid,
                    "Nodes": nodes,
                    "Speed_kmh": speed,
                    "Flows": flows,
                    "Packet_Rate_pps": pps,
                    "Mobility_Seed": mobility_seed,
                    "Retry_Round": retry_round,
                    "Attempts_This_Round":
                        MAX_ATTEMPTS,
                    "Last_Simulation_Seed":
                        last_sim_seed,
                    "Last_Error":
                        last_error
                }

                save_failures(
                    failures
                )

                print(
                    f"  {iid} : DEFERRED"
                )

                print(
                    "  No failed/partial QoS row "
                    "was saved."
                )

                print(
                    "  Batch continues to next instance."
                )

    except KeyboardInterrupt:

        print()
        print(
            "USER STOP REQUESTED — completed "
            "triplets remain checkpointed."
        )

    print()
    show_status()


if __name__ == "__main__":
    main()
