#!/usr/bin/env python3

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path


# ================================================================
# FINAL VANET QoS EXTRACTOR
#
# Usage:
# python3 06_RESULTS/extract_qos.py \
#   TRACE_FILE PROTOCOL NODES SPEED FLOWS PPS SEED OUTPUT_CSV
#
# QoS definitions:
#
# Sent:
#   source-side AGT CBR send events
#
# Received:
#   destination-side AGT CBR receive events
#
# Packet loss:
#   Sent - Received
#
# PDR:
#   Received / Sent * 100
#
# End-to-end delay:
#   AGT receive time - AGT send time
#   matched using NS-2 global packet ID
#
# Jitter:
#   mean absolute difference between consecutive
#   packet delays within each communication flow
#
# Throughput:
#   successfully received APPLICATION payload bits
#   divided by the aggregate staggered traffic window
#
# Routing overhead:
#   routing packet transmissions at RTR layer,
#   counting both originated (s) and forwarded (f)
#   routing packets.
# ================================================================


if len(sys.argv) != 9:
    print(
        "Usage: python3 extract_qos.py "
        "TRACE PROTOCOL NODES SPEED FLOWS PPS SEED OUTPUT"
    )
    sys.exit(1)


trace_file = Path(sys.argv[1])
protocol = sys.argv[2].upper()
nodes = int(sys.argv[3])
speed = int(sys.argv[4])
flows = int(sys.argv[5])
pps = int(sys.argv[6])
seed = int(sys.argv[7])
output_file = Path(sys.argv[8])


PACKET_PAYLOAD_BYTES = 512

BASE_TRAFFIC_START = 20.0
BASE_TRAFFIC_STOP = 290.0

FLOW_START_STAGGER = 0.037

PER_FLOW_TRAFFIC_DURATION = (
    BASE_TRAFFIC_STOP - BASE_TRAFFIC_START
)

LATEST_FLOW_START = (
    BASE_TRAFFIC_START
    + FLOW_START_STAGGER * (flows - 1)
)

LATEST_FLOW_STOP = (
    BASE_TRAFFIC_STOP
    + FLOW_START_STAGGER * (flows - 1)
)

AGGREGATE_TRAFFIC_WINDOW = (
    LATEST_FLOW_STOP
    - BASE_TRAFFIC_START
)


if not trace_file.exists():
    print(f"ERROR: Trace file not found: {trace_file}")
    sys.exit(1)


# ------------------------------------------------
# CBR AGT event parser
# ------------------------------------------------
cbr_pattern = re.compile(
    r'^([sr])\s+'                 # event
    r'([0-9.]+)\s+'              # time
    r'_(\d+)_\s+'                # current node
    r'AGT\s+'
    r'\S+\s+'
    r'(\d+)\s+'                  # global packet ID
    r'cbr\s+'
    r'(\d+)\s+'                  # trace packet size
    r'.*?'
    r'\[(\d+):\d+\s+'            # source
    r'(\d+):\d+'                 # destination
)


send_times = {}
send_flows = {}

received_packets = set()

delays = []
flow_delays = defaultdict(list)

sent_count = 0
received_count = 0

routing_transmissions = 0
routing_originated = 0
routing_forwarded = 0

cbr_rtr_drops = 0

first_cbr_send = None
last_cbr_send = None

first_cbr_receive = None
last_cbr_receive = None


with trace_file.open(
    "r",
    encoding="utf-8",
    errors="replace"
) as f:

    for line in f:

        # ----------------------------------------
        # Application CBR traffic
        # ----------------------------------------
        match = cbr_pattern.match(line)

        if match:

            event = match.group(1)
            time = float(match.group(2))
            packet_id = int(match.group(4))
            src = int(match.group(6))
            dst = int(match.group(7))

            flow = (src, dst)

            if event == "s":

                sent_count += 1

                # Global packet ID is unique.
                send_times[packet_id] = time
                send_flows[packet_id] = flow

                if first_cbr_send is None:
                    first_cbr_send = time

                last_cbr_send = time


            elif event == "r":

                # Guard against accidental duplicated
                # application receive trace events.
                if packet_id in received_packets:
                    continue

                received_packets.add(packet_id)
                received_count += 1

                if first_cbr_receive is None:
                    first_cbr_receive = time

                last_cbr_receive = time

                if packet_id in send_times:

                    delay = time - send_times[packet_id]

                    if delay >= 0:

                        delays.append(delay)

                        flow = send_flows[packet_id]

                        flow_delays[flow].append(
                            (
                                send_times[packet_id],
                                delay
                            )
                        )


        # ----------------------------------------
        # Token-based checks for RTR events
        # ----------------------------------------
        parts = line.split()

        if len(parts) >= 8:

            event = parts[0]
            layer = parts[3]
            packet_type = parts[6]

            # Application data drops at routing layer.
            if (
                event == "D"
                and layer == "RTR"
                and packet_type.lower() == "cbr"
            ):
                cbr_rtr_drops += 1

            # Routing packet transmissions.
            routing_packet_types = {
                "AODV": {"AODV"},
                "AOMDV": {"AOMDV"},
                "DSDV": {"MESSAGE"},
            }

            valid_routing_types = routing_packet_types.get(
                protocol,
                {protocol}
            )

            if (
                layer == "RTR"
                and packet_type.upper() in valid_routing_types
                and event in {"s", "f"}
            ):

                routing_transmissions += 1

                if event == "s":
                    routing_originated += 1

                elif event == "f":
                    routing_forwarded += 1


# ================================================================
# QoS CALCULATIONS
# ================================================================

lost_count = sent_count - received_count


if sent_count > 0:

    pdr = (
        received_count
        / sent_count
        * 100.0
    )

    packet_loss_percent = (
        lost_count
        / sent_count
        * 100.0
    )

else:

    pdr = 0.0
    packet_loss_percent = 0.0


if delays:

    average_delay_s = (
        sum(delays)
        / len(delays)
    )

    average_delay_ms = (
        average_delay_s
        * 1000.0
    )

else:

    average_delay_s = 0.0
    average_delay_ms = 0.0


# ------------------------------------------------
# Per-flow jitter
# ------------------------------------------------
jitter_samples = []

for flow, records in flow_delays.items():

    records.sort(
        key=lambda x: x[0]
    )

    flow_packet_delays = [
        delay
        for _, delay in records
    ]

    for i in range(
        1,
        len(flow_packet_delays)
    ):

        jitter_samples.append(
            abs(
                flow_packet_delays[i]
                - flow_packet_delays[i - 1]
            )
        )


if jitter_samples:

    average_jitter_s = (
        sum(jitter_samples)
        / len(jitter_samples)
    )

    average_jitter_ms = (
        average_jitter_s
        * 1000.0
    )

else:

    average_jitter_s = 0.0
    average_jitter_ms = 0.0


# ------------------------------------------------
# Goodput / application throughput
#
# Use 512-byte CBR application payload rather than
# the 532-byte network-layer received trace size.
# ------------------------------------------------
received_payload_bits = (
    received_count
    * PACKET_PAYLOAD_BYTES
    * 8
)

throughput_kbps = (
    received_payload_bits
    / AGGREGATE_TRAFFIC_WINDOW
    / 1000.0
)


# ------------------------------------------------
# Routing load
# ------------------------------------------------
if received_count > 0:

    normalized_routing_load = (
        routing_transmissions
        / received_count
    )

else:

    normalized_routing_load = 0.0


# ================================================================
# VALIDATION
# ================================================================

expected_sent = (
    flows
    * pps
    * int(PER_FLOW_TRAFFIC_DURATION)
)

# CBR generates immediately at traffic start,
# then every interval until stop command.
# For the current validated runner this should
# match exactly.
sent_validation = (
    sent_count == expected_sent
)

receive_validation = (
    0 <= received_count <= sent_count
)

delay_validation = (
    len(delays) == received_count
)

time_validation = (
    first_cbr_send is not None
    and last_cbr_send is not None
    and first_cbr_send >= BASE_TRAFFIC_START
    and first_cbr_send <= LATEST_FLOW_START
    and last_cbr_send < LATEST_FLOW_STOP
)

status = (
    "PASS"
    if (
        sent_validation
        and receive_validation
        and delay_validation
        and time_validation
    )
    else "FAIL"
)


# ================================================================
# RESULT ROW
# ================================================================

row = {

    "Protocol": protocol,

    "Nodes": nodes,

    "Speed_kmh": speed,

    "Flows": flows,

    "Packet_Rate_pps": pps,

    "Mobility_Seed": seed,

    "Packet_Size_Bytes":
        PACKET_PAYLOAD_BYTES,

    "Base_Traffic_Start_s":
        BASE_TRAFFIC_START,

    "Base_Traffic_Stop_s":
        BASE_TRAFFIC_STOP,

    "Flow_Start_Stagger_s":
        FLOW_START_STAGGER,

    "Latest_Flow_Start_s":
        round(LATEST_FLOW_START, 6),

    "Latest_Flow_Stop_s":
        round(LATEST_FLOW_STOP, 6),

    "Per_Flow_Traffic_Duration_s":
        PER_FLOW_TRAFFIC_DURATION,

    "Aggregate_Traffic_Window_s":
        round(AGGREGATE_TRAFFIC_WINDOW, 6),

    "Sent_Packets":
        sent_count,

    "Received_Packets":
        received_count,

    "Lost_Packets":
        lost_count,

    "PDR_Percent":
        round(pdr, 6),

    "Packet_Loss_Percent":
        round(packet_loss_percent, 6),

    "Avg_End_to_End_Delay_ms":
        round(average_delay_ms, 6),

    "Avg_Jitter_ms":
        round(average_jitter_ms, 6),

    "Throughput_kbps":
        round(throughput_kbps, 6),

    "Routing_Packets_Tx":
        routing_transmissions,

    "Routing_Packets_Originated":
        routing_originated,

    "Routing_Packets_Forwarded":
        routing_forwarded,

    "Normalized_Routing_Load":
        round(
            normalized_routing_load,
            6
        ),

    "RTR_CBR_Drop_Events":
        cbr_rtr_drops,

    "First_CBR_Send_s":
        first_cbr_send,

    "Last_CBR_Send_s":
        last_cbr_send,

    "First_CBR_Receive_s":
        first_cbr_receive,

    "Last_CBR_Receive_s":
        last_cbr_receive,

    "QoS_Validation":
        status,
}


output_file.parent.mkdir(
    parents=True,
    exist_ok=True
)


with output_file.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=row.keys()
    )

    writer.writeheader()
    writer.writerow(row)


# ================================================================
# REPORT
# ================================================================

print("=" * 72)
print("FINAL VANET QoS EXTRACTION")
print("=" * 72)

print(f"Protocol                  : {protocol}")
print(f"Nodes                     : {nodes}")
print(f"Speed                     : {speed} km/h")
print(f"Flows                     : {flows}")
print(f"Packet rate               : {pps} pps")
print(f"Mobility seed             : {seed}")

print()

print(f"Sent packets              : {sent_count}")
print(f"Received packets          : {received_count}")
print(f"Lost packets              : {lost_count}")

print()

print(f"PDR                       : {pdr:.3f} %")
print(
    f"Packet loss               : "
    f"{packet_loss_percent:.3f} %"
)

print(
    f"Average E2E delay         : "
    f"{average_delay_ms:.3f} ms"
)

print(
    f"Average jitter            : "
    f"{average_jitter_ms:.3f} ms"
)

print(
    f"Throughput                : "
    f"{throughput_kbps:.3f} kbps"
)

print()

print(
    f"Routing transmissions     : "
    f"{routing_transmissions}"
)

print(
    f"  Originated              : "
    f"{routing_originated}"
)

print(
    f"  Forwarded               : "
    f"{routing_forwarded}"
)

print(
    f"Normalized routing load   : "
    f"{normalized_routing_load:.3f}"
)

print(
    f"RTR CBR drop events       : "
    f"{cbr_rtr_drops}"
)

print()

print(
    f"Expected sent packets     : "
    f"{expected_sent}"
)

print(
    f"Actual sent packets       : "
    f"{sent_count}"
)

print()

print(
    "QoS VALIDATION            :",
    status
)

print(
    f"Saved                     : "
    f"{output_file}"
)

print("=" * 72)


if status != "PASS":
    sys.exit(1)
