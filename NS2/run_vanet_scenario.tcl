# ================================================================
# FINAL VANET NS-2 SCENARIO RUNNER
# Arguments:
#   1 protocol       AODV / DSDV / AOMDV
#   2 node_count     50 / 100 / 150
#   3 speed_kmh      20 / 30 / 40 / 50
#   4 flow_count     5 / 7 / 10 / 12 / 15 / 18 / 20 / 22 / 25
#   5 packet_rate    2 / 3 / 4 / 5 / 6 / 8 packets/sec
#   6 mobility_seed  1 / 2 / 3 / 4 / 5
# ================================================================

if {$argc != 6 && $argc != 7} {
    puts "ERROR: Expected 6 or 7 arguments."
    puts "Usage:"
    puts "ns 04_NS2/run_vanet_scenario.tcl PROTOCOL NODES SPEED FLOWS PPS MOBILITY_SEED ?SIMULATION_SEED?"
    exit 1
}

set protocol      [lindex $argv 0]
set nodeCount     [expr {int([lindex $argv 1])}]
set speedKmh      [expr {int([lindex $argv 2])}]
set flowCount     [expr {int([lindex $argv 3])}]
set packetRate    [expr {int([lindex $argv 4])}]
set mobilitySeed  [expr {int([lindex $argv 5])}]

if {$argc == 7} {
    set simulationSeed [expr {int([lindex $argv 6])}]
} else {
    # Deterministic fallback
    set simulationSeed [expr {
        100000
        + ($nodeCount * 100)
        + ($speedKmh * 10)
        + $flowCount
        + ($packetRate * 1000)
        + $mobilitySeed
    }]
}

# ------------------------------------------------
# Validate experiment parameters
# ------------------------------------------------
if {[lsearch -exact {AODV DSDV AOMDV} $protocol] < 0} {
    puts "ERROR: Invalid protocol: $protocol"
    exit 1
}

if {[lsearch -exact {50 100 150} $nodeCount] < 0} {
    puts "ERROR: Invalid node count: $nodeCount"
    exit 1
}

if {[lsearch -exact {20 30 40 50} $speedKmh] < 0} {
    puts "ERROR: Invalid speed: $speedKmh"
    exit 1
}

if {[lsearch -exact {5 7 10 12 15 18 20 22 25} $flowCount] < 0} {
    puts "ERROR: Invalid flow count: $flowCount"
    exit 1
}

if {[lsearch -exact {2 3 4 5 6 8} $packetRate] < 0} {
    puts "ERROR: Invalid packet rate: $packetRate"
    exit 1
}

if {[lsearch -exact {1 2 3 4 5} $mobilitySeed] < 0} {
    puts "ERROR: Invalid mobility seed: $mobilitySeed"
    exit 1
}

# ------------------------------------------------
# Final experiment constants
# ------------------------------------------------
set simulationDuration 300.0
set trafficStart       20.0
set trafficStop        290.0
set flowStagger        0.037
set packetSize         512

set areaX 1000
set areaY 1000

# ------------------------------------------------
# Clean scenario/file naming
# ------------------------------------------------
set nodeTag  [format "%03d" $nodeCount]
set speedTag [format "%02d" $speedKmh]
set flowTag  [format "%02d" $flowCount]
set ppsTag   [format "%02d" $packetRate]
set seedTag  [format "%02d" $mobilitySeed]

set mobilityFile [format \
    "03_MOVE/mobility_%sveh_%skmh_seed%s.tcl" \
    $nodeTag $speedTag $seedTag]

set traceFile [format \
    "05_TRACES/trace_%sveh_%skmh_%sflows_%spps_seed%s_%s.tr" \
    $nodeTag $speedTag $flowTag $ppsTag $seedTag $protocol]

if {![file exists $mobilityFile]} {
    puts "ERROR: Mobility file not found:"
    puts $mobilityFile
    exit 1
}

# ------------------------------------------------
# Simulator + trace
# ------------------------------------------------
# Reproducible NS-2 random stream
ns-random $simulationSeed

set ns_ [new Simulator]

set tracefd [open $traceFile w]
$ns_ trace-all $tracefd

set topo [new Topography]
$topo load_flatgrid $areaX $areaY

create-god $nodeCount

set chan [new Channel/WirelessChannel]

$ns_ node-config \
    -adhocRouting $protocol \
    -llType LL \
    -macType Mac/802_11 \
    -ifqType Queue/DropTail/PriQueue \
    -ifqLen 50 \
    -antType Antenna/OmniAntenna \
    -propType Propagation/TwoRayGround \
    -phyType Phy/WirelessPhy \
    -channel $chan \
    -topoInstance $topo \
    -agentTrace ON \
    -routerTrace ON \
    -macTrace OFF \
    -movementTrace OFF

# ------------------------------------------------
# Create nodes
# ------------------------------------------------
for {set i 0} {$i < $nodeCount} {incr i} {
    set node_($i) [$ns_ node]

    $node_($i) set X_ 0.0
    $node_($i) set Y_ 0.0
    $node_($i) set Z_ 0.0
}

# ------------------------------------------------
# Load validated MOVE mobility
# ------------------------------------------------
source $mobilityFile

# ------------------------------------------------
# Deterministic communication pairs
#
# Same node_count + seed + flow_count produces the
# exact same src/dst pairs for all three protocols.
# ------------------------------------------------
set interval [expr {1.0 / double($packetRate)}]

# Exact offered traffic:
# 270 seconds × configured packet rate
set packetsPerFlow [expr {270 * $packetRate}]

puts "============================================================"
puts "NS-2 FINAL SCENARIO"
puts "============================================================"
puts "Protocol       : $protocol"
puts "Nodes          : $nodeCount"
puts "Speed          : $speedKmh km/h"
puts "Flows          : $flowCount"
puts "Packet rate    : $packetRate pps"
puts "Packet size    : $packetSize bytes"
puts "Mobility seed  : $mobilitySeed"
puts "Simulation RNG : $simulationSeed"
puts "Base traffic   : $trafficStart to $trafficStop s"
puts "Flow stagger   : $flowStagger s"
puts "Simulation     : $simulationDuration s"
puts ""
puts "Communication pairs:"

for {set i 0} {$i < $flowCount} {incr i} {

    set src [expr {
        ($mobilitySeed * 17 + $i * 13) % $nodeCount
    }]

    set dst [expr {
        ($mobilitySeed * 29 + $i * 31 + ($nodeCount / 2))
        % $nodeCount
    }]

    if {$dst == $src} {
        set dst [expr {($dst + 1) % $nodeCount}]
    }

    puts [format "  Flow %02d : node %d -> node %d" \
        [expr {$i + 1}] $src $dst]

    set udp_($i) [new Agent/UDP]
    $ns_ attach-agent $node_($src) $udp_($i)

    set sink_($i) [new Agent/Null]
    $ns_ attach-agent $node_($dst) $sink_($i)

    $ns_ connect $udp_($i) $sink_($i)

    set cbr_($i) [new Application/Traffic/CBR]
    $cbr_($i) set packetSize_ $packetSize
    $cbr_($i) set interval_ $interval
    $cbr_($i) set random_ 0
    $cbr_($i) set maxpkts_ $packetsPerFlow
    $cbr_($i) attach-agent $udp_($i)

    set flowStart [expr {
        $trafficStart + ($flowStagger * $i)
    }]

    $ns_ at $flowStart "$cbr_($i) start"
}

# ------------------------------------------------
# Clean simulation finish
# ------------------------------------------------
proc finish {} {
    global ns_ tracefd traceFile

    $ns_ flush-trace
    close $tracefd

    puts ""
    puts "============================================================"
    puts "NS2_RUN_PASS"
    puts "Trace file : $traceFile"
    puts "============================================================"

    exit 0
}

$ns_ at $simulationDuration "finish"

$ns_ run
