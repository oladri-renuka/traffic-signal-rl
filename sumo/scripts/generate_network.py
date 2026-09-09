#!/usr/bin/env python3
"""
Generate a 4x4 grid SUMO network for traffic signal RL experiments.
Uses SUMO's netgenerate tool to create the network topology.
"""

import os
import subprocess
import random
import math
import shutil
from pathlib import Path
import numpy as np

# Detect SUMO installation
SUMO_HOME = None
NETGENERATE = None

# 1. Try conda installation
try:
    import sumo as sumo_module
    conda_sumo_home = sumo_module.SUMO_HOME
    if os.path.exists(conda_sumo_home):
        SUMO_HOME = conda_sumo_home
        NETGENERATE = os.path.join(SUMO_HOME, 'bin', 'netgenerate')
except:
    pass

# 2. Try environment variable
if not NETGENERATE:
    env_sumo_home = os.getenv('SUMO_HOME')
    if env_sumo_home and os.path.exists(env_sumo_home):
        SUMO_HOME = env_sumo_home
        NETGENERATE = os.path.join(SUMO_HOME, 'bin', 'netgenerate')

# 3. Try which command
if not NETGENERATE:
    netgen_path = shutil.which('netgenerate')
    if netgen_path:
        NETGENERATE = netgen_path

# 4. Fallback paths
if not NETGENERATE:
    fallback_paths = [
        '/opt/anaconda3/lib/python3.13/site-packages/sumo/bin/netgenerate',
        '/opt/homebrew/opt/sumo/share/sumo/bin/netgenerate',
        '/usr/share/sumo/bin/netgenerate',
        '/usr/bin/netgenerate',
    ]
    for path in fallback_paths:
        if os.path.exists(path):
            NETGENERATE = path
            SUMO_HOME = os.path.dirname(os.path.dirname(path))
            break

if not NETGENERATE:
    raise RuntimeError(
        f"netgenerate not found. SUMO_HOME={SUMO_HOME}\n"
        "Install SUMO: pip install eclipse-sumo  OR  brew install sumo"
    )

OUTPUT_DIR = Path(__file__).parent.parent / 'network'


def generate_grid_network():
    """Generate a 4x4 grid network using netgenerate."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    net_file = OUTPUT_DIR / 'grid_4x4.net.xml'

    print(f"\n🌐 Generating 4×4 grid network...")
    print(f"  - Using netgenerate: {NETGENERATE}")

    # netgenerate command: creates 4x4 grid with 100m road segments
    cmd = [
        NETGENERATE,
        '--grid',
        '--grid.number=4',
        '--grid.length=100',
        '--grid.attach-length=0',
        '-p', '1.0',
        '-o', str(net_file)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ✗ Error: {result.stderr}")
        raise RuntimeError(f"netgenerate failed: {result.stderr}")

    print(f"  ✓ Generated 4×4 grid (16 intersections, 64 roads)")
    print(f"    File: {net_file}")
    return str(net_file)


def generate_traffic_demand():
    """
    Generate traffic demand with Poisson arrivals and peak hours.
    Creates routes and vehicle definitions.
    """
    rou_file = OUTPUT_DIR / 'grid_4x4.rou.xml'
    net_file = OUTPUT_DIR / 'grid_4x4.net.xml'

    print(f"\n📍 Generating traffic demand...")

    # Seed for reproducibility
    random.seed(42)

    # Read edges from the generated network file
    edges = []
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(net_file)
        root = tree.getroot()
        for edge in root.findall('.//edge'):
            edge_id = edge.get('id')
            if edge_id and not edge_id.startswith(':'):  # Skip junction edges
                edges.append(edge_id)
        print(f"  - Found {len(edges)} edges from network")
    except Exception as e:
        print(f"  ⚠ Warning: Could not read edges from network: {e}")
        print(f"  - Using fallback edge generation")
        # Fallback: generate grid edges
        grid_size = 4
        for i in range(grid_size):
            for j in range(grid_size):
                if j < grid_size - 1:
                    edges.append(f'{i}_{j}_{i}_{j+1}')
                    edges.append(f'{i}_{j+1}_{i}_{j}')
                if i < grid_size - 1:
                    edges.append(f'{i}_{j}_{i+1}_{j}')
                    edges.append(f'{i+1}_{j}_{i}_{j}')

    if not edges:
        raise RuntimeError("No edges found in network. Network generation may have failed.")

    with open(rou_file, 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n')

        # Define vehicle type
        f.write('  <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5.0" maxSpeed="25.0" />\n')

        # Generate random routes with Poisson arrivals
        # Peak hours: 8am-9am (3600-7200s) and 5pm-6pm (57600-61200s)
        # Off-peak: 2 vehicles per second, peak: 4 vehicles per second
        vehicle_id = 0
        total_time = 3600  # 1 hour simulation

        print(f"  - Generating Poisson traffic (1 hour = 3600 seconds)...")
        for t in range(0, total_time, 1):  # check every second
            # Progress
            if t % 600 == 0:  # Every 10 minutes
                print(f"    {t//60:2d}m / 60m: {vehicle_id:5d} vehicles so far")

            # Determine if peak hour
            hour = (t / 3600) % 24
            if 8 <= hour < 9 or 17 <= hour < 18:
                lambda_rate = 0.8  # ~0.8 vehicles/sec = 2880 vehicles/hour in peak
            else:
                lambda_rate = 0.4  # ~0.4 vehicles/sec = 1440 vehicles/hour off-peak

            # Poisson process: probability of arrival in this second
            num_arrivals = int(np.random.poisson(lambda_rate))
            for _ in range(num_arrivals):
                if len(edges) > 1:
                    from_edge = random.choice(edges)
                    to_edge = random.choice(edges)

                    # Ensure from_edge != to_edge
                    attempts = 0
                    while from_edge == to_edge and attempts < 5:
                        to_edge = random.choice(edges)
                        attempts += 1

                    if from_edge != to_edge:
                        route_id = f'route_{vehicle_id}'
                        f.write(f'  <route id="{route_id}" edges="{from_edge} {to_edge}" />\n')
                        f.write(f'  <vehicle id="veh_{vehicle_id}" type="car" route="{route_id}" depart="{t}" />\n')
                        vehicle_id += 1

        f.write('</routes>\n')

    print(f"  ✓ Traffic demand generated: {vehicle_id} vehicles")
    print(f"    File: {rou_file}")
    return str(rou_file)


def generate_sumo_config():
    """Generate SUMO configuration file."""
    cfg_file = OUTPUT_DIR / 'grid_4x4.sumocfg'
    net_file = 'grid_4x4.net.xml'
    rou_file = 'grid_4x4.rou.xml'

    print(f"\n⚙️  Generating SUMO configuration...")

    with open(cfg_file, 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">\n')
        f.write('  <input>\n')
        f.write(f'    <net-file value="{net_file}"/>\n')
        f.write(f'    <route-files value="{rou_file}"/>\n')
        f.write('  </input>\n')
        f.write('  <time>\n')
        f.write('    <begin value="0"/>\n')
        f.write('    <end value="99999"/>\n')
        f.write('    <step-length value="0.1"/>\n')
        f.write('  </time>\n')
        f.write('  <processing>\n')
        f.write('    <lateral-resolution value="0.8"/>\n')
        f.write('    <collision.action value="warn"/>\n')
        f.write('    <step-log value="false"/>\n')
        f.write('  </processing>\n')
        f.write('  <output>\n')
        f.write('    <summary-output value="summary.xml"/>\n')
        f.write('  </output>\n')
        f.write('</configuration>\n')

    print(f"  ✓ SUMO config created (3600 second simulation)")
    print(f"    File: {cfg_file}")
    return str(cfg_file)


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  SUMO Network Generator for Traffic Signal RL")
    print("="*60)

    try:
        net_file = generate_grid_network()
        rou_file = generate_traffic_demand()
        cfg_file = generate_sumo_config()

        print("\n" + "="*60)
        print("  ✓ SUCCESS! Network ready for simulation")
        print("="*60)
        print(f"\n📂 Output directory: {OUTPUT_DIR}")
        print(f"   - Network:  grid_4x4.net.xml  (4×4 grid)")
        print(f"   - Routes:   grid_4x4.rou.xml  (Poisson traffic)")
        print(f"   - Config:   grid_4x4.sumocfg  (3600s simulation)")
        print(f"\n🚗 Traffic profile:")
        print(f"   - Off-peak: ~0.4 vehicles/sec (1440/hour)")
        print(f"   - Peak:     ~0.8 vehicles/sec (2880/hour)")
        print(f"   - Random O/D pairs across 4×4 grid")
        print(f"\n✅ Next: Run baseline evaluation or training")
        print(f"   python experiments/eval_baselines.py")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        print("="*60 + "\n")
        exit(1)
