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

    print(f"Generating network with command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error generating network: {result.stderr}")
        raise RuntimeError(f"netgenerate failed: {result.stderr}")

    print(f"Network generated successfully: {net_file}")
    return str(net_file)


def generate_traffic_demand():
    """
    Generate traffic demand with Poisson arrivals and peak hours.
    Creates routes and vehicle definitions.
    """
    rou_file = OUTPUT_DIR / 'grid_4x4.rou.xml'

    # Seed for reproducibility
    random.seed(42)

    # Define grid intersections (4x4)
    grid_size = 4
    intersections = [f'{i}_{j}' for i in range(grid_size) for j in range(grid_size)]
    edges = []

    # Collect all edges from the network
    # For a 4x4 grid: horizontal and vertical edges
    for i in range(grid_size):
        for j in range(grid_size):
            if j < grid_size - 1:  # horizontal edges (East-West)
                edges.append(f'{i}_{j}_{i}_{j+1}')
                edges.append(f'{i}_{j+1}_{i}_{j}')  # reverse
            if i < grid_size - 1:  # vertical edges (North-South)
                edges.append(f'{i}_{j}_{i+1}_{j}')
                edges.append(f'{i+1}_{j}_{i}_{j}')  # reverse

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

        for t in range(0, total_time, 1):  # check every second
            # Determine if peak hour
            hour = (t / 3600) % 24
            if 8 <= hour < 9 or 17 <= hour < 18:
                lambda_rate = 0.8  # ~0.8 vehicles/sec = 2880 vehicles/hour in peak
            else:
                lambda_rate = 0.4  # ~0.4 vehicles/sec = 1440 vehicles/hour off-peak

            # Poisson process: probability of arrival in this second
            num_arrivals = random.poisson(lambda_rate)
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

    print(f"Traffic demand generated: {rou_file} ({vehicle_id} vehicles)")
    return str(rou_file)


def generate_sumo_config():
    """Generate SUMO configuration file."""
    cfg_file = OUTPUT_DIR / 'grid_4x4.sumocfg'
    net_file = 'grid_4x4.net.xml'
    rou_file = 'grid_4x4.rou.xml'

    with open(cfg_file, 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">\n')
        f.write('  <input>\n')
        f.write(f'    <net-file value="{net_file}"/>\n')
        f.write(f'    <route-files value="{rou_file}"/>\n')
        f.write('  </input>\n')
        f.write('  <time>\n')
        f.write('    <begin value="0"/>\n')
        f.write('    <end value="3600"/>\n')
        f.write('    <step-length value="0.1"/>\n')
        f.write('  </time>\n')
        f.write('  <processing>\n')
        f.write('    <lateral-resolution value="0.8"/>\n')
        f.write('    <collision.action value="warn"/>\n')
        f.write('  </processing>\n')
        f.write('  <output>\n')
        f.write('    <summary-output value="summary.xml"/>\n')
        f.write('  </output>\n')
        f.write('</configuration>\n')

    print(f"SUMO config generated: {cfg_file}")
    return str(cfg_file)


if __name__ == '__main__':
    print("Generating SUMO 4x4 grid network...")
    net_file = generate_grid_network()
    print(f"\nGenerating traffic demand...")
    rou_file = generate_traffic_demand()
    print(f"\nGenerating SUMO configuration...")
    cfg_file = generate_sumo_config()
    print(f"\nSuccess! Network files ready in: {OUTPUT_DIR}")
    print(f"  - Network: {net_file}")
    print(f"  - Routes: {rou_file}")
    print(f"  - Config: {cfg_file}")
