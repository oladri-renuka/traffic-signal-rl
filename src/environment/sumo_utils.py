"""SUMO TraCI utilities for traffic simulation and state extraction."""

import os
import subprocess
import time
import numpy as np
from typing import Dict, List, Tuple
from threading import Lock

try:
    import traci
except ImportError:
    traci = None

import shutil as _shutil

from src.utils.logger import get_logger
from src.utils.sumo_config import (
    SUMO_BINARY, SUMO_CONFIG_FILE, GRID_SIZE, NUM_AGENTS,
    SIMULATION_TIME_STEP, STATE_DIMS, EPA_CO2_PER_MINUTE_KG, SUMO_HOME
)

logger = get_logger(__name__)


class TraCIManager:
    """Singleton manager for SUMO TraCI connections."""

    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.connection = None
        self.sumo_process = None
        self._initialized = True
        self.idle_minutes_accumulator = 0.0
        self.vehicle_counter = 0
        self.edge_list = []  # Will be populated from network
        self.tl_id_map = {}  # Maps agent_id -> actual SUMO traffic light ID

    def connect(self, gui=False, port=8813, verbose=False):
        """
        Connect to SUMO via TraCI.

        Args:
            gui: If True, use SUMO GUI; otherwise use sumo headless
            port: TraCI port number
            verbose: If True, show SUMO output

        Returns:
            traci connection object
        """
        if traci is None:
            raise ImportError("traci not available. Ensure SUMO_HOME is set correctly.")

        if self.connection is not None:
            logger.warning("TraCI already connected, returning existing connection")
            return self.connection

        try:
            sumo_cmd = [
                SUMO_BINARY if not gui else SUMO_BINARY.replace('sumo', 'sumo-gui'),
                '-c', str(SUMO_CONFIG_FILE),
                '--remote-port', str(port),
                '--seed', '42',
                '--no-warnings',
            ]

            logger.info(f"Starting SUMO on port {port}...")

            # Set SUMO_HOME in subprocess environment for TraCI to work
            env = os.environ.copy()
            env['SUMO_HOME'] = SUMO_HOME
            logger.debug(f"SUMO_HOME={SUMO_HOME}")

            # Redirect SUMO stderr to devnull to suppress routing errors
            # These are expected when random edge pairs aren't connected
            self.sumo_process = subprocess.Popen(
                sumo_cmd,
                env=env,
                stderr=subprocess.DEVNULL
            )

            # Wait longer for SUMO to start and listen
            logger.info("Waiting for SUMO to initialize...")
            time.sleep(3)

            # Connect via TraCI with retries
            max_retries = 15
            for attempt in range(max_retries):
                try:
                    traci.init(port=port)
                    self.connection = traci
                    logger.info(f"✓ TraCI connected on port {port}")
                    self.idle_minutes_accumulator = 0.0

                    # Build traffic light ID mapping
                    self._build_tl_mapping()

                    return self.connection
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.debug(f"Retry {attempt+1}/{max_retries}: {type(e).__name__}")
                        time.sleep(1)
                    else:
                        raise

        except Exception as e:
            logger.error(f"Failed to connect to SUMO: {e}")
            self.disconnect()
            raise

    def disconnect(self):
        """Disconnect from SUMO and clean up."""
        try:
            if self.connection is not None:
                traci.close()
                self.connection = None
                logger.info("TraCI disconnected")

            if self.sumo_process is not None:
                self.sumo_process.terminate()
                self.sumo_process.wait(timeout=5)
                self.sumo_process = None
                logger.info("SUMO process terminated")
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")

    def get_agent_state(self, agent_id: int) -> np.ndarray:
        """
        Extract state for a single agent (traffic light controller).

        State: [queue_0, queue_1, queue_2, queue_3,
                wait_0, wait_1, wait_2, wait_3,
                phase, elapsed_time,
                neighbor_queue_0-15]

        Args:
            agent_id: Agent ID (0-15)

        Returns:
            26-dimensional state array (normalized to [0, 1])
        """
        if self.connection is None:
            raise RuntimeError("TraCI not connected")

        state = np.zeros(STATE_DIMS, dtype=np.float32)
        tl_id = self._agent_id_to_tl(agent_id)

        try:
            # Get incoming lanes for this traffic light
            incoming_lanes = traci.trafficlight.getControlledLanes(tl_id)

            # Group lanes by direction (assuming standard grid layout)
            # For a grid, each intersection has 4 directions: N, S, E, W
            lane_groups = self._group_lanes_by_direction(incoming_lanes, agent_id)

            # Get queue lengths and wait times for each direction
            for dir_idx, lanes in enumerate(lane_groups[:4]):
                queue_len = sum(traci.lane.getLastStepVehicleNumber(lane) for lane in lanes)
                wait_time = sum(traci.lane.getWaitingTime(lane) for lane in lanes) / max(1, len(lanes))

                state[dir_idx] = queue_len / 20.0  # normalize: max 20 vehicles
                state[4 + dir_idx] = min(wait_time / 120.0, 1.0)  # normalize: max 120s

            # Current phase and elapsed time
            phase = traci.trafficlight.getPhase(tl_id)
            elapsed = traci.trafficlight.getPhaseDuration(tl_id) - traci.trafficlight.getNextSwitch(tl_id)

            state[8] = phase / 4.0  # 4 phases
            state[9] = elapsed / 120.0  # max phase duration ~120s

            # Get neighbor queue lengths (8 neighbors in grid, 2 lanes each = 16 values)
            neighbors = self._get_neighbor_agents(agent_id)
            for neighbor_idx, neighbor_id in enumerate(neighbors[:16]):
                if neighbor_id >= 0:
                    neighbor_tl = self._agent_id_to_tl(neighbor_id)
                    neighbor_lanes = traci.trafficlight.getControlledLanes(neighbor_tl)
                    if neighbor_lanes:
                        queue = sum(traci.lane.getLastStepVehicleNumber(lane) for lane in neighbor_lanes) / 20.0
                        state[10 + neighbor_idx] = min(queue, 1.0)

            return np.clip(state, 0.0, 1.0)

        except Exception as e:
            logger.error(f"Error extracting state for agent {agent_id}: {e}")
            return state

    def get_all_states(self) -> Dict[int, np.ndarray]:
        """Get states for all agents."""
        return {agent_id: self.get_agent_state(agent_id) for agent_id in range(NUM_AGENTS)}

    def set_agent_phase(self, agent_id: int, phase: int) -> None:
        """
        Set traffic light phase for an agent.

        Args:
            agent_id: Agent ID (0-15)
            phase: Phase number (0-3)
        """
        if self.connection is None:
            raise RuntimeError("TraCI not connected")

        tl_id = self._agent_id_to_tl(agent_id)
        try:
            traci.trafficlight.setPhase(tl_id, int(phase) % 4)
        except Exception as e:
            logger.error(f"Error setting phase for agent {agent_id}: {e}")

    def simulate_step(self) -> None:
        """Execute one simulation step."""
        if self.connection is None:
            raise RuntimeError("TraCI not connected")

        try:
            traci.simulation.step()
            # Accumulate idle time (vehicles with speed < 0.1 m/s)
            self._update_idle_time()
        except Exception as e:
            logger.error(f"Error during simulation step: {e}")

    def _update_idle_time(self) -> None:
        """Track total idle vehicle minutes."""
        try:
            vehicle_ids = traci.vehicle.getIDList()
            num_idle = sum(1 for vid in vehicle_ids if traci.vehicle.getSpeed(vid) < 0.1)
            # Convert to minutes: SIMULATION_TIME_STEP seconds × num_idle vehicles
            self.idle_minutes_accumulator += (SIMULATION_TIME_STEP / 60.0) * num_idle
        except Exception:
            pass

    def get_idle_co2(self) -> float:
        """Get total CO₂ emissions from idle time (kg)."""
        return self.idle_minutes_accumulator * EPA_CO2_PER_MINUTE_KG

    def reset_idle_accumulator(self) -> None:
        """Reset idle time accumulator for new episode."""
        self.idle_minutes_accumulator = 0.0

    def _build_tl_mapping(self) -> None:
        """Build mapping from agent IDs to actual SUMO traffic light IDs."""
        try:
            tl_ids = traci.trafficlight.getIDList()
            # Sort to ensure consistent mapping
            tl_ids = sorted([tl for tl in tl_ids if not tl.startswith(':')])

            # Map agents 0-15 to available traffic lights
            for agent_id in range(NUM_AGENTS):
                if agent_id < len(tl_ids):
                    self.tl_id_map[agent_id] = tl_ids[agent_id]
                else:
                    logger.warning(f"Not enough traffic lights for agent {agent_id}")

            logger.info(f"Mapped {len(self.tl_id_map)} agents to traffic lights")
        except Exception as e:
            logger.error(f"Failed to build TL mapping: {e}")

    def _agent_id_to_tl(self, agent_id: int) -> str:
        """Convert agent ID to SUMO traffic light ID."""
        # Use pre-built mapping if available
        if agent_id in self.tl_id_map:
            return self.tl_id_map[agent_id]
        # Fallback to old naming
        row = agent_id // GRID_SIZE
        col = agent_id % GRID_SIZE
        return f'{row}_{col}'

    def _group_lanes_by_direction(self, lanes: List[str], agent_id: int) -> List[List[str]]:
        """
        Group incoming lanes by direction (N, S, E, W).
        For a grid layout, this assumes standard naming conventions.
        """
        # Simplified: return lanes grouped arbitrarily (exact grouping depends on SUMO naming)
        n = len(lanes)
        return [lanes[i:i+1] for i in range(min(4, n))] + [[] for _ in range(max(0, 4 - n))]

    def _get_neighbor_agents(self, agent_id: int) -> List[int]:
        """Get list of neighboring agent IDs (up to 8 neighbors in grid)."""
        row = agent_id // GRID_SIZE
        col = agent_id % GRID_SIZE
        neighbors = []

        # 8-connected neighborhood
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                    neighbors.append(nr * GRID_SIZE + nc)
                else:
                    neighbors.append(-1)  # padding for non-existent neighbors

        return neighbors[:16]  # max 16 neighbors in our state space

    def get_network_stats(self) -> Dict:
        """Get network-wide statistics."""
        try:
            vehicle_ids = traci.vehicle.getIDList()
            total_wait = sum(traci.vehicle.getWaitingTime(vid) for vid in vehicle_ids)
            avg_wait = total_wait / len(vehicle_ids) if vehicle_ids else 0

            return {
                'num_vehicles': len(vehicle_ids),
                'total_wait': total_wait,
                'avg_wait': avg_wait,
                'co2_kg': self.get_idle_co2(),
            }
        except Exception as e:
            logger.error(f"Error getting network stats: {e}")
            return {}

    def is_connected(self) -> bool:
        """Check if TraCI is connected."""
        return self.connection is not None

    def add_vehicles_continuously(self, step: int) -> None:
        """
        Continuously add vehicles to simulation using Poisson distribution.
        Ensures SUMO never runs out of vehicles and connection stays alive.

        Args:
            step: Current simulation step (each step = 0.1s)
        """
        if not self.connection:
            return

        # Get edges from network on first call
        if not self.edge_list:
            try:
                self.edge_list = list(traci.edge.getIDList())
                self.edge_list = [e for e in self.edge_list if not e.startswith(':')]
                logger.debug(f"Loaded {len(self.edge_list)} edges for vehicle routing")
            except Exception as e:
                logger.error(f"Failed to get edge list: {e}")
                return

        if not self.edge_list:
            return

        # Only add vehicles every 10 steps (1 second)
        if step % 10 != 0:
            return

        # Determine arrival rate based on time of day
        # Simulate 1 hour: steps 0-36000 (3600s / 0.1s per step)
        elapsed_minutes = (step * SIMULATION_TIME_STEP) / 60.0

        # Peak hours: 8am-9am and 5pm-6pm
        # In our 1-hour simulation, simulate these as peak periods
        is_peak = False
        if 0 <= elapsed_minutes < 15:  # First 15 min = morning peak
            is_peak = True
        elif 45 <= elapsed_minutes < 60:  # Last 15 min = evening peak
            is_peak = True

        # Poisson arrival rate (vehicles per step)
        # Base: 0.03/step * 10 = 0.3/sec, Peak: 0.06/step * 10 = 0.6/sec
        lambda_rate = 0.06 if is_peak else 0.03

        # Generate Poisson arrivals
        try:
            num_arrivals = int(np.random.poisson(lambda_rate))

            for _ in range(num_arrivals):
                # Random O/D pair using safe routing
                from_edge = np.random.choice(self.edge_list)
                to_edge = np.random.choice(self.edge_list)

                # Ensure different edges
                attempts = 0
                while from_edge == to_edge and attempts < 5:
                    to_edge = np.random.choice(self.edge_list)
                    attempts += 1

                if from_edge == to_edge:
                    continue

                # Create route, skip invalid routes gracefully
                route_id = f'route_{self.vehicle_counter}'
                try:
                    traci.route.add(route_id, [from_edge, to_edge])
                except traci.TraCIException:
                    # Invalid route (edges not connected), skip this vehicle
                    continue

                # Add vehicle
                vehicle_id = f'veh_{self.vehicle_counter}'
                try:
                    traci.vehicle.add(
                        vehID=vehicle_id,
                        routeID=route_id,
                        typeID='car',
                        depart=step
                    )
                    self.vehicle_counter += 1
                except traci.TraCIException as e:
                    logger.debug(f"Could not add vehicle {vehicle_id}: {e}")

        except Exception as e:
            logger.error(f"Error adding vehicles at step {step}: {e}")

    def clear_vehicles(self) -> None:
        """Clear all vehicles and reset vehicle counter (keeps SUMO running)."""
        if not self.connection:
            return

        try:
            vehicle_ids = traci.vehicle.getIDList()
            for vehicle_id in vehicle_ids:
                traci.vehicle.remove(vehicle_id)
            self.vehicle_counter = 0
            logger.debug(f"Cleared {len(vehicle_ids)} vehicles for episode reset")
        except Exception as e:
            logger.error(f"Error clearing vehicles: {e}")
