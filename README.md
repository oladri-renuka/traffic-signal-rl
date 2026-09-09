# Multi-Agent RL for Traffic Signal Control

A comprehensive multi-agent reinforcement learning system for coordinating traffic signals in a 4×4 grid of intersections to minimize network-wide vehicle idle time and CO₂ emissions.

## Overview

This project implements a **multi-agent RL system** that coordinates 16 traffic signal agents using:
- **SUMO** traffic simulator with Python TraCI API
- **RLlib** for multi-agent RL (PPO algorithm)
- **PyTorch** for policy networks
- **PettingZoo** multi-agent environment
- **EPA idle emission factors** (2.4g CO₂/min/vehicle) for realistic environmental impact

The system is compared against three baselines:
1. **Fixed Timing**: Standard 30s/30s green/red cycles
2. **Actuated Control**: Queue-responsive signal extension
3. **Independent RL**: Single-agent RL per intersection (no coordination)

## Project Structure

```
traffic_signal_rl/
├── sumo/                              # SUMO network and configuration
│   ├── network/
│   │   ├── grid_4x4.net.xml          # 4×4 grid network (generated)
│   │   ├── grid_4x4.rou.xml          # Traffic demand (generated)
│   │   └── grid_4x4.sumocfg          # SUMO config (generated)
│   └── scripts/
│       └── generate_network.py        # Network/traffic generator
├── src/
│   ├── environment/
│   │   ├── sumo_env.py               # PettingZoo multi-agent environment
│   │   └── sumo_utils.py             # TraCI connection manager
│   ├── agents/
│   │   ├── policy_network.py         # PyTorch MLP networks
│   │   └── agent_config.py           # Agent configuration
│   ├── training/
│   │   ├── trainer.py                # RLlib training loop
│   │   ├── callbacks.py              # Custom callbacks
│   │   └── config.py                 # RLlib configuration
│   ├── baselines/
│   │   ├── fixed_timing.py           # Baseline 1
│   │   ├── actuated_control.py       # Baseline 2
│   │   └── independent_rl.py         # Baseline 3
│   ├── evaluation/
│   │   ├── evaluator.py              # Multi-run evaluator
│   │   ├── metrics.py                # CO₂, statistics, significance tests
│   │   └── plotter.py                # Plotting utilities
│   └── utils/
│       ├── sumo_config.py            # Configuration constants
│       └── logger.py                 # Logging setup
├── dashboard/
│   └── app.py                        # Streamlit dashboard
├── experiments/
│   ├── train_coordinated.py          # Training script
│   ├── eval_baselines.py             # Baseline evaluation script
│   └── results/                      # Results and plots
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites
- Python 3.8+
- SUMO >= 1.14.0 (download from [sumo.dlr.de](https://sumo.dlr.de))

### Setup

1. **Install SUMO** (if not already installed):
   ```bash
   # macOS
   brew install sumo
   
   # Linux (Ubuntu)
   sudo apt-get install sumo sumo-tools
   
   # Or download from https://sumo.dlr.de/docs/Downloads.php
   ```

2. **Set SUMO_HOME environment variable**:
   ```bash
   export SUMO_HOME=/usr/local/opt/sumo/share/sumo  # macOS
   export SUMO_HOME=/usr/share/sumo                 # Linux
   ```

3. **Clone and install dependencies**:
   ```bash
   cd traffic_signal_rl
   pip install -r requirements.txt
   ```

## Quick Start

### Generate SUMO Network

```bash
python sumo/scripts/generate_network.py
```

This creates:
- `sumo/network/grid_4x4.net.xml` - 4×4 grid network
- `sumo/network/grid_4x4.rou.xml` - Realistic traffic demand (Poisson arrivals, peak hours)
- `sumo/network/grid_4x4.sumocfg` - SUMO configuration

### Run Baseline Evaluation

Evaluate all three baselines (fixed timing, actuated, independent RL):

```bash
python experiments/eval_baselines.py --num-runs 3 --num-episodes 100
```

Output:
- Console summary with mean ± std metrics
- **CO₂ Savings**: "Actuated control saved 150 kg CO₂ vs Fixed Timing (12.1% reduction)"
- **Statistical Significance**: t-test p-values for each method
- JSON results: `experiments/results/baseline_evaluation.json`
- Plots: Learning curves, CO₂ comparison, waiting time distribution

### Train Coordinated Multi-Agent RL

```bash
python experiments/train_coordinated.py --seed 42 --episodes 1000 --eval-interval 100
```

Saves:
- Trained model: `experiments/results/coordinated_model_seed42/`
- Training log: `experiments/results/training_log_seed42.json`

### Launch Dashboard

```bash
streamlit run dashboard/app.py
```

Features:
- **Live Simulation**: Real-time 4×4 grid with traffic light states
- **CO₂ Counter**: "🌱 Saved 45.2 kg CO₂ vs Fixed Timing (22.3% reduction)"
- **Training Results**: Learning curves, CO₂ comparison, statistics
- **Interactive Controls**: Select method, run inference, download results

## System Design

### Environment: SUMOTrafficEnv (PettingZoo)

**Agents**: 16 traffic light controllers (one per intersection)

**State Space** (26-dim, normalized to [0, 1]):
- Queue lengths on 4 approaches (4 values)
- Average waiting time per approach (4 values)
- Current phase and elapsed time (2 values)
- Neighbor agents' queue lengths (16 values)

**Action Space** (discrete):
- 4 actions: Phase transitions (EW-green → NS-green, hold, etc.)

**Reward Function**:
```
reward = 0.5 × local_reward + 0.5 × global_reward + pedestrian_penalty

local_reward = -avg_waiting_time_at_intersection
global_reward = -avg_waiting_time_network_wide
pedestrian_penalty = -0.1 if red_phase > 60s else 0
```

### RL Training: RLlib PPO

**Algorithm**: PPO (Proximal Policy Optimization)
- Shared policy across all agents (single NN for 16 agents)
- Synchronized multi-agent updates
- Centralized critic for advantage estimation

**Training Parameters**:
- Learning rate: 5e-4
- Batch size: 128
- Gamma: 0.99
- Entropy coefficient: 0.01
- Episode length: 3600 simulated seconds (1 hour)
- Training episodes: 1000
- Evaluation interval: Every 100 episodes

**Policy Network**: Small MLP
```
Input (26-dim) → Dense(128, ReLU) → Dense(128, ReLU) → Output(4 logits)
```

### Baselines

1. **Fixed Timing** (30s/30s): Deterministic, no learning
2. **Actuated Control**: Queue threshold (5 vehicles) extends green up to 60s
3. **Independent RL**: Each agent learns only local reward (no global component)

### Evaluation

**Metrics**:
- Average waiting time (seconds)
- CO₂ emissions (kg): idle_minutes × 2.4g/min ÷ 1000
- CO₂ savings vs fixed timing (percentage improvement)

**Statistical Tests**:
- Paired t-test between coordinated RL and baselines
- Cohen's d effect size
- Significance threshold: p < 0.05

**Multiple Runs**:
- Each method evaluated 3 times with different SUMO seeds
- Mean ± std reported across runs

## Key Features

✅ **Real SUMO Simulator**: No toy environments; realistic traffic dynamics  
✅ **Multi-Agent Coordination**: Global reward component drives cooperation  
✅ **EPA Emissions**: 2.4g CO₂/min/vehicle (validated environmental metric)  
✅ **Three Baselines**: Fixed, actuated, independent RL for comprehensive comparison  
✅ **Statistical Rigor**: t-tests, effect sizes, multiple runs, confidence intervals  
✅ **Live Dashboard**: Real-time visualization with CO₂ counter  
✅ **Modular Design**: Easy to extend with new baselines, reward functions, etc.

## Expected Results

Based on the design, the coordinated multi-agent RL system should achieve:
- **15-25% reduction** in average waiting time vs fixed timing
- **18-30% reduction** in CO₂ emissions vs fixed timing
- **Statistically significant** improvement (p < 0.05) across runs

Actuated control typically performs ~10-15% better than fixed timing, while independent RL without coordination is usually similar to fixed timing.

## Configuration

Edit constants in `src/utils/sumo_config.py`:

```python
GRID_SIZE = 4                    # 4×4 grid = 16 agents
EPISODE_LENGTH_SECONDS = 3600    # 1 simulated hour
NUM_ACTIONS = 4                  # Phase options per intersection
EPA_CO2_PER_MINUTE = 2.4         # grams CO2/min (EPA idle factor)
LOCAL_REWARD_WEIGHT = 0.5        # Weight for local rewards
GLOBAL_REWARD_WEIGHT = 0.5       # Weight for network reward
```

## Troubleshooting

### SUMO Connection Errors
```
Error: "Cannot connect to SUMO on port 8813"
```
**Solution**: Check SUMO_HOME is set correctly:
```bash
echo $SUMO_HOME
sumo --version
```

### TraCI Import Errors
```
ImportError: No module named 'traci'
```
**Solution**: SUMO Python bindings are installed automatically with SUMO. Add to PATH:
```bash
export PYTHONPATH="${SUMO_HOME}/tools:$PYTHONPATH"
```

### Out of Memory (OOM)
- Reduce `BATCH_SIZE` in config
- Reduce number of evaluation episodes
- Use smaller network: `[64, 64]` instead of `[128, 128]`

## Future Extensions

- [ ] Distributed training with ray.tune
- [ ] Curriculum learning (start with fixed timing, gradually enable coordination)
- [ ] Pedestrian safety as hard constraint
- [ ] Travel time prediction model
- [ ] Multi-modal networks (bikes, buses)
- [ ] Real-world traffic data integration
- [ ] Model deployment to actual SCATS/SCOOT systems

## References

- SUMO: https://sumo.dlr.de
- RLlib: https://docs.ray.io/en/latest/rllib.html
- PettingZoo: https://www.pettingzoo.ml
- EPA Idle Emissions: https://www.epa.gov/sites/production/files/2015-08/documents/420f05018.pdf

## License

MIT

## Author

Multi-Agent RL Traffic Signal Control Team

---

**Last Updated**: 2024
