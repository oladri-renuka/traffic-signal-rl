# Quick Start Guide

Get the multi-agent RL traffic signal system running in 5 steps.

## Prerequisites

```bash
# Install SUMO (requires system install)
# macOS: brew install sumo
# Linux: sudo apt-get install sumo sumo-tools

# Set SUMO_HOME
export SUMO_HOME=/usr/local/opt/sumo/share/sumo  # macOS
export SUMO_HOME=/usr/share/sumo                 # Linux
```

## Installation

```bash
pip install -r requirements.txt
```

## Step 1: Generate SUMO Network

Create the 4×4 grid traffic network with realistic demand:

```bash
python sumo/scripts/generate_network.py
```

**Output:**
- `sumo/network/grid_4x4.net.xml` - Network topology
- `sumo/network/grid_4x4.rou.xml` - Traffic demand (Poisson arrivals, peak hours)
- `sumo/network/grid_4x4.sumocfg` - SUMO configuration

## Step 2: Evaluate Baselines (Optional)

Compare three baseline strategies:

```bash
python experiments/eval_baselines.py --num-runs 3 --num-episodes 50
```

**Output:**
```
Fixed Timing:   1,245 kg CO₂
Actuated:       1,100 kg CO₂  (11.6% reduction)
Independent RL: 1,200 kg CO₂  (3.6% reduction)
```

Console shows:
- Mean ± std for each method
- Statistical significance tests (p-values)
- Plots: `experiments/results/learning_curves.png`, `co2_comparison.png`, etc.

## Step 3: Train Coordinated RL (Optional)

Train the multi-agent RL system:

```bash
python experiments/train_coordinated.py --episodes 100 --seed 42
```

**Output:**
- Trained model: `experiments/results/coordinated_model_seed42.pt`
- Training log: `experiments/results/training_results_seed42.json`

## Step 4: View Results Dashboard

Launch interactive Streamlit dashboard:

```bash
streamlit run dashboard/app.py
```

**Dashboard shows:**
- Live simulation grid with traffic lights
- Real-time CO₂ counter: "🌱 Saved 45.2 kg CO₂ vs Fixed Timing"
- Learning curves comparing all methods
- CO₂ savings bar chart
- Statistical summaries

## Step 5: Interpret Results

**Example output:**
```
EVALUATION SUMMARY
============================================================

Method: fixed_timing
Episodes run: 150
Avg wait time: 28.5 ± 3.2 seconds
CO₂ emissions: 1245.3 ± 125.8 kg
Avg reward: -28.4 ± 3.5

Method: coordinated_rl
Episodes run: 150
Avg wait time: 21.2 ± 2.8 seconds
CO₂ emissions: 925.6 ± 98.2 kg
Avg reward: -21.1 ± 2.9
Statistical significance: p=0.0023 ✓ SIGNIFICANT

============================================================
CO₂ SAVINGS SUMMARY
============================================================

fixed_timing     :  1245.32 kg CO₂
actuated         :  1100.48 kg CO₂ (saved:   -144.84 kg / -11.6%)
independent_rl   :  1200.15 kg CO₂ (saved:    -45.17 kg / -3.6%)
coordinated_rl   :   925.60 kg CO₂ (saved:   +319.72 kg / +25.7%)
```

## Key Configuration

Edit `src/utils/sumo_config.py` to customize:

```python
GRID_SIZE = 4              # Network size (4x4 = 16 agents)
EPISODE_LENGTH_SECONDS = 3600  # 1 hour simulation
EPA_CO2_PER_MINUTE = 2.4   # grams per minute (EPA idle factor)
LOCAL_REWARD_WEIGHT = 0.5  # vs GLOBAL_REWARD_WEIGHT = 0.5
```

## Troubleshooting

### SUMO not found
```bash
echo $SUMO_HOME
sumo --version
export SUMO_HOME=/path/to/sumo/share/sumo
```

### Out of memory
- Reduce `EPISODE_LENGTH_STEPS` in config
- Reduce batch size or evaluation episodes

### Permission denied
```bash
chmod +x sumo/scripts/generate_network.py
chmod +x experiments/train_coordinated.py
chmod +x experiments/eval_baselines.py
```

## Next Steps

1. **Analyze Results**: Check plots and statistical tests
2. **Experiment**: Try different reward weights, network sizes
3. **Deploy**: Load trained model for real-time inference
4. **Extend**: Add pedestrian constraints, multiple vehicle types

## Architecture

```
User Input (Dashboard/CLI)
        ↓
    Trainer/Evaluator
        ↓
  Policy Network (PyTorch)
        ↓
   PettingZoo Environment
        ↓
   SUMO Traffic Simulator ←→ TraCI API
        ↓
    Metrics & Statistics
        ↓
  Results + Visualizations
```

## Files

| File | Purpose |
|------|---------|
| `sumo/scripts/generate_network.py` | Generate network and traffic demand |
| `experiments/train_coordinated.py` | Train multi-agent RL |
| `experiments/eval_baselines.py` | Evaluate all baseline methods |
| `dashboard/app.py` | Streamlit dashboard |
| `src/environment/sumo_env.py` | Multi-agent environment |
| `src/training/trainer.py` | Training loop |
| `src/baselines/*.py` | Baseline implementations |
| `src/evaluation/*.py` | Metrics, statistics, plots |
