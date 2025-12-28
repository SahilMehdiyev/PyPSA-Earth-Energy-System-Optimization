# PyPSA-Earth Energy System Optimization

A comprehensive Python framework for simulating and optimizing global energy systems using PyPSA-Earth. This project enables modeling of energy generation, transmission, storage, and consumption with support for linear programming and genetic algorithm optimization.

## 🎯 Project Overview

This project provides a complete toolkit for:

- **Energy System Modeling**: Generation, transmission, storage, and demand components
- **Optimization**: Linear programming (LP) and genetic algorithm (GA) approaches
- **Simulation**: Scenario-based temporal simulations with hourly resolution
- **Visualization**: Interactive plots and comprehensive reporting
- **Analysis**: Comparative scenario analysis and performance metrics

### Key Features

- 🌍 Multi-regional energy system modeling
- ⚡ Renewable and conventional generation support
- 🔋 Multiple storage technologies (battery, pumped hydro, hydrogen)
- 📊 Comprehensive visualization and reporting
- 🔧 Flexible scenario management
- 🚀 Parallel simulation support
- 📈 Multi-objective optimization

## 📋 Requirements

- **Python 3.10+**
- **PyPSA-Earth** framework
- **Optimization Solvers**: Gurobi (recommended), CBC, or GLPK
- **Data**: Energy system data (generation profiles, demand, transmission networks)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd PyPSA-Earth
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

Using `uv` (recommended):

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv pip install -e .
```

Or using pip:

```bash
pip install -e .
```

### 4. Install Optimization Solvers

**Gurobi** (Commercial, recommended):
```bash
# Download from https://www.gurobi.com/downloads/
# Follow installation instructions for your platform
pip install gurobipy
```

**CBC** (Open source):
```bash
# On macOS
brew install cbc

# On Ubuntu/Debian
sudo apt-get install coinor-cbc

# On Windows: Download from https://www.coin-or.org/download/binary/Cbc/
```

**GLPK** (Open source):
```bash
# On macOS
brew install glpk

# On Ubuntu/Debian
sudo apt-get install glpk-utils libglpk-dev
```

## 🎮 Quick Start

### Basic Simulation

```bash
python main.py simulate --scenario baseline --year 2025 --region europe
```

### Optimization Run

```bash
python main.py optimize --method lp --objective cost --solver gurobi
```

### Genetic Algorithm Optimization

```bash
python main.py optimize --method ga --generations 100 --population 200
```

### Generate Visualizations

```bash
python main.py visualize --scenario high_renewable --output results/visualizations/
```

## 📖 Usage Guide

### Command Line Interface

The main entry point provides several modes:

#### Simulation Mode

```bash
python main.py simulate [OPTIONS]

Options:
  --scenario TEXT     Scenario name (baseline, high_renewable, storage_heavy, grid_expansion)
  --year INTEGER      Target year (default: 2025)
  --region TEXT       Region identifier (e.g., europe, asia, global)
  --output PATH       Output directory for results
  --parallel          Enable parallel processing
```

#### Optimization Mode

```bash
python main.py optimize [OPTIONS]

Options:
  --method TEXT       Optimization method (lp, ga)
  --objective TEXT    Objective function (cost, emissions, multi)
  --solver TEXT       LP solver (gurobi, cbc, glpk)
  --generations INT   GA generations (default: 100)
  --population INT    GA population size (default: 200)
  --output PATH       Output directory
```

#### Analysis Mode

```bash
python main.py analyze [OPTIONS]

Options:
  --scenarios TEXT    Comma-separated scenario names
  --metrics TEXT      Metrics to compare
  --output PATH       Output directory
```

### Configuration

Edit `src/config/settings.py` to customize:

- Solver settings
- Optimization parameters
- Data paths
- Simulation parameters
- Output formats

### Creating Custom Scenarios

1. Create a YAML file in `data/scenarios/`:

```yaml
name: custom_scenario
year: 2030
region: europe
parameters:
  renewable_penetration: 0.8
  storage_capacity_gw: 100
  transmission_expansion: true
```

2. Run simulation:

```bash
python main.py simulate --scenario custom_scenario
```

## 🏗️ Architecture

### Project Structure

```
energy-system-optimization/
├── src/
│   ├── config/          # Configuration management
│   ├── models/          # Energy system models
│   ├── optimization/    # Optimization algorithms
│   ├── simulation/      # Simulation engine
│   ├── data/            # Data loading and preprocessing
│   └── visualization/   # Plotting and reporting
├── tests/               # Test suite
├── data/                # Data files
├── results/             # Simulation results
├── notebooks/           # Jupyter notebooks
└── docs/                # Documentation
```

### Module Overview

- **models/**: Core energy system components (generation, transmission, storage, demand)
- **optimization/**: LP and GA optimization implementations
- **simulation/**: Scenario management and temporal simulation
- **data/**: Data loading, preprocessing, and validation
- **visualization/**: Plotting, reporting, and export functionality

## 📊 Examples

See `notebooks/analysis.ipynb` for detailed examples and tutorials.

### Example: Simple Network Creation

```python
from src.models.energy_system import EnergySystem
from src.models.generation import GenerationUnit

# Create energy system
system = EnergySystem(name="test_system")

# Add generation units
solar = GenerationUnit(
    name="solar_farm",
    technology="solar",
    capacity_mw=100,
    efficiency=0.20
)
system.add_generation_unit(solar)

# Solve network
system.solve()
```

## 🧪 Testing

Run the test suite:

```bash
pytest tests/ -v
```

With coverage:

```bash
pytest tests/ --cov=src --cov-report=html
```

## 📈 Results

Simulation results are saved in `results/` directory:

- `results/simulations/`: Simulation outputs (JSON, CSV)
- `results/optimizations/`: Optimization results
- `results/visualizations/`: Generated plots and reports

## 🤝 Contributing

Contributions are welcome! Please see `CONTRIBUTING.md` for guidelines.

## 📄 License

This project is licensed under the MIT License.

## 📚 References

- **PyPSA-Earth**: Parzen, M., et al. "PyPSA-Earth: A multi-energy system model of the world." Applied Energy 347 (2023): 121477.
- **PyPSA Documentation**: https://pypsa.readthedocs.io/
- **PyPSA-Earth Repository**: https://github.com/pypsa-meets-earth/pypsa-earth
- **Pyomo Documentation**: https://pyomo.readthedocs.io/

## 🙏 Acknowledgments

This project builds upon the excellent PyPSA-Earth framework and the broader PyPSA ecosystem.

