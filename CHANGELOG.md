# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-XX

### Added
- Initial release of PyPSA-Earth Energy System Optimization framework
- Energy system models:
  - Generation units (solar, wind, hydro, fossil fuels)
  - Transmission lines (AC/DC)
  - Storage systems (battery, pumped hydro, hydrogen)
  - Demand profiles with elasticity modeling
- Optimization algorithms:
  - Linear programming optimizer using Pyomo
  - Genetic algorithm optimizer for multi-objective optimization
- Simulation engine:
  - Scenario-based temporal simulations
  - Scenario manager with YAML/JSON support
  - Parallel processing support
- Data management:
  - Data loader for CSV, JSON, NetCDF, GeoJSON formats
  - Data preprocessor with cleaning, normalization, outlier detection
  - Renewable profile generation
- Visualization and reporting:
  - Generation stack plots
  - Demand vs generation plots
  - Storage operation visualization
  - Cost breakdown and emissions comparison
  - HTML report generation
- CLI interface with multiple commands:
  - `simulate`: Run energy system simulations
  - `optimize`: Run optimization algorithms
  - `visualize`: Generate visualizations
  - `analyze`: Comparative scenario analysis
- Comprehensive test suite with pytest
- Configuration management with Pydantic
- Project setup with pyproject.toml and uv integration

### Features
- Multi-objective optimization (cost, emissions, renewable penetration)
- Support for multiple solvers (Gurobi, CBC, GLPK)
- Flexible scenario management
- Comprehensive validation and error handling
- Type hints and docstrings throughout codebase

[0.1.0]: https://github.com/yourusername/pypsa-earth-optimization/releases/tag/v0.1.0

