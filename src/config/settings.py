"""Configuration settings for the energy system optimization project."""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class SolverSettings(BaseModel):
    """Solver configuration settings."""

    name: str = Field(default="gurobi", description="Solver name (gurobi, cbc, glpk)")
    timeout: Optional[int] = Field(default=None, description="Solver timeout in seconds")
    threads: Optional[int] = Field(default=None, description="Number of threads")
    mip_gap: float = Field(default=0.01, description="MIP gap tolerance")
    log_level: int = Field(default=1, description="Logging level (0=silent, 1=normal, 2=verbose)")

    @field_validator("name")
    @classmethod
    def validate_solver_name(cls, v: str) -> str:
        """Validate solver name."""
        valid_solvers = ["gurobi", "cbc", "glpk"]
        if v.lower() not in valid_solvers:
            raise ValueError(f"Solver must be one of {valid_solvers}")
        return v.lower()


class OptimizationSettings(BaseModel):
    """Optimization algorithm settings."""

    method: str = Field(default="lp", description="Optimization method (lp, ga)")
    objective: str = Field(default="cost", description="Objective (cost, emissions, multi)")
    max_iterations: int = Field(default=1000, description="Maximum iterations")
    convergence_tolerance: float = Field(default=1e-6, description="Convergence tolerance")
    
    # Genetic Algorithm specific
    ga_population_size: int = Field(default=200, description="GA population size")
    ga_generations: int = Field(default=100, description="GA number of generations")
    ga_mutation_rate: float = Field(default=0.1, description="GA mutation rate")
    ga_crossover_rate: float = Field(default=0.8, description="GA crossover rate")
    ga_elite_size: int = Field(default=20, description="GA elite population size")

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        """Validate optimization method."""
        if v.lower() not in ["lp", "ga"]:
            raise ValueError("Method must be 'lp' or 'ga'")
        return v.lower()

    @field_validator("objective")
    @classmethod
    def validate_objective(cls, v: str) -> str:
        """Validate objective function."""
        valid_objectives = ["cost", "emissions", "multi"]
        if v.lower() not in valid_objectives:
            raise ValueError(f"Objective must be one of {valid_objectives}")
        return v.lower()


class SimulationSettings(BaseModel):
    """Simulation settings."""

    timestep: str = Field(default="1H", description="Time step (e.g., '1H', '30min')")
    start_date: str = Field(default="2025-01-01", description="Simulation start date")
    end_date: str = Field(default="2025-12-31", description="Simulation end date")
    parallel: bool = Field(default=False, description="Enable parallel processing")
    n_processes: Optional[int] = Field(default=None, description="Number of parallel processes")
    cache_results: bool = Field(default=True, description="Cache intermediate results")


class DataSettings(BaseModel):
    """Data paths and settings."""

    base_path: Path = Field(default=Path("data"), description="Base data directory")
    raw_data_path: Path = Field(default=Path("data/raw"), description="Raw data directory")
    processed_data_path: Path = Field(default=Path("data/processed"), description="Processed data directory")
    scenarios_path: Path = Field(default=Path("data/scenarios"), description="Scenarios directory")
    
    def __init__(self, **data):
        """Initialize with path resolution."""
        super().__init__(**data)
        # Resolve paths relative to project root
        project_root = Path(__file__).parent.parent.parent
        self.base_path = (project_root / self.base_path).resolve()
        self.raw_data_path = (project_root / self.raw_data_path).resolve()
        self.processed_data_path = (project_root / self.processed_data_path).resolve()
        self.scenarios_path = (project_root / self.scenarios_path).resolve()


class OutputSettings(BaseModel):
    """Output and results settings."""

    base_path: Path = Field(default=Path("results"), description="Base results directory")
    simulations_path: Path = Field(default=Path("results/simulations"), description="Simulations output")
    optimizations_path: Path = Field(default=Path("results/optimizations"), description="Optimizations output")
    visualizations_path: Path = Field(default=Path("results/visualizations"), description="Visualizations output")
    
    # Export formats
    export_formats: list[str] = Field(default=["json", "csv"], description="Export formats")
    figure_format: str = Field(default="png", description="Figure format (png, svg, pdf)")
    figure_dpi: int = Field(default=300, description="Figure resolution (DPI)")
    
    def __init__(self, **data):
        """Initialize with path resolution."""
        super().__init__(**data)
        project_root = Path(__file__).parent.parent.parent
        self.base_path = (project_root / self.base_path).resolve()
        self.simulations_path = (project_root / self.simulations_path).resolve()
        self.optimizations_path = (project_root / self.optimizations_path).resolve()
        self.visualizations_path = (project_root / self.visualizations_path).resolve()


class Settings(BaseModel):
    """Main settings class containing all configuration."""

    solver: SolverSettings = Field(default_factory=SolverSettings)
    optimization: OptimizationSettings = Field(default_factory=OptimizationSettings)
    simulation: SimulationSettings = Field(default_factory=SimulationSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    output: OutputSettings = Field(default_factory=OutputSettings)
    
    # System constraints
    min_renewable_penetration: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum renewable penetration")
    reserve_margin: float = Field(default=0.15, ge=0.0, description="Reserve margin requirement")
    max_transmission_loss: float = Field(default=0.10, ge=0.0, le=1.0, description="Maximum transmission loss")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[Path] = Field(default=None, description="Log file path")

    class Config:
        """Pydantic configuration."""
        frozen = False
        validate_assignment = True


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def load_settings_from_file(file_path: Path) -> Settings:
    """Load settings from a YAML or JSON file."""
    import yaml
    import json
    
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Settings file not found: {file_path}")
    
    with open(file_path, "r") as f:
        if file_path.suffix in [".yaml", ".yml"]:
            data = yaml.safe_load(f)
        elif file_path.suffix == ".json":
            data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    return Settings(**data)


def save_settings_to_file(settings: Settings, file_path: Path) -> None:
    """Save settings to a YAML or JSON file."""
    import yaml
    import json
    
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = settings.model_dump()
    
    with open(file_path, "w") as f:
        if file_path.suffix in [".yaml", ".yml"]:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        elif file_path.suffix == ".json":
            json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

