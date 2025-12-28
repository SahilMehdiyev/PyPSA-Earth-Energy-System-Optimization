"""Scenario management for energy system simulations."""

import yaml
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import logging

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


@dataclass
class Scenario:
    """Represents a simulation scenario."""
    
    name: str
    description: str = ""
    year: int = 2025
    region: str = "global"
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Scenario-specific settings
    renewable_penetration: float = 0.0
    storage_capacity_gw: float = 0.0
    transmission_expansion: bool = False
    demand_growth: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "year": self.year,
            "region": self.region,
            "parameters": self.parameters,
            "renewable_penetration": self.renewable_penetration,
            "storage_capacity_gw": self.storage_capacity_gw,
            "transmission_expansion": self.transmission_expansion,
            "demand_growth": self.demand_growth,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Scenario":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            year=data.get("year", 2025),
            region=data.get("region", "global"),
            parameters=data.get("parameters", {}),
            renewable_penetration=data.get("renewable_penetration", 0.0),
            storage_capacity_gw=data.get("storage_capacity_gw", 0.0),
            transmission_expansion=data.get("transmission_expansion", False),
            demand_growth=data.get("demand_growth", 0.0),
        )


class ScenarioManager:
    """Manages energy system scenarios."""
    
    def __init__(self, scenarios_path: Path) -> None:
        """
        Initialize scenario manager.
        
        Args:
            scenarios_path: Path to scenarios directory
        """
        self.scenarios_path = Path(scenarios_path)
        self.scenarios_path.mkdir(parents=True, exist_ok=True)
        self._scenarios: Dict[str, Scenario] = {}
        self._load_scenarios()
    
    def _load_scenarios(self) -> None:
        """Load all scenarios from directory."""
        # Load YAML files
        for yaml_file in self.scenarios_path.glob("*.yaml"):
            try:
                scenario = self._load_scenario_file(yaml_file)
                if scenario:
                    self._scenarios[scenario.name] = scenario
            except Exception as e:
                logger.warning(f"Error loading scenario from {yaml_file}: {e}")
        
        for yaml_file in self.scenarios_path.glob("*.yml"):
            try:
                scenario = self._load_scenario_file(yaml_file)
                if scenario:
                    self._scenarios[scenario.name] = scenario
            except Exception as e:
                logger.warning(f"Error loading scenario from {yaml_file}: {e}")
        
        # Load JSON files
        for json_file in self.scenarios_path.glob("*.json"):
            try:
                scenario = self._load_scenario_file(json_file)
                if scenario:
                    self._scenarios[scenario.name] = scenario
            except Exception as e:
                logger.warning(f"Error loading scenario from {json_file}: {e}")
        
        logger.info(f"Loaded {len(self._scenarios)} scenarios")
    
    def _load_scenario_file(self, file_path: Path) -> Optional[Scenario]:
        """Load scenario from file."""
        with open(file_path, "r") as f:
            if file_path.suffix in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            elif file_path.suffix == ".json":
                data = json.load(f)
            else:
                return None
        
        scenario = Scenario.from_dict(data)
        return scenario
    
    def load_scenario(self, name: str) -> Optional[Scenario]:
        """
        Load a specific scenario by name.
        
        Args:
            name: Scenario name
            
        Returns:
            Scenario object or None if not found
        """
        return self._scenarios.get(name)
    
    def list_scenarios(self) -> List[str]:
        """List all available scenario names."""
        return list(self._scenarios.keys())
    
    def save_scenario(self, scenario: Scenario) -> None:
        """
        Save scenario to file.
        
        Args:
            scenario: Scenario to save
        """
        file_path = self.scenarios_path / f"{scenario.name}.yaml"
        
        with open(file_path, "w") as f:
            yaml.dump(scenario.to_dict(), f, default_flow_style=False, sort_keys=False)
        
        self._scenarios[scenario.name] = scenario
        logger.info(f"Saved scenario '{scenario.name}' to {file_path}")
    
    def create_default_scenarios(self) -> None:
        """Create default scenario set."""
        scenarios = [
            Scenario(
                name="baseline",
                description="Baseline scenario with current system (fossil heavy)",
                year=2025,
                renewable_penetration=0.3,
                storage_capacity_gw=10.0,
                transmission_expansion=False,
                demand_growth=0.0,
            ),
            Scenario(
                name="high_renewable",
                description="High renewable penetration scenario (80%+)",
                year=2030,
                renewable_penetration=0.8,
                storage_capacity_gw=50.0,
                transmission_expansion=True,
                demand_growth=0.2,
            ),
            Scenario(
                name="storage_heavy",
                description="Massive battery/hydrogen deployment scenario",
                year=2035,
                renewable_penetration=0.6,
                storage_capacity_gw=200.0,
                transmission_expansion=False,
                demand_growth=0.3,
            ),
            Scenario(
                name="grid_expansion",
                description="Enhanced transmission network scenario",
                year=2030,
                renewable_penetration=0.7,
                storage_capacity_gw=30.0,
                transmission_expansion=True,
                demand_growth=0.15,
            ),
        ]
        
        for scenario in scenarios:
            self.save_scenario(scenario)
        
        logger.info(f"Created {len(scenarios)} default scenarios")

