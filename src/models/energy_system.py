"""Main energy system model using PyPSA."""

import json
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

import pypsa
import pandas as pd
import numpy as np

from src.models.generation import GenerationUnit
from src.models.transmission import TransmissionLine
from src.models.storage import StorageSystem
from src.models.demand import DemandProfile


logger = logging.getLogger(__name__)


class EnergySystem:
    """Main energy system wrapper around PyPSA network."""
    
    def __init__(
        self,
        name: str = "energy_system",
        snapshots: Optional[pd.DatetimeIndex] = None,
        freq: str = "1H",
    ) -> None:
        """
        Initialize energy system.
        
        Args:
            name: System name
            snapshots: Time index for simulation
            freq: Frequency string if snapshots not provided
        """
        self.name = name
        self.network = pypsa.Network(name=name)
        
        if snapshots is not None:
            self.network.set_snapshots(snapshots)
        else:
            # Default: one year of hourly data
            self.network.set_snapshots(pd.date_range("2025-01-01", "2025-12-31", freq=freq))
        
        self.generation_units: Dict[str, GenerationUnit] = {}
        self.transmission_lines: Dict[str, TransmissionLine] = {}
        self.storage_systems: Dict[str, StorageSystem] = {}
        self.demand_profiles: Dict[str, DemandProfile] = {}
        
        logger.info(f"Initialized energy system '{name}' with {len(self.network.snapshots)} snapshots")
    
    def add_generation_unit(self, unit: GenerationUnit) -> None:
        """Add generation unit to the system."""
        if unit.name in self.generation_units:
            raise ValueError(f"Generation unit '{unit.name}' already exists")
        
        self.generation_units[unit.name] = unit
        
        # Add to PyPSA network
        if unit.node is None:
            unit.node = f"node_{len(self.network.buses)}"
        
        # Create bus if it doesn't exist
        if unit.node not in self.network.buses.index:
            self.network.add("Bus", unit.node)
        
        # Add generator
        self.network.add(
            "Generator",
            unit.name,
            bus=unit.node,
            p_nom=unit.capacity_mw,
            p_min_pu=unit.min_load_factor,
            p_max_pu=unit.max_load_factor,
            marginal_cost=unit.variable_cost_per_mwh,
            capital_cost=unit.fixed_cost_per_mw_year,
            efficiency=unit.efficiency,
        )
        
        # Add carrier type
        if unit.is_renewable:
            self.network.generators.loc[unit.name, "carrier"] = "renewable"
        else:
            self.network.generators.loc[unit.name, "carrier"] = "conventional"
        
        logger.debug(f"Added generation unit '{unit.name}' ({unit.technology.value})")
    
    def add_transmission_line(self, line: TransmissionLine) -> None:
        """Add transmission line to the system."""
        if line.name in self.transmission_lines:
            raise ValueError(f"Transmission line '{line.name}' already exists")
        
        self.transmission_lines[line.name] = line
        
        # Create buses if they don't exist
        if line.from_node not in self.network.buses.index:
            self.network.add("Bus", line.from_node)
        if line.to_node not in self.network.buses.index:
            self.network.add("Bus", line.to_node)
        
        # Add link (for DC) or line (for AC)
        if line.line_type.value == "dc":
            self.network.add(
                "Link",
                line.name,
                bus0=line.from_node,
                bus1=line.to_node,
                p_nom=line.capacity_mw,
                efficiency=1.0 - line.total_loss,
            )
        else:
            self.network.add(
                "Line",
                line.name,
                bus0=line.from_node,
                bus1=line.to_node,
                s_nom=line.s_nom or line.capacity_mw,
                x=line.x or 0.1,  # Default reactance
            )
        
        logger.debug(f"Added transmission line '{line.name}' ({line.line_type.value})")
    
    def add_storage_system(self, storage: StorageSystem) -> None:
        """Add storage system to the system."""
        if storage.name in self.storage_systems:
            raise ValueError(f"Storage system '{storage.name}' already exists")
        
        self.storage_systems[storage.name] = storage
        
        # Create bus if it doesn't exist
        if storage.node is None:
            storage.node = f"node_{len(self.network.buses)}"
        
        if storage.node not in self.network.buses.index:
            self.network.add("Bus", storage.node)
        
        # Add store (energy capacity)
        self.network.add(
            "Store",
            f"{storage.name}_store",
            bus=storage.node,
            e_nom=storage.energy_capacity_mwh,
            e_min_pu=storage.min_state_of_charge,
            e_max_pu=storage.max_state_of_charge,
            standing_loss=storage.standing_loss_per_hour,
        )
        
        # Add store link (charge/discharge)
        self.network.add(
            "Link",
            f"{storage.name}_charger",
            bus0=storage.node,
            bus1=storage.node,
            p_nom=storage.power_capacity_mw,
            efficiency=storage.charge_efficiency,
        )
        
        self.network.add(
            "Link",
            f"{storage.name}_discharger",
            bus0=storage.node,
            bus1=storage.node,
            p_nom=storage.power_capacity_mw,
            efficiency=storage.discharge_efficiency,
        )
        
        logger.debug(f"Added storage system '{storage.name}' ({storage.storage_type.value})")
    
    def add_demand_profile(self, profile: DemandProfile) -> None:
        """Add demand profile to the system."""
        if profile.name in self.demand_profiles:
            raise ValueError(f"Demand profile '{profile.name}' already exists")
        
        self.demand_profiles[profile.name] = profile
        
        # Create bus if it doesn't exist
        if profile.node not in self.network.buses.index:
            self.network.add("Bus", profile.node)
        
        # Generate demand profile
        demand_series = profile.generate_profile(self.network.snapshots)
        
        # Add load to network
        self.network.add(
            "Load",
            profile.name,
            bus=profile.node,
            p_set=demand_series,
        )
        
        logger.debug(f"Added demand profile '{profile.name}' at node '{profile.node}'")
    
    def remove_component(self, component_type: str, name: str) -> None:
        """Remove a component from the system."""
        if component_type == "generation":
            if name not in self.generation_units:
                raise ValueError(f"Generation unit '{name}' not found")
            del self.generation_units[name]
            self.network.remove("Generator", name)
        
        elif component_type == "transmission":
            if name not in self.transmission_lines:
                raise ValueError(f"Transmission line '{name}' not found")
            del self.transmission_lines[name]
            if name in self.network.lines.index:
                self.network.remove("Line", name)
            elif name in self.network.links.index:
                self.network.remove("Link", name)
        
        elif component_type == "storage":
            if name not in self.storage_systems:
                raise ValueError(f"Storage system '{name}' not found")
            del self.storage_systems[name]
            self.network.remove("Store", f"{name}_store")
            self.network.remove("Link", f"{name}_charger")
            self.network.remove("Link", f"{name}_discharger")
        
        elif component_type == "demand":
            if name not in self.demand_profiles:
                raise ValueError(f"Demand profile '{name}' not found")
            del self.demand_profiles[name]
            self.network.remove("Load", name)
        
        else:
            raise ValueError(f"Unknown component type: {component_type}")
    
    def validate(self) -> tuple[bool, List[str]]:
        """
        Validate system consistency.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check network consistency
        try:
            self.network.consistency_check()
        except Exception as e:
            errors.append(f"Network consistency error: {e}")
        
        # Check for isolated buses
        if len(self.network.buses) > 0:
            # Check if all buses are connected
            # (simplified check - could be enhanced)
            pass
        
        # Check generation capacity vs demand
        total_generation_capacity = sum(unit.capacity_mw for unit in self.generation_units.values())
        total_demand = sum(
            profile.base_demand_mw for profile in self.demand_profiles.values()
        )
        
        if total_generation_capacity < total_demand * 1.1:  # 10% margin
            errors.append(
                f"Generation capacity ({total_generation_capacity:.1f} MW) "
                f"may be insufficient for demand ({total_demand:.1f} MW)"
            )
        
        return len(errors) == 0, errors
    
    def solve(self, solver_name: str = "glpk", **solver_options) -> Dict[str, Any]:
        """
        Solve the energy system optimization problem.
        
        Args:
            solver_name: Solver to use
            **solver_options: Additional solver options
            
        Returns:
            Dictionary with solution status and results
        """
        logger.info(f"Solving energy system with solver: {solver_name}")
        
        try:
            self.network.optimize.create_model()
            self.network.optimize.solve_model(solver_name=solver_name, **solver_options)
            
            status = self.network.optimize.status
            objective = self.network.objective
            
            results = {
                "status": status,
                "objective": objective,
                "optimal": status == "ok",
            }
            
            if status == "ok":
                logger.info(f"Optimization successful. Objective: {objective:.2f}")
            else:
                logger.warning(f"Optimization status: {status}")
            
            return results
        
        except Exception as e:
            logger.error(f"Error solving network: {e}", exc_info=True)
            raise
    
    def get_results_summary(self) -> Dict[str, Any]:
        """Get summary of simulation results."""
        if not hasattr(self.network, "objective") or self.network.objective is None:
            return {"status": "not_solved"}
        
        summary = {
            "status": self.network.optimize.status,
            "objective": float(self.network.objective),
            "total_generation_mwh": float(
                self.network.generators_t.p.sum().sum()
            ) if hasattr(self.network, "generators_t") else 0.0,
            "total_demand_mwh": float(
                self.network.loads_t.p.sum().sum()
            ) if hasattr(self.network, "loads_t") else 0.0,
            "renewable_generation_mwh": 0.0,
            "conventional_generation_mwh": 0.0,
        }
        
        # Calculate renewable vs conventional
        if hasattr(self.network, "generators_t"):
            renewable_mask = self.network.generators.carrier == "renewable"
            if renewable_mask.any():
                summary["renewable_generation_mwh"] = float(
                    self.network.generators_t.p.loc[:, renewable_mask].sum().sum()
                )
            conventional_mask = self.network.generators.carrier == "conventional"
            if conventional_mask.any():
                summary["conventional_generation_mwh"] = float(
                    self.network.generators_t.p.loc[:, conventional_mask].sum().sum()
                )
        
        return summary
    
    def export_to_json(self, file_path: Path) -> None:
        """Export system configuration to JSON."""
        data = {
            "name": self.name,
            "snapshots": self.network.snapshots.tolist(),
            "generation_units": [unit.to_dict() for unit in self.generation_units.values()],
            "transmission_lines": [line.to_dict() for line in self.transmission_lines.values()],
            "storage_systems": [storage.to_dict() for storage in self.storage_systems.values()],
            "demand_profiles": [profile.to_dict() for profile in self.demand_profiles.values()],
        }
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Exported system configuration to {file_path}")
    
    @classmethod
    def from_json(cls, file_path: Path) -> "EnergySystem":
        """Load system configuration from JSON."""
        with open(file_path, "r") as f:
            data = json.load(f)
        
        snapshots = pd.DatetimeIndex(data["snapshots"])
        system = cls(name=data["name"], snapshots=snapshots)
        
        # Load components
        for unit_data in data.get("generation_units", []):
            unit = GenerationUnit.from_dict(unit_data)
            system.add_generation_unit(unit)
        
        for line_data in data.get("transmission_lines", []):
            line = TransmissionLine.from_dict(line_data)
            system.add_transmission_line(line)
        
        for storage_data in data.get("storage_systems", []):
            storage = StorageSystem.from_dict(storage_data)
            system.add_storage_system(storage)
        
        for profile_data in data.get("demand_profiles", []):
            profile = DemandProfile.from_dict(profile_data)
            system.add_demand_profile(profile)
        
        logger.info(f"Loaded system configuration from {file_path}")
        return system

