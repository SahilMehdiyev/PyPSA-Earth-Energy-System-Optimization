"""Simulation engine for energy systems."""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

import pandas as pd
import numpy as np

from src.config import Settings
from src.models.energy_system import EnergySystem
from src.simulation.scenario_manager import Scenario


logger = logging.getLogger(__name__)


class Simulator:
    """Simulates energy system operation over time."""
    
    def __init__(self, settings: Settings) -> None:
        """
        Initialize simulator.
        
        Args:
            settings: Configuration settings
        """
        self.settings = settings
    
    def run_scenario(
        self,
        scenario: Scenario,
        output_path: Optional[Path] = None,
        parallel: bool = False,
    ) -> Dict[str, Any]:
        """
        Run simulation for a scenario.
        
        Args:
            scenario: Scenario to simulate
            output_path: Path to save results
            parallel: Enable parallel processing
            
        Returns:
            Dictionary with simulation results
        """
        logger.info(f"Running simulation for scenario: {scenario.name}")
        
        # Create energy system from scenario
        system = self._create_system_from_scenario(scenario)
        
        # Validate system
        is_valid, errors = system.validate()
        if not is_valid:
            logger.warning(f"System validation issues: {errors}")
        
        # Run simulation
        start_time = datetime.now()
        
        try:
            # Solve network
            solve_results = system.solve(
                solver_name=self.settings.solver.name,
                timeout=self.settings.solver.timeout,
            )
            
            # Extract results
            results = self._extract_results(system, scenario, solve_results)
            
            end_time = datetime.now()
            results["simulation_time_seconds"] = (end_time - start_time).total_seconds()
            
            # Save results
            if output_path:
                self._save_results(results, output_path, scenario.name)
            
            logger.info(f"Simulation completed successfully for scenario: {scenario.name}")
            
            return results
        
        except Exception as e:
            logger.error(f"Error during simulation: {e}", exc_info=True)
            raise
    
    def _create_system_from_scenario(self, scenario: Scenario) -> EnergySystem:
        """Create energy system from scenario parameters."""
        # Create time index
        start_date = f"{scenario.year}-01-01"
        end_date = f"{scenario.year}-12-31"
        timestep = self.settings.simulation.timestep.replace('H', 'h')
        snapshots = pd.date_range(start_date, end_date, freq=timestep)
        
        system = EnergySystem(name=scenario.name, snapshots=snapshots)
        
        # Add components based on scenario
        # This is a simplified version - full implementation would load from data
        
        # Example: Add some generation units
        from src.models.generation import GenerationUnit, TechnologyType
        
        # Solar
        solar = GenerationUnit(
            name="solar_farm",
            technology=TechnologyType.SOLAR,
            capacity_mw=1000.0 * (1 + scenario.renewable_penetration),
            efficiency=0.20,
            variable_cost_per_mwh=0.0,
            fixed_cost_per_mw_year=50000.0,
            node="node_1",
        )
        system.add_generation_unit(solar)
        
        # Wind
        wind = GenerationUnit(
            name="wind_farm",
            technology=TechnologyType.WIND_ONSHORE,
            capacity_mw=800.0 * (1 + scenario.renewable_penetration),
            efficiency=0.35,
            variable_cost_per_mwh=0.0,
            fixed_cost_per_mw_year=60000.0,
            node="node_1",
        )
        system.add_generation_unit(wind)
        
        # Gas (conventional)
        gas = GenerationUnit(
            name="gas_plant",
            technology=TechnologyType.GAS,
            capacity_mw=500.0,
            efficiency=0.55,
            variable_cost_per_mwh=50.0,
            fixed_cost_per_mw_year=30000.0,
            emission_factor_kg_co2_per_mwh=350.0,
            node="node_1",
        )
        system.add_generation_unit(gas)
        
        # Add storage if specified
        if scenario.storage_capacity_gw > 0:
            from src.models.storage import StorageSystem, StorageType
            
            storage = StorageSystem(
                name="battery_storage",
                storage_type=StorageType.BATTERY,
                energy_capacity_mwh=scenario.storage_capacity_gw * 1000 * 4,  # 4 hours
                power_capacity_mw=scenario.storage_capacity_gw * 1000,
                charge_efficiency=0.9,
                discharge_efficiency=0.9,
                node="node_1",
            )
            system.add_storage_system(storage)
        
        # Add demand
        from src.models.demand import DemandProfile
        
        base_demand = 1000.0 * (1 + scenario.demand_growth)
        demand = DemandProfile(
            name="electricity_demand",
            node="node_1",
            base_demand_mw=base_demand,
        )
        system.add_demand_profile(demand)
        
        return system
    
    def _extract_results(
        self,
        system: EnergySystem,
        scenario: Scenario,
        solve_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract results from solved system."""
        summary = system.get_results_summary()
        
        results = {
            "scenario": scenario.name,
            "year": scenario.year,
            "region": scenario.region,
            "solve_status": solve_results.get("status", "unknown"),
            "objective": solve_results.get("objective", 0.0),
            "total_generation_mwh": summary.get("total_generation_mwh", 0.0),
            "total_demand_mwh": summary.get("total_demand_mwh", 0.0),
            "renewable_generation_mwh": summary.get("renewable_generation_mwh", 0.0),
            "conventional_generation_mwh": summary.get("conventional_generation_mwh", 0.0),
            "renewable_penetration": 0.0,
            "total_cost": 0.0,
            "total_emissions_kg_co2": 0.0,
            "storage_utilization": 0.0,
        }
        
        # Calculate renewable penetration
        total_gen = results["total_generation_mwh"]
        if total_gen > 0:
            results["renewable_penetration"] = results["renewable_generation_mwh"] / total_gen
        
        # Calculate costs and emissions (simplified)
        for gen_name, gen_unit in system.generation_units.items():
            if hasattr(system.network, "generators_t"):
                gen_series = system.network.generators_t.p[gen_name]
                annual_gen = gen_series.sum()
                results["total_cost"] += gen_unit.calculate_annual_cost(annual_gen)
                results["total_emissions_kg_co2"] += gen_unit.calculate_emissions(annual_gen)
        
        return results
    
    def _save_results(
        self,
        results: Dict[str, Any],
        output_path: Path,
        scenario_name: str,
    ) -> None:
        """Save simulation results."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save JSON results
        results_file = output_path / f"{scenario_name}_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_file}")
    
    def batch_simulate(
        self,
        scenarios: List[Scenario],
        output_path: Optional[Path] = None,
        parallel: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run multiple scenarios in batch.
        
        Args:
            scenarios: List of scenarios to simulate
            output_path: Path to save results
            parallel: Enable parallel processing
            
        Returns:
            Dictionary mapping scenario names to results
        """
        all_results = {}
        
        for scenario in scenarios:
            try:
                results = self.run_scenario(scenario, output_path, parallel)
                all_results[scenario.name] = results
            except Exception as e:
                logger.error(f"Error simulating scenario {scenario.name}: {e}")
                all_results[scenario.name] = {"error": str(e)}
        
        return all_results

