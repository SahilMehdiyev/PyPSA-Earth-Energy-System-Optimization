"""Linear programming optimization using Pyomo."""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

import pandas as pd
import numpy as np
from pyomo.environ import (
    ConcreteModel, Var, Objective, Constraint, Set, Param,
    minimize, maximize, SolverFactory, value, NonNegativeReals, Reals,
)
from pyomo.opt import SolverStatus, TerminationCondition

from src.config import Settings
from src.models.energy_system import EnergySystem


logger = logging.getLogger(__name__)


class LinearProgrammingOptimizer:
    """Linear programming optimizer for energy systems."""
    
    def __init__(self, settings: Settings) -> None:
        """
        Initialize LP optimizer.
        
        Args:
            settings: Configuration settings
        """
        self.settings = settings
        self.model: Optional[ConcreteModel] = None
        self.results: Optional[Dict[str, Any]] = None
    
    def _detect_solver(self) -> str:
        """Detect available solver."""
        solver_name = self.settings.solver.name.lower()
        
        # Check solver availability
        try:
            solver = SolverFactory(solver_name)
            if solver.available():
                logger.info(f"Using solver: {solver_name}")
                return solver_name
        except Exception as e:
            logger.warning(f"Solver {solver_name} not available: {e}")
        
        # Try fallback solvers
        fallback_solvers = ["cbc", "glpk", "cplex"]
        for fallback in fallback_solvers:
            if fallback == solver_name:
                continue
            try:
                solver = SolverFactory(fallback)
                if solver.available():
                    logger.info(f"Using fallback solver: {fallback}")
                    return fallback
            except Exception:
                continue
        
        raise RuntimeError("No suitable solver found. Please install Gurobi, CBC, or GLPK.")
    
    def build_model(self, system: EnergySystem) -> ConcreteModel:
        """
        Build Pyomo optimization model from energy system.
        
        Args:
            system: Energy system to optimize
            
        Returns:
            Pyomo ConcreteModel
        """
        logger.info("Building linear programming model")
        
        model = ConcreteModel(name="energy_system_optimization")
        
        # Sets
        model.T = Set(initialize=range(len(system.network.snapshots)), doc="Time periods")
        model.G = Set(initialize=list(system.generation_units.keys()), doc="Generators")
        model.L = Set(initialize=list(system.transmission_lines.keys()), doc="Transmission lines")
        model.S = Set(initialize=list(system.storage_systems.keys()), doc="Storage systems")
        model.N = Set(initialize=list(system.network.buses.index), doc="Nodes")
        model.D = Set(initialize=list(system.demand_profiles.keys()), doc="Demand profiles")
        
        # Parameters
        # Generation parameters
        model.p_max = Param(
            model.G,
            initialize={g: system.generation_units[g].capacity_mw for g in model.G},
            doc="Maximum generation capacity"
        )
        model.p_min = Param(
            model.G,
            initialize={g: system.generation_units[g].capacity_mw * system.generation_units[g].min_load_factor for g in model.G},
            doc="Minimum generation"
        )
        model.c_var = Param(
            model.G,
            initialize={g: system.generation_units[g].variable_cost_per_mwh for g in model.G},
            doc="Variable cost"
        )
        model.c_fixed = Param(
            model.G,
            initialize={g: system.generation_units[g].fixed_cost_per_mw_year / 8760 for g in model.G},  # Per hour
            doc="Fixed cost per hour"
        )
        model.emission_factor = Param(
            model.G,
            initialize={g: system.generation_units[g].emission_factor_kg_co2_per_mwh for g in model.G},
            doc="Emission factor"
        )
        
        # Transmission parameters
        model.f_max = Param(
            model.L,
            initialize={l: system.transmission_lines[l].capacity_mw for l in model.L},
            doc="Maximum transmission capacity"
        )
        model.transmission_loss = Param(
            model.L,
            initialize={l: system.transmission_lines[l].total_loss for l in model.L},
            doc="Transmission loss factor"
        )
        
        # Storage parameters
        model.e_max = Param(
            model.S,
            initialize={s: system.storage_systems[s].energy_capacity_mwh for s in model.S},
            doc="Maximum energy capacity"
        )
        model.p_charge_max = Param(
            model.S,
            initialize={s: system.storage_systems[s].power_capacity_mw for s in model.S},
            doc="Maximum charge power"
        )
        model.p_discharge_max = Param(
            model.S,
            initialize={s: system.storage_systems[s].power_capacity_mw for s in model.S},
            doc="Maximum discharge power"
        )
        model.eta_charge = Param(
            model.S,
            initialize={s: system.storage_systems[s].charge_efficiency for s in model.S},
            doc="Charge efficiency"
        )
        model.eta_discharge = Param(
            model.S,
            initialize={s: system.storage_systems[s].discharge_efficiency for s in model.S},
            doc="Discharge efficiency"
        )
        
        # Demand parameters
        demand_data = {}
        for d in model.D:
            profile = system.demand_profiles[d]
            demand_series = profile.generate_profile(system.network.snapshots)
            demand_data[d] = {t: demand_series.iloc[t] for t in model.T}
        
        model.demand = Param(
            model.D, model.T,
            initialize=lambda m, d, t: demand_data.get(d, {}).get(t, 0.0),
            doc="Demand at each node and time"
        )
        
        # Node mapping
        model.gen_node = Param(
            model.G,
            initialize={g: system.generation_units[g].node for g in model.G},
            doc="Generator node"
        )
        model.load_node = Param(
            model.D,
            initialize={d: system.demand_profiles[d].node for d in model.D},
            doc="Load node"
        )
        model.line_from = Param(
            model.L,
            initialize={l: system.transmission_lines[l].from_node for l in model.L},
            doc="Line from node"
        )
        model.line_to = Param(
            model.L,
            initialize={l: system.transmission_lines[l].to_node for l in model.L},
            doc="Line to node"
        )
        model.storage_node = Param(
            model.S,
            initialize={s: system.storage_systems[s].node for s in model.S},
            doc="Storage node"
        )
        
        # Variables
        model.p_gen = Var(model.G, model.T, domain=NonNegativeReals, doc="Generation")
        model.f_flow = Var(model.L, model.T, domain=Reals, doc="Transmission flow")
        model.e_storage = Var(model.S, model.T, domain=NonNegativeReals, doc="Storage energy")
        model.p_charge = Var(model.S, model.T, domain=NonNegativeReals, doc="Charge power")
        model.p_discharge = Var(model.S, model.T, domain=NonNegativeReals, doc="Discharge power")
        
        # Objective function
        if self.settings.optimization.objective == "cost":
            model.objective = Objective(
                expr=sum(
                    model.c_var[g] * model.p_gen[g, t] +
                    model.c_fixed[g] * model.p_max[g]
                    for g in model.G for t in model.T
                ),
                sense=minimize
            )
        elif self.settings.optimization.objective == "emissions":
            model.objective = Objective(
                expr=sum(
                    model.emission_factor[g] * model.p_gen[g, t]
                    for g in model.G for t in model.T
                ),
                sense=minimize
            )
        else:  # multi-objective (weighted)
            model.objective = Objective(
                expr=sum(
                    0.7 * (model.c_var[g] * model.p_gen[g, t] + model.c_fixed[g] * model.p_max[g]) +
                    0.3 * model.emission_factor[g] * model.p_gen[g, t]
                    for g in model.G for t in model.T
                ),
                sense=minimize
            )
        
        # Constraints
        # Generation limits
        def gen_max_rule(m, g, t):
            return m.p_gen[g, t] <= m.p_max[g]
        model.gen_max = Constraint(model.G, model.T, rule=gen_max_rule)
        
        def gen_min_rule(m, g, t):
            return m.p_gen[g, t] >= m.p_min[g]
        model.gen_min = Constraint(model.G, model.T, rule=gen_min_rule)
        
        # Transmission limits
        def flow_max_rule(m, l, t):
            return -m.f_max[l] <= m.f_flow[l, t] <= m.f_max[l]
        model.flow_max = Constraint(model.L, model.T, rule=flow_max_rule)
        
        # Storage constraints
        def storage_energy_max_rule(m, s, t):
            return m.e_storage[s, t] <= m.e_max[s]
        model.storage_energy_max = Constraint(model.S, model.T, rule=storage_energy_max_rule)
        
        def storage_charge_max_rule(m, s, t):
            return m.p_charge[s, t] <= m.p_charge_max[s]
        model.storage_charge_max = Constraint(model.S, model.T, rule=storage_charge_max_rule)
        
        def storage_discharge_max_rule(m, s, t):
            return m.p_discharge[s, t] <= m.p_discharge_max[s]
        model.storage_discharge_max = Constraint(model.S, model.T, rule=storage_discharge_max_rule)
        
        # Storage energy balance
        def storage_balance_rule(m, s, t):
            if t == 0:
                return m.e_storage[s, t] == 0.5 * m.e_max[s] + m.eta_charge[s] * m.p_charge[s, t] - m.p_discharge[s, t] / m.eta_discharge[s]
            else:
                return m.e_storage[s, t] == m.e_storage[s, t-1] + m.eta_charge[s] * m.p_charge[s, t] - m.p_discharge[s, t] / m.eta_discharge[s]
        model.storage_balance = Constraint(model.S, model.T, rule=storage_balance_rule)
        
        # Power balance at each node (simplified - would need proper node mapping)
        # This is a simplified version - full implementation would require proper node balance
        
        self.model = model
        logger.info("Model built successfully")
        
        return model
    
    def optimize(
        self,
        system: Optional[EnergySystem] = None,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Run optimization.
        
        Args:
            system: Energy system to optimize (if None, uses existing model)
            output_path: Path to save results
            
        Returns:
            Dictionary with optimization results
        """
        if system is not None:
            self.build_model(system)
        
        if self.model is None:
            raise ValueError("No model available. Provide system or build model first.")
        
        # Detect and configure solver
        solver_name = self._detect_solver()
        solver = SolverFactory(solver_name)
        
        # Configure solver options
        if self.settings.solver.timeout:
            solver.options["time_limit"] = self.settings.solver.timeout
        if self.settings.solver.threads:
            solver.options["threads"] = self.settings.solver.threads
        if solver_name == "gurobi":
            solver.options["MIPGap"] = self.settings.solver.mip_gap
        
        logger.info(f"Solving optimization problem with {solver_name}")
        
        # Solve
        try:
            results = solver.solve(self.model, tee=self.settings.solver.log_level > 1)
            
            # Extract results
            status = results.solver.status
            termination = results.solver.termination_condition
            
            if status == SolverStatus.ok and termination == TerminationCondition.optimal:
                objective_value = value(self.model.objective)
                
                self.results = {
                    "status": "optimal",
                    "objective": objective_value,
                    "solver": solver_name,
                    "variables": self._extract_variables(),
                }
                
                logger.info(f"Optimization successful. Objective: {objective_value:.2f}")
            else:
                self.results = {
                    "status": "not_optimal",
                    "solver_status": str(status),
                    "termination": str(termination),
                    "solver": solver_name,
                }
                logger.warning(f"Optimization did not reach optimal solution: {termination}")
        
        except Exception as e:
            logger.error(f"Error during optimization: {e}", exc_info=True)
            self.results = {
                "status": "error",
                "error": str(e),
                "solver": solver_name,
            }
        
        # Save results
        if output_path:
            self._save_results(output_path)
        
        return self.results
    
    def _extract_variables(self) -> Dict[str, Any]:
        """Extract variable values from solved model."""
        if self.model is None:
            return {}
        
        variables = {}
        
        # Extract generation
        if hasattr(self.model, "p_gen"):
            gen_data = {}
            for g in self.model.G:
                gen_data[g] = [value(self.model.p_gen[g, t]) for t in self.model.T]
            variables["generation"] = gen_data
        
        # Extract storage
        if hasattr(self.model, "e_storage"):
            storage_data = {}
            for s in self.model.S:
                storage_data[s] = [value(self.model.e_storage[s, t]) for t in self.model.T]
            variables["storage"] = storage_data
        
        return variables
    
    def _save_results(self, output_path: Path) -> None:
        """Save optimization results to file."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results_file = output_path / "optimization_results.json"
        
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_file}")

