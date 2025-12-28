"""Genetic algorithm optimization for energy systems."""

import json
import random
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging

import pandas as pd
import numpy as np

try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False
    logging.warning("DEAP not available. Genetic algorithm will use simplified implementation.")

from src.config import Settings
from src.models.energy_system import EnergySystem


logger = logging.getLogger(__name__)


class GeneticAlgorithmOptimizer:
    """Genetic algorithm optimizer for energy systems."""
    
    def __init__(self, settings: Settings) -> None:
        """
        Initialize GA optimizer.
        
        Args:
            settings: Configuration settings
        """
        self.settings = settings
        self.population: List[Dict[str, Any]] = []
        self.history: List[Dict[str, Any]] = []
        self.best_individual: Optional[Dict[str, Any]] = None
    
    def _initialize_population(
        self,
        system: EnergySystem,
        population_size: int,
    ) -> List[Dict[str, Any]]:
        """
        Initialize random population.
        
        Args:
            system: Energy system
            population_size: Size of population
            
        Returns:
            List of individuals (each is a dict with generation/storage capacities)
        """
        population = []
        
        for _ in range(population_size):
            individual = {}
            
            # Random generation capacities (normalized 0-1)
            for gen_name, gen_unit in system.generation_units.items():
                # Random capacity factor (0.5 to 1.5 of base capacity)
                individual[f"gen_{gen_name}"] = random.uniform(0.5, 1.5)
            
            # Random storage capacities
            for storage_name, storage in system.storage_systems.items():
                individual[f"storage_energy_{storage_name}"] = random.uniform(0.5, 2.0)
                individual[f"storage_power_{storage_name}"] = random.uniform(0.5, 2.0)
            
            population.append(individual)
        
        return population
    
    def _evaluate_fitness(
        self,
        individual: Dict[str, Any],
        system: EnergySystem,
    ) -> Tuple[float, float, float]:
        """
        Evaluate fitness of an individual.
        
        Args:
            individual: Individual to evaluate
            system: Energy system
            
        Returns:
            Tuple of (cost, emissions, renewable_penetration)
            Lower is better for cost and emissions, higher is better for renewable
        """
        # Create modified system based on individual
        modified_system = self._apply_individual_to_system(individual, system)
        
        # Run simulation (simplified - would need full PyPSA solve)
        try:
            # Calculate approximate metrics
            total_cost = 0.0
            total_emissions = 0.0
            total_generation = 0.0
            renewable_generation = 0.0
            
            for gen_name, gen_unit in modified_system.generation_units.items():
                # Simplified: assume capacity factor of 0.5
                annual_gen = gen_unit.capacity_mw * 0.5 * 8760
                total_generation += annual_gen
                
                cost = gen_unit.calculate_annual_cost(annual_gen)
                emissions = gen_unit.calculate_emissions(annual_gen)
                
                total_cost += cost
                total_emissions += emissions
                
                if gen_unit.is_renewable:
                    renewable_generation += annual_gen
            
            renewable_penetration = renewable_generation / total_generation if total_generation > 0 else 0.0
            
            # Multi-objective: return all three objectives
            return (total_cost, total_emissions, -renewable_penetration)  # Negative for minimization
        
        except Exception as e:
            logger.warning(f"Error evaluating individual: {e}")
            # Return poor fitness
            return (1e10, 1e10, 1.0)
    
    def _apply_individual_to_system(
        self,
        individual: Dict[str, Any],
        system: EnergySystem,
    ) -> EnergySystem:
        """Apply individual's parameters to create modified system."""
        # Create copy (simplified - would need proper deep copy)
        modified_system = system
        
        # Modify generation capacities
        for gen_name, gen_unit in modified_system.generation_units.items():
            key = f"gen_{gen_name}"
            if key in individual:
                gen_unit.capacity_mw *= individual[key]
        
        # Modify storage capacities
        for storage_name, storage in modified_system.storage_systems.items():
            energy_key = f"storage_energy_{storage_name}"
            power_key = f"storage_power_{storage_name}"
            if energy_key in individual:
                storage.energy_capacity_mwh *= individual[energy_key]
            if power_key in individual:
                storage.power_capacity_mw *= individual[power_key]
        
        return modified_system
    
    def _crossover(
        self,
        parent1: Dict[str, Any],
        parent2: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Perform crossover between two parents."""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Single-point crossover
        keys = list(parent1.keys())
        if len(keys) > 1:
            crossover_point = random.randint(1, len(keys) - 1)
            
            for key in keys[crossover_point:]:
                child1[key], child2[key] = child2[key], child1[key]
        
        return child1, child2
    
    def _mutate(
        self,
        individual: Dict[str, Any],
        mutation_rate: float,
    ) -> Dict[str, Any]:
        """Mutate an individual."""
        mutated = individual.copy()
        
        for key in mutated.keys():
            if random.random() < mutation_rate:
                # Gaussian mutation
                mutated[key] = max(0.1, mutated[key] + random.gauss(0, 0.1))
        
        return mutated
    
    def _select_parents(
        self,
        population: List[Dict[str, Any]],
        fitness_scores: List[Tuple[float, float, float]],
        tournament_size: int = 3,
    ) -> Dict[str, Any]:
        """Tournament selection."""
        tournament = random.sample(
            list(zip(population, fitness_scores)),
            min(tournament_size, len(population))
        )
        
        # Select best (lowest cost)
        tournament.sort(key=lambda x: x[1][0])  # Sort by cost
        return tournament[0][0]
    
    def optimize(
        self,
        system: EnergySystem,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Run genetic algorithm optimization.
        
        Args:
            system: Energy system to optimize
            output_path: Path to save results
            
        Returns:
            Dictionary with optimization results
        """
        logger.info("Starting genetic algorithm optimization")
        
        population_size = self.settings.optimization.ga_population_size
        generations = self.settings.optimization.ga_generations
        mutation_rate = self.settings.optimization.ga_mutation_rate
        crossover_rate = self.settings.optimization.ga_crossover_rate
        elite_size = self.settings.optimization.ga_elite_size
        
        # Initialize population
        self.population = self._initialize_population(system, population_size)
        
        # Evaluate initial population
        fitness_scores = [
            self._evaluate_fitness(ind, system) for ind in self.population
        ]
        
        # Track best
        best_idx = min(range(len(fitness_scores)), key=lambda i: fitness_scores[i][0])
        self.best_individual = self.population[best_idx].copy()
        best_fitness = fitness_scores[best_idx]
        
        logger.info(f"Initial best cost: {best_fitness[0]:.2f}, emissions: {best_fitness[1]:.2f}")
        
        # Evolution loop
        for generation in range(generations):
            # Create new population
            new_population = []
            
            # Elitism: keep best individuals
            elite_indices = sorted(
                range(len(fitness_scores)),
                key=lambda i: fitness_scores[i][0]
            )[:elite_size]
            
            for idx in elite_indices:
                new_population.append(self.population[idx].copy())
            
            # Generate offspring
            while len(new_population) < population_size:
                # Selection
                parent1 = self._select_parents(self.population, fitness_scores)
                parent2 = self._select_parents(self.population, fitness_scores)
                
                # Crossover
                if random.random() < crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation
                child1 = self._mutate(child1, mutation_rate)
                child2 = self._mutate(child2, mutation_rate)
                
                new_population.extend([child1, child2])
            
            # Trim to population size
            new_population = new_population[:population_size]
            
            # Evaluate new population
            new_fitness = [
                self._evaluate_fitness(ind, system) for ind in new_population
            ]
            
            # Update best
            best_idx = min(range(len(new_fitness)), key=lambda i: new_fitness[i][0])
            if new_fitness[best_idx][0] < best_fitness[0]:
                self.best_individual = new_population[best_idx].copy()
                best_fitness = new_fitness[best_idx]
                logger.info(
                    f"Generation {generation+1}: New best cost: {best_fitness[0]:.2f}, "
                    f"emissions: {best_fitness[1]:.2f}"
                )
            
            # Update population
            self.population = new_population
            fitness_scores = new_fitness
            
            # Record history
            self.history.append({
                "generation": generation + 1,
                "best_cost": best_fitness[0],
                "best_emissions": best_fitness[1],
                "avg_cost": np.mean([f[0] for f in fitness_scores]),
            })
        
        # Prepare results
        results = {
            "status": "completed",
            "method": "genetic_algorithm",
            "generations": generations,
            "population_size": population_size,
            "best_individual": self.best_individual,
            "best_fitness": {
                "cost": best_fitness[0],
                "emissions": best_fitness[1],
                "renewable_penetration": -best_fitness[2],
            },
            "history": self.history,
        }
        
        # Save results
        if output_path:
            self._save_results(output_path, results)
        
        logger.info("Genetic algorithm optimization completed")
        
        return results
    
    def _save_results(
        self,
        output_path: Path,
        results: Dict[str, Any],
    ) -> None:
        """Save optimization results."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results_file = output_path / "ga_results.json"
        
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save convergence history
        if self.history:
            history_df = pd.DataFrame(self.history)
            history_file = output_path / "convergence_history.csv"
            history_df.to_csv(history_file, index=False)
        
        logger.info(f"Results saved to {output_path}")

