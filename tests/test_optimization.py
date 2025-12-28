"""Tests for optimization modules."""

import pytest
from pathlib import Path
import pandas as pd

from src.config import Settings
from src.models.energy_system import EnergySystem
from src.models.generation import GenerationUnit, TechnologyType
from src.models.demand import DemandProfile
from src.optimization.linear_programming import LinearProgrammingOptimizer
from src.optimization.genetic_algorithm import GeneticAlgorithmOptimizer


@pytest.fixture
def test_system():
    """Create a minimal test energy system."""
    system = EnergySystem(name="test", snapshots=pd.date_range("2025-01-01", periods=10, freq="1H"))
    
    # Add generation
    solar = GenerationUnit(
        name="solar",
        technology=TechnologyType.SOLAR,
        capacity_mw=100.0,
        variable_cost_per_mwh=0.0,
    )
    system.add_generation_unit(solar)
    
    gas = GenerationUnit(
        name="gas",
        technology=TechnologyType.GAS,
        capacity_mw=50.0,
        variable_cost_per_mwh=50.0,
    )
    system.add_generation_unit(gas)
    
    # Add demand
    demand = DemandProfile(
        name="demand",
        node="node_1",
        base_demand_mw=75.0,
    )
    system.add_demand_profile(demand)
    
    return system


@pytest.fixture
def test_settings():
    """Create test settings."""
    return Settings()


class TestLinearProgrammingOptimizer:
    """Tests for linear programming optimizer."""
    
    def test_initialization(self, test_settings):
        """Test optimizer initialization."""
        optimizer = LinearProgrammingOptimizer(test_settings)
        
        assert optimizer.settings == test_settings
        assert optimizer.model is None
    
    def test_build_model(self, test_settings, test_system):
        """Test model building."""
        optimizer = LinearProgrammingOptimizer(test_settings)
        
        model = optimizer.build_model(test_system)
        
        assert model is not None
        assert hasattr(model, "G")  # Generators set
        assert hasattr(model, "T")  # Time periods set
    
    @pytest.mark.skip(reason="Requires solver installation")
    def test_optimize(self, test_settings, test_system, tmp_path):
        """Test optimization (requires solver)."""
        optimizer = LinearProgrammingOptimizer(test_settings)
        
        results = optimizer.optimize(system=test_system, output_path=tmp_path)
        
        assert results is not None
        assert "status" in results


class TestGeneticAlgorithmOptimizer:
    """Tests for genetic algorithm optimizer."""
    
    def test_initialization(self, test_settings):
        """Test optimizer initialization."""
        optimizer = GeneticAlgorithmOptimizer(test_settings)
        
        assert optimizer.settings == test_settings
        assert len(optimizer.population) == 0
    
    def test_initialize_population(self, test_settings, test_system):
        """Test population initialization."""
        optimizer = GeneticAlgorithmOptimizer(test_settings)
        
        population = optimizer._initialize_population(test_system, population_size=10)
        
        assert len(population) == 10
        assert all(isinstance(ind, dict) for ind in population)
    
    @pytest.mark.skip(reason="Long running test")
    def test_optimize(self, test_settings, test_system, tmp_path):
        """Test GA optimization."""
        optimizer = GeneticAlgorithmOptimizer(test_settings)
        
        # Use smaller population and generations for testing
        test_settings.optimization.ga_population_size = 10
        test_settings.optimization.ga_generations = 5
        
        results = optimizer.optimize(system=test_system, output_path=tmp_path)
        
        assert results is not None
        assert "status" in results
        assert "best_individual" in results

