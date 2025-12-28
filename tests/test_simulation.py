"""Tests for simulation modules."""

import pytest
from pathlib import Path

from src.config import Settings
from src.simulation.scenario_manager import ScenarioManager, Scenario
from src.simulation.simulator import Simulator


@pytest.fixture
def test_settings():
    """Create test settings."""
    return Settings()


@pytest.fixture
def temp_scenarios_dir(tmp_path):
    """Create temporary scenarios directory."""
    scenarios_dir = tmp_path / "scenarios"
    scenarios_dir.mkdir()
    return scenarios_dir


class TestScenarioManager:
    """Tests for ScenarioManager class."""
    
    def test_initialization(self, temp_scenarios_dir):
        """Test scenario manager initialization."""
        manager = ScenarioManager(temp_scenarios_dir)
        
        assert manager.scenarios_path == temp_scenarios_dir
    
    def test_create_default_scenarios(self, temp_scenarios_dir):
        """Test creating default scenarios."""
        manager = ScenarioManager(temp_scenarios_dir)
        manager.create_default_scenarios()
        
        scenarios = manager.list_scenarios()
        assert len(scenarios) >= 4
        assert "baseline" in scenarios
        assert "high_renewable" in scenarios
    
    def test_save_and_load_scenario(self, temp_scenarios_dir):
        """Test saving and loading scenarios."""
        manager = ScenarioManager(temp_scenarios_dir)
        
        scenario = Scenario(
            name="test_scenario",
            description="Test",
            year=2025,
            renewable_penetration=0.5,
        )
        
        manager.save_scenario(scenario)
        
        loaded = manager.load_scenario("test_scenario")
        
        assert loaded is not None
        assert loaded.name == "test_scenario"
        assert loaded.renewable_penetration == 0.5


class TestSimulator:
    """Tests for Simulator class."""
    
    def test_initialization(self, test_settings):
        """Test simulator initialization."""
        simulator = Simulator(test_settings)
        
        assert simulator.settings == test_settings
    
    def test_create_system_from_scenario(self, test_settings):
        """Test creating system from scenario."""
        simulator = Simulator(test_settings)
        
        scenario = Scenario(
            name="test",
            year=2025,
            renewable_penetration=0.8,
            storage_capacity_gw=10.0,
        )
        
        system = simulator._create_system_from_scenario(scenario)
        
        assert system is not None
        assert len(system.generation_units) > 0
        assert len(system.demand_profiles) > 0
    
    @pytest.mark.skip(reason="Requires solver and full system setup")
    def test_run_scenario(self, test_settings, tmp_path):
        """Test running a scenario (requires solver)."""
        simulator = Simulator(test_settings)
        
        scenario = Scenario(
            name="test",
            year=2025,
            renewable_penetration=0.5,
        )
        
        # This would require a solver
        # results = simulator.run_scenario(scenario, output_path=tmp_path)
        # assert results is not None

