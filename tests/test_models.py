"""Tests for energy system models."""

import pytest
import pandas as pd
import numpy as np

from src.models.generation import GenerationUnit, TechnologyType
from src.models.transmission import TransmissionLine, LineType
from src.models.storage import StorageSystem, StorageType
from src.models.demand import DemandProfile
from src.models.energy_system import EnergySystem


class TestGenerationUnit:
    """Tests for GenerationUnit class."""
    
    def test_create_solar_unit(self):
        """Test creating a solar generation unit."""
        unit = GenerationUnit(
            name="solar_farm",
            technology=TechnologyType.SOLAR,
            capacity_mw=100.0,
            efficiency=0.20,
        )
        
        assert unit.name == "solar_farm"
        assert unit.technology == TechnologyType.SOLAR
        assert unit.capacity_mw == 100.0
        assert unit.is_renewable is True
        assert unit.is_dispatchable is False
    
    def test_create_gas_unit(self):
        """Test creating a gas generation unit."""
        unit = GenerationUnit(
            name="gas_plant",
            technology=TechnologyType.GAS,
            capacity_mw=500.0,
            efficiency=0.55,
            variable_cost_per_mwh=50.0,
            emission_factor_kg_co2_per_mwh=350.0,
        )
        
        assert unit.is_renewable is False
        assert unit.is_dispatchable is True
    
    def test_validation_errors(self):
        """Test validation raises errors for invalid inputs."""
        with pytest.raises(ValueError):
            GenerationUnit(
                name="invalid",
                technology=TechnologyType.SOLAR,
                capacity_mw=-100.0,  # Negative capacity
            )
        
        with pytest.raises(ValueError):
            GenerationUnit(
                name="invalid",
                technology=TechnologyType.SOLAR,
                capacity_mw=100.0,
                efficiency=1.5,  # Efficiency > 1
            )
    
    def test_calculate_annual_cost(self):
        """Test annual cost calculation."""
        unit = GenerationUnit(
            name="test",
            technology=TechnologyType.GAS,
            capacity_mw=100.0,
            fixed_cost_per_mw_year=30000.0,
            variable_cost_per_mwh=50.0,
        )
        
        annual_gen = 100.0 * 0.5 * 8760  # 50% capacity factor
        cost = unit.calculate_annual_cost(annual_gen)
        
        assert cost > 0
        assert cost == 30000.0 * 100.0 + 50.0 * annual_gen


class TestTransmissionLine:
    """Tests for TransmissionLine class."""
    
    def test_create_ac_line(self):
        """Test creating an AC transmission line."""
        line = TransmissionLine(
            name="line_1",
            from_node="node_a",
            to_node="node_b",
            capacity_mw=1000.0,
            line_type=LineType.AC,
            length_km=500.0,
        )
        
        assert line.name == "line_1"
        assert line.capacity_mw == 1000.0
        assert line.line_type == LineType.AC
    
    def test_calculate_loss(self):
        """Test loss calculation."""
        line = TransmissionLine(
            name="test",
            from_node="a",
            to_node="b",
            capacity_mw=1000.0,
            loss_percent=5.0,
        )
        
        loss = line.calculate_loss(500.0)
        assert loss == 500.0 * 0.05


class TestStorageSystem:
    """Tests for StorageSystem class."""
    
    def test_create_battery(self):
        """Test creating a battery storage system."""
        storage = StorageSystem(
            name="battery",
            storage_type=StorageType.BATTERY,
            energy_capacity_mwh=1000.0,
            power_capacity_mw=250.0,
        )
        
        assert storage.duration_hours == 4.0
        assert storage.round_trip_efficiency == 0.81  # 0.9 * 0.9
    
    def test_state_of_charge_calculation(self):
        """Test state of charge calculation."""
        storage = StorageSystem(
            name="test",
            storage_type=StorageType.BATTERY,
            energy_capacity_mwh=100.0,
            power_capacity_mw=50.0,
            charge_efficiency=0.9,
            discharge_efficiency=0.9,
        )
        
        # Charge for 1 hour
        new_soc = storage.calculate_state_of_charge(
            initial_soc=0.5,
            charge_power_mw=50.0,
            discharge_power_mw=0.0,
            timestep_hours=1.0,
        )
        
        assert new_soc > 0.5
        assert new_soc <= 1.0


class TestDemandProfile:
    """Tests for DemandProfile class."""
    
    def test_create_demand_profile(self):
        """Test creating a demand profile."""
        profile = DemandProfile(
            name="demand",
            node="node_1",
            base_demand_mw=1000.0,
        )
        
        assert profile.base_demand_mw == 1000.0
    
    def test_generate_profile(self):
        """Test profile generation."""
        profile = DemandProfile(
            name="test",
            node="node_1",
            base_demand_mw=1000.0,
        )
        
        time_index = pd.date_range("2025-01-01", periods=100, freq="1H")
        demand_series = profile.generate_profile(time_index)
        
        assert len(demand_series) == 100
        assert all(demand_series > 0)


class TestEnergySystem:
    """Tests for EnergySystem class."""
    
    def test_create_system(self):
        """Test creating an energy system."""
        system = EnergySystem(name="test_system")
        
        assert system.name == "test_system"
        assert len(system.network.snapshots) > 0
    
    def test_add_components(self):
        """Test adding components to system."""
        system = EnergySystem(name="test")
        
        # Add generation
        solar = GenerationUnit(
            name="solar",
            technology=TechnologyType.SOLAR,
            capacity_mw=100.0,
        )
        system.add_generation_unit(solar)
        
        assert "solar" in system.generation_units
        
        # Add demand
        demand = DemandProfile(
            name="demand",
            node="node_1",
            base_demand_mw=50.0,
        )
        system.add_demand_profile(demand)
        
        assert "demand" in system.demand_profiles
    
    def test_validation(self):
        """Test system validation."""
        system = EnergySystem(name="test")
        
        # Add generation
        solar = GenerationUnit(
            name="solar",
            technology=TechnologyType.SOLAR,
            capacity_mw=100.0,
        )
        system.add_generation_unit(solar)
        
        # Add demand
        demand = DemandProfile(
            name="demand",
            node="node_1",
            base_demand_mw=50.0,
        )
        system.add_demand_profile(demand)
        
        is_valid, errors = system.validate()
        # Should be valid with generation > demand
        assert is_valid or len(errors) == 0

