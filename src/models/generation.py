"""Generation unit models for energy systems."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np
import pandas as pd


class TechnologyType(str, Enum):
    """Generation technology types."""
    
    SOLAR = "solar"
    WIND_ONSHORE = "wind_onshore"
    WIND_OFFSHORE = "wind_offshore"
    HYDRO = "hydro"
    NUCLEAR = "nuclear"
    COAL = "coal"
    GAS = "gas"
    OIL = "oil"
    BIOMASS = "biomass"
    GEOTHERMAL = "geothermal"


@dataclass
class GenerationUnit:
    """Represents a generation unit in the energy system."""
    
    name: str
    technology: TechnologyType
    capacity_mw: float
    efficiency: float = 1.0
    variable_cost_per_mwh: float = 0.0
    fixed_cost_per_mw_year: float = 0.0
    emission_factor_kg_co2_per_mwh: float = 0.0
    min_load_factor: float = 0.0
    max_load_factor: float = 1.0
    ramp_rate_mw_per_hour: Optional[float] = None
    location: Optional[tuple[float, float]] = None  # (lat, lon)
    node: Optional[str] = None
    availability_profile: Optional[pd.Series] = None
    fuel_type: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validate generation unit parameters."""
        if self.capacity_mw <= 0:
            raise ValueError("Capacity must be positive")
        if not 0 <= self.efficiency <= 1:
            raise ValueError("Efficiency must be between 0 and 1")
        if not 0 <= self.min_load_factor <= self.max_load_factor <= 1:
            raise ValueError("Load factors must be valid (0 <= min <= max <= 1)")
        if self.variable_cost_per_mwh < 0:
            raise ValueError("Variable cost cannot be negative")
        if self.fixed_cost_per_mw_year < 0:
            raise ValueError("Fixed cost cannot be negative")
        if self.emission_factor_kg_co2_per_mwh < 0:
            raise ValueError("Emission factor cannot be negative")
    
    @property
    def is_renewable(self) -> bool:
        """Check if generation unit is renewable."""
        renewable_techs = {
            TechnologyType.SOLAR,
            TechnologyType.WIND_ONSHORE,
            TechnologyType.WIND_OFFSHORE,
            TechnologyType.HYDRO,
            TechnologyType.BIOMASS,
            TechnologyType.GEOTHERMAL,
        }
        return self.technology in renewable_techs
    
    @property
    def is_dispatchable(self) -> bool:
        """Check if generation unit is dispatchable."""
        non_dispatchable = {
            TechnologyType.SOLAR,
            TechnologyType.WIND_ONSHORE,
            TechnologyType.WIND_OFFSHORE,
        }
        return self.technology not in non_dispatchable
    
    def calculate_generation_profile(
        self,
        time_index: pd.DatetimeIndex,
        capacity_factor: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Calculate generation profile for given time period.
        
        Args:
            time_index: Time index for the profile
            capacity_factor: Optional capacity factor time series (0-1)
            
        Returns:
            Generation profile in MW
        """
        if capacity_factor is not None:
            if len(capacity_factor) != len(time_index):
                raise ValueError("Capacity factor length must match time index")
            profile = self.capacity_mw * capacity_factor * self.efficiency
        else:
            # Default: full capacity if dispatchable, otherwise use availability
            if self.is_dispatchable:
                profile = pd.Series(
                    self.capacity_mw * self.max_load_factor,
                    index=time_index,
                )
            elif self.availability_profile is not None:
                # Interpolate availability profile to match time index
                profile = self.availability_profile.reindex(
                    time_index, method="nearest"
                ).fillna(0) * self.capacity_mw
            else:
                # Default: assume 0 for non-dispatchable without profile
                profile = pd.Series(0.0, index=time_index)
        
        # Apply min/max load factor constraints
        min_gen = self.capacity_mw * self.min_load_factor
        max_gen = self.capacity_mw * self.max_load_factor
        profile = profile.clip(lower=min_gen, upper=max_gen)
        
        return profile
    
    def calculate_annual_cost(self, generation_mwh: float) -> float:
        """
        Calculate total annual cost.
        
        Args:
            generation_mwh: Annual generation in MWh
            
        Returns:
            Total annual cost in currency units
        """
        fixed_cost = self.fixed_cost_per_mw_year * self.capacity_mw
        variable_cost = self.variable_cost_per_mwh * generation_mwh
        return fixed_cost + variable_cost
    
    def calculate_emissions(self, generation_mwh: float) -> float:
        """
        Calculate CO2 emissions.
        
        Args:
            generation_mwh: Generation in MWh
            
        Returns:
            CO2 emissions in kg
        """
        return self.emission_factor_kg_co2_per_mwh * generation_mwh
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "technology": self.technology.value,
            "capacity_mw": self.capacity_mw,
            "efficiency": self.efficiency,
            "variable_cost_per_mwh": self.variable_cost_per_mwh,
            "fixed_cost_per_mw_year": self.fixed_cost_per_mw_year,
            "emission_factor_kg_co2_per_mwh": self.emission_factor_kg_co2_per_mwh,
            "min_load_factor": self.min_load_factor,
            "max_load_factor": self.max_load_factor,
            "ramp_rate_mw_per_hour": self.ramp_rate_mw_per_hour,
            "location": self.location,
            "node": self.node,
            "fuel_type": self.fuel_type,
            "is_renewable": self.is_renewable,
            "is_dispatchable": self.is_dispatchable,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "GenerationUnit":
        """Create from dictionary."""
        tech = TechnologyType(data["technology"])
        return cls(
            name=data["name"],
            technology=tech,
            capacity_mw=data["capacity_mw"],
            efficiency=data.get("efficiency", 1.0),
            variable_cost_per_mwh=data.get("variable_cost_per_mwh", 0.0),
            fixed_cost_per_mw_year=data.get("fixed_cost_per_mw_year", 0.0),
            emission_factor_kg_co2_per_mwh=data.get("emission_factor_kg_co2_per_mwh", 0.0),
            min_load_factor=data.get("min_load_factor", 0.0),
            max_load_factor=data.get("max_load_factor", 1.0),
            ramp_rate_mw_per_hour=data.get("ramp_rate_mw_per_hour"),
            location=tuple(data["location"]) if data.get("location") else None,
            node=data.get("node"),
            fuel_type=data.get("fuel_type"),
        )

