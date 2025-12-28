"""Demand profile models for energy systems."""

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class DemandProfile:
    """Represents electricity demand profile."""
    
    name: str
    node: str
    base_demand_mw: float
    demand_profile: Optional[pd.Series] = None
    seasonal_factor: float = 1.0
    daily_pattern: Optional[pd.Series] = None  # Hourly pattern (0-23)
    weekly_pattern: Optional[pd.Series] = None  # Daily pattern (0-6, Mon-Sun)
    elasticity: float = 0.0  # Price elasticity of demand
    location: Optional[tuple[float, float]] = None
    
    def __post_init__(self) -> None:
        """Validate demand profile parameters."""
        if self.base_demand_mw <= 0:
            raise ValueError("Base demand must be positive")
        if not 0 <= self.elasticity <= 1:
            raise ValueError("Elasticity must be between 0 and 1")
        
        # Validate daily pattern
        if self.daily_pattern is not None:
            if len(self.daily_pattern) != 24:
                raise ValueError("Daily pattern must have 24 values (one per hour)")
            if not all(0 <= v <= 2 for v in self.daily_pattern):
                raise ValueError("Daily pattern values should be between 0 and 2")
        
        # Validate weekly pattern
        if self.weekly_pattern is not None:
            if len(self.weekly_pattern) != 7:
                raise ValueError("Weekly pattern must have 7 values (one per day)")
            if not all(0 <= v <= 2 for v in self.weekly_pattern):
                raise ValueError("Weekly pattern values should be between 0 and 2")
    
    def generate_profile(
        self,
        time_index: pd.DatetimeIndex,
        use_existing: bool = True,
    ) -> pd.Series:
        """
        Generate demand profile for given time period.
        
        Args:
            time_index: Time index for the profile
            use_existing: If True and demand_profile exists, use it (with interpolation)
            
        Returns:
            Demand profile in MW
        """
        if use_existing and self.demand_profile is not None:
            # Interpolate existing profile
            profile = self.demand_profile.reindex(
                time_index, method="nearest"
            ).fillna(method="ffill").fillna(method="bfill")
            return profile
        
        # Generate synthetic profile
        profile = pd.Series(index=time_index, dtype=float)
        
        for timestamp in time_index:
            demand = self.base_demand_mw
            
            # Apply seasonal factor (month-based)
            month = timestamp.month
            # Simple seasonal variation: higher in winter (Dec-Feb) and summer (Jun-Aug)
            if month in [12, 1, 2, 6, 7, 8]:
                seasonal_mult = 1.1
            else:
                seasonal_mult = 0.95
            demand *= seasonal_mult * self.seasonal_factor
            
            # Apply daily pattern
            if self.daily_pattern is not None:
                hour = timestamp.hour
                demand *= self.daily_pattern.iloc[hour]
            
            # Apply weekly pattern
            if self.weekly_pattern is not None:
                day_of_week = timestamp.weekday()
                demand *= self.weekly_pattern.iloc[day_of_week]
            
            profile[timestamp] = demand
        
        return profile
    
    def apply_price_elasticity(
        self,
        demand_profile: pd.Series,
        price_profile: pd.Series,
        base_price: float = 50.0,
    ) -> pd.Series:
        """
        Apply price elasticity to demand profile.
        
        Args:
            demand_profile: Base demand profile
            price_profile: Electricity price profile
            base_price: Base price for elasticity calculation
            
        Returns:
            Adjusted demand profile
        """
        if self.elasticity == 0.0:
            return demand_profile
        
        # Simple elasticity model: demand_change = -elasticity * price_change
        price_change = (price_profile - base_price) / base_price
        demand_change = -self.elasticity * price_change
        
        adjusted_demand = demand_profile * (1 + demand_change)
        
        # Ensure non-negative
        adjusted_demand = adjusted_demand.clip(lower=0)
        
        return adjusted_demand
    
    def calculate_annual_demand(self, demand_profile: Optional[pd.Series] = None) -> float:
        """
        Calculate total annual demand.
        
        Args:
            demand_profile: Optional demand profile (if None, uses stored profile)
            
        Returns:
            Annual demand in MWh
        """
        if demand_profile is None:
            if self.demand_profile is None:
                # Estimate from base demand
                return self.base_demand_mw * 8760  # hours per year
            demand_profile = self.demand_profile
        
        # Calculate timestep (assume hourly if not clear)
        if len(demand_profile) > 1:
            timestep_hours = (
                demand_profile.index[1] - demand_profile.index[0]
            ).total_seconds() / 3600.0
        else:
            timestep_hours = 1.0
        
        return demand_profile.sum() * timestep_hours
    
    def get_peak_demand(
        self,
        demand_profile: Optional[pd.Series] = None,
    ) -> tuple[float, pd.Timestamp]:
        """
        Get peak demand and timestamp.
        
        Args:
            demand_profile: Optional demand profile
            
        Returns:
            Tuple of (peak_demand_mw, timestamp)
        """
        if demand_profile is None:
            if self.demand_profile is None:
                return self.base_demand_mw, pd.Timestamp.now()
            demand_profile = self.demand_profile
        
        peak_idx = demand_profile.idxmax()
        peak_value = demand_profile.loc[peak_idx]
        
        return peak_value, peak_idx
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "node": self.node,
            "base_demand_mw": self.base_demand_mw,
            "seasonal_factor": self.seasonal_factor,
            "elasticity": self.elasticity,
            "location": self.location,
            "has_demand_profile": self.demand_profile is not None,
            "has_daily_pattern": self.daily_pattern is not None,
            "has_weekly_pattern": self.weekly_pattern is not None,
        }
    
    @classmethod
    def from_dict(cls, data: dict, demand_profile: Optional[pd.Series] = None) -> "DemandProfile":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            node=data["node"],
            base_demand_mw=data["base_demand_mw"],
            demand_profile=demand_profile,
            seasonal_factor=data.get("seasonal_factor", 1.0),
            daily_pattern=pd.Series(data["daily_pattern"]) if data.get("daily_pattern") else None,
            weekly_pattern=pd.Series(data["weekly_pattern"]) if data.get("weekly_pattern") else None,
            elasticity=data.get("elasticity", 0.0),
            location=tuple(data["location"]) if data.get("location") else None,
        )

