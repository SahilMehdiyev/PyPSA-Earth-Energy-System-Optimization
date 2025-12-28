"""Storage system models for energy systems."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import pandas as pd


class StorageType(str, Enum):
    """Storage technology types."""
    
    BATTERY = "battery"
    PUMPED_HYDRO = "pumped_hydro"
    HYDROGEN = "hydrogen"
    COMPRESSED_AIR = "compressed_air"
    FLYWHEEL = "flywheel"


@dataclass
class StorageSystem:
    """Represents a storage system in the energy system."""
    
    name: str
    storage_type: StorageType
    energy_capacity_mwh: float
    power_capacity_mw: float
    charge_efficiency: float = 0.9
    discharge_efficiency: float = 0.9
    standing_loss_per_hour: float = 0.0
    min_state_of_charge: float = 0.0
    max_state_of_charge: float = 1.0
    capital_cost_per_mwh: float = 0.0
    capital_cost_per_mw: float = 0.0
    fixed_cost_per_mw_year: float = 0.0
    variable_cost_per_mwh: float = 0.0
    node: Optional[str] = None
    location: Optional[tuple[float, float]] = None
    degradation_rate_per_cycle: float = 0.0
    max_cycles: Optional[int] = None
    
    def __post_init__(self) -> None:
        """Validate storage system parameters."""
        if self.energy_capacity_mwh <= 0:
            raise ValueError("Energy capacity must be positive")
        if self.power_capacity_mw <= 0:
            raise ValueError("Power capacity must be positive")
        if not 0 <= self.charge_efficiency <= 1:
            raise ValueError("Charge efficiency must be between 0 and 1")
        if not 0 <= self.discharge_efficiency <= 1:
            raise ValueError("Discharge efficiency must be between 0 and 1")
        if not 0 <= self.standing_loss_per_hour <= 1:
            raise ValueError("Standing loss must be between 0 and 1")
        if not 0 <= self.min_state_of_charge <= self.max_state_of_charge <= 1:
            raise ValueError("State of charge limits must be valid")
        if self.power_capacity_mw > self.energy_capacity_mwh:
            # Warning: power capacity higher than energy capacity means < 1 hour storage
            pass  # This is valid for some storage types
    
    @property
    def round_trip_efficiency(self) -> float:
        """Calculate round-trip efficiency."""
        return self.charge_efficiency * self.discharge_efficiency
    
    @property
    def duration_hours(self) -> float:
        """Calculate storage duration at full power."""
        return self.energy_capacity_mwh / self.power_capacity_mw
    
    def calculate_state_of_charge(
        self,
        initial_soc: float,
        charge_power_mw: float,
        discharge_power_mw: float,
        timestep_hours: float = 1.0,
    ) -> float:
        """
        Calculate state of charge after charge/discharge operation.
        
        Args:
            initial_soc: Initial state of charge (0-1)
            charge_power_mw: Charging power in MW (positive)
            discharge_power_mw: Discharging power in MW (positive)
            timestep_hours: Time step duration in hours
            
        Returns:
            New state of charge (0-1)
        """
        if charge_power_mw < 0 or discharge_power_mw < 0:
            raise ValueError("Power values must be non-negative")
        
        # Clamp charge/discharge to capacity limits
        charge_power_mw = min(charge_power_mw, self.power_capacity_mw)
        discharge_power_mw = min(discharge_power_mw, self.power_capacity_mw)
        
        # Calculate energy change
        energy_charged = charge_power_mw * timestep_hours * self.charge_efficiency
        energy_discharged = discharge_power_mw * timestep_hours / self.discharge_efficiency
        
        # Apply standing losses
        current_energy = initial_soc * self.energy_capacity_mwh
        standing_loss = current_energy * self.standing_loss_per_hour * timestep_hours
        
        # Update energy
        new_energy = current_energy + energy_charged - energy_discharged - standing_loss
        
        # Convert to state of charge
        new_soc = new_energy / self.energy_capacity_mwh
        
        # Clamp to limits
        new_soc = max(self.min_state_of_charge, min(self.max_state_of_charge, new_soc))
        
        return new_soc
    
    def simulate_operation(
        self,
        time_index: pd.DatetimeIndex,
        charge_schedule: pd.Series,
        discharge_schedule: pd.Series,
        initial_soc: float = 0.5,
    ) -> pd.DataFrame:
        """
        Simulate storage operation over time.
        
        Args:
            time_index: Time index
            charge_schedule: Charging power schedule (MW)
            discharge_schedule: Discharging power schedule (MW)
            initial_soc: Initial state of charge
            
        Returns:
            DataFrame with columns: soc, energy_mwh, charge_mw, discharge_mw
        """
        if len(charge_schedule) != len(time_index) or len(discharge_schedule) != len(time_index):
            raise ValueError("Schedule length must match time index")
        
        # Calculate timestep (assume hourly if not clear)
        timestep_hours = (time_index[1] - time_index[0]).total_seconds() / 3600.0
        
        soc_values = []
        energy_values = []
        charge_values = []
        discharge_values = []
        
        current_soc = initial_soc
        
        for i, (charge, discharge) in enumerate(zip(charge_schedule, discharge_schedule)):
            current_soc = self.calculate_state_of_charge(
                current_soc, charge, discharge, timestep_hours
            )
            
            soc_values.append(current_soc)
            energy_values.append(current_soc * self.energy_capacity_mwh)
            charge_values.append(charge)
            discharge_values.append(discharge)
        
        return pd.DataFrame(
            {
                "soc": soc_values,
                "energy_mwh": energy_values,
                "charge_mw": charge_values,
                "discharge_mw": discharge_values,
            },
            index=time_index,
        )
    
    def calculate_annual_cost(self, energy_throughput_mwh: float) -> float:
        """
        Calculate total annual cost.
        
        Args:
            energy_throughput_mwh: Annual energy throughput in MWh
            
        Returns:
            Total annual cost in currency units
        """
        # Capital costs (annualized)
        energy_capital = self.capital_cost_per_mwh * self.energy_capacity_mwh
        power_capital = self.capital_cost_per_mw * self.power_capacity_mw
        total_capital = energy_capital + power_capital
        
        # Annualize (assuming 15-year lifetime, 5% discount rate)
        annualized_capital = total_capital * 0.0963
        
        # Fixed and variable costs
        fixed_cost = self.fixed_cost_per_mw_year * self.power_capacity_mw
        variable_cost = self.variable_cost_per_mwh * energy_throughput_mwh
        
        return annualized_capital + fixed_cost + variable_cost
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "storage_type": self.storage_type.value,
            "energy_capacity_mwh": self.energy_capacity_mwh,
            "power_capacity_mw": self.power_capacity_mw,
            "charge_efficiency": self.charge_efficiency,
            "discharge_efficiency": self.discharge_efficiency,
            "standing_loss_per_hour": self.standing_loss_per_hour,
            "min_state_of_charge": self.min_state_of_charge,
            "max_state_of_charge": self.max_state_of_charge,
            "capital_cost_per_mwh": self.capital_cost_per_mwh,
            "capital_cost_per_mw": self.capital_cost_per_mw,
            "fixed_cost_per_mw_year": self.fixed_cost_per_mw_year,
            "variable_cost_per_mwh": self.variable_cost_per_mwh,
            "node": self.node,
            "location": self.location,
            "degradation_rate_per_cycle": self.degradation_rate_per_cycle,
            "max_cycles": self.max_cycles,
            "round_trip_efficiency": self.round_trip_efficiency,
            "duration_hours": self.duration_hours,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "StorageSystem":
        """Create from dictionary."""
        storage_type = StorageType(data["storage_type"])
        return cls(
            name=data["name"],
            storage_type=storage_type,
            energy_capacity_mwh=data["energy_capacity_mwh"],
            power_capacity_mw=data["power_capacity_mw"],
            charge_efficiency=data.get("charge_efficiency", 0.9),
            discharge_efficiency=data.get("discharge_efficiency", 0.9),
            standing_loss_per_hour=data.get("standing_loss_per_hour", 0.0),
            min_state_of_charge=data.get("min_state_of_charge", 0.0),
            max_state_of_charge=data.get("max_state_of_charge", 1.0),
            capital_cost_per_mwh=data.get("capital_cost_per_mwh", 0.0),
            capital_cost_per_mw=data.get("capital_cost_per_mw", 0.0),
            fixed_cost_per_mw_year=data.get("fixed_cost_per_mw_year", 0.0),
            variable_cost_per_mwh=data.get("variable_cost_per_mwh", 0.0),
            node=data.get("node"),
            location=tuple(data["location"]) if data.get("location") else None,
            degradation_rate_per_cycle=data.get("degradation_rate_per_cycle", 0.0),
            max_cycles=data.get("max_cycles"),
        )

