"""Transmission line models for energy systems."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class LineType(str, Enum):
    """Transmission line types."""
    
    AC = "ac"
    DC = "dc"


@dataclass
class TransmissionLine:
    """Represents a transmission line in the energy system."""
    
    name: str
    from_node: str
    to_node: str
    capacity_mw: float
    line_type: LineType = LineType.AC
    length_km: Optional[float] = None
    resistance_per_km: float = 0.0
    reactance_per_km: float = 0.0
    loss_percent: float = 0.0
    capital_cost_per_mw_km: float = 0.0
    fixed_cost_per_mw_year: float = 0.0
    x: Optional[float] = None  # Reactance (p.u.)
    s_nom: Optional[float] = None  # Nominal apparent power (MVA)
    
    def __post_init__(self) -> None:
        """Validate transmission line parameters."""
        if self.capacity_mw <= 0:
            raise ValueError("Capacity must be positive")
        if self.from_node == self.to_node:
            raise ValueError("From and to nodes must be different")
        if not 0 <= self.loss_percent <= 100:
            raise ValueError("Loss percent must be between 0 and 100")
        if self.length_km is not None and self.length_km <= 0:
            raise ValueError("Length must be positive if provided")
        
        # Set default s_nom if not provided
        if self.s_nom is None:
            self.s_nom = self.capacity_mw
        
        # Calculate loss if length and resistance provided
        if self.length_km is not None and self.resistance_per_km > 0:
            # Approximate loss based on resistance
            # Loss = I^2 * R, assuming nominal voltage
            # Simplified: loss_percent ≈ (resistance_per_km * length_km * 100) / base_impedance
            if self.loss_percent == 0.0:
                # Rough estimate: 0.1% per 100km for AC, 0.05% for DC
                base_loss = 0.1 if self.line_type == LineType.AC else 0.05
                self.loss_percent = base_loss * (self.length_km / 100.0)
    
    @property
    def total_loss(self) -> float:
        """Calculate total transmission loss as fraction."""
        return self.loss_percent / 100.0
    
    def calculate_loss(self, power_flow_mw: float) -> float:
        """
        Calculate power loss for given flow.
        
        Args:
            power_flow_mw: Power flow in MW
            
        Returns:
            Power loss in MW
        """
        # Linear loss model (can be extended to quadratic)
        return abs(power_flow_mw) * self.total_loss
    
    def calculate_annual_cost(self) -> float:
        """
        Calculate total annual cost.
        
        Returns:
            Total annual cost in currency units
        """
        fixed_cost = self.fixed_cost_per_mw_year * self.capacity_mw
        
        if self.length_km is not None:
            capital_cost = self.capital_cost_per_mw_km * self.capacity_mw * self.length_km
            # Annualize capital cost (assuming 20-year lifetime, 5% discount rate)
            annualized_capital = capital_cost * 0.0802  # Capital recovery factor
            return fixed_cost + annualized_capital
        
        return fixed_cost
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "from_node": self.from_node,
            "to_node": self.to_node,
            "capacity_mw": self.capacity_mw,
            "line_type": self.line_type.value,
            "length_km": self.length_km,
            "resistance_per_km": self.resistance_per_km,
            "reactance_per_km": self.reactance_per_km,
            "loss_percent": self.loss_percent,
            "capital_cost_per_mw_km": self.capital_cost_per_mw_km,
            "fixed_cost_per_mw_year": self.fixed_cost_per_mw_year,
            "x": self.x,
            "s_nom": self.s_nom,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "TransmissionLine":
        """Create from dictionary."""
        line_type = LineType(data.get("line_type", "ac"))
        return cls(
            name=data["name"],
            from_node=data["from_node"],
            to_node=data["to_node"],
            capacity_mw=data["capacity_mw"],
            line_type=line_type,
            length_km=data.get("length_km"),
            resistance_per_km=data.get("resistance_per_km", 0.0),
            reactance_per_km=data.get("reactance_per_km", 0.0),
            loss_percent=data.get("loss_percent", 0.0),
            capital_cost_per_mw_km=data.get("capital_cost_per_mw_km", 0.0),
            fixed_cost_per_mw_year=data.get("fixed_cost_per_mw_year", 0.0),
            x=data.get("x"),
            s_nom=data.get("s_nom"),
        )

