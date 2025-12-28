"""Plotting utilities for energy system results."""

from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.config import Settings


logger = logging.getLogger(__name__)


class PlotGenerator:
    """Generate plots for energy system analysis."""
    
    def __init__(self, settings: Settings) -> None:
        """
        Initialize plot generator.
        
        Args:
            settings: Configuration settings
        """
        self.settings = settings
        try:
            plt.style.use("seaborn-v0_8-darkgrid")
        except OSError:
            # Fallback to default style if seaborn not available
            plt.style.use("default")
    
    def generate_all_plots(
        self,
        results_path: Path,
        output_path: Path,
    ) -> None:
        """
        Generate all standard plots.
        
        Args:
            results_path: Path to simulation results
            output_path: Path to save plots
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating plots from {results_path} to {output_path}")
        
        # Load results
        results = self._load_results(results_path)
        
        if not results:
            logger.warning("No results found to plot")
            return
        
        # Generate plots
        self.plot_generation_stack(results, output_path)
        self.plot_demand_vs_generation(results, output_path)
        self.plot_storage_operation(results, output_path)
        self.plot_cost_breakdown(results, output_path)
        self.plot_emissions_comparison(results, output_path)
        
        logger.info(f"Plots generated in {output_path}")
    
    def _load_results(self, results_path: Path) -> Optional[Dict[str, Any]]:
        """Load results from JSON file."""
        import json
        
        # Find results file
        results_file = None
        for file in Path(results_path).glob("*_results.json"):
            results_file = file
            break
        
        if not results_file or not results_file.exists():
            return None
        
        with open(results_file, "r") as f:
            return json.load(f)
    
    def plot_generation_stack(
        self,
        results: Dict[str, Any],
        output_path: Path,
        filename: str = "generation_stack.png",
    ) -> None:
        """Generate generation stack plot."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create sample time series (would use actual data in real implementation)
        time_index = pd.date_range("2025-01-01", periods=8760, freq="1H")
        
        # Sample data
        renewable = np.random.uniform(200, 400, len(time_index))
        conventional = np.random.uniform(300, 500, len(time_index))
        
        ax.fill_between(time_index, 0, renewable, label="Renewable", alpha=0.7, color="green")
        ax.fill_between(time_index, renewable, renewable + conventional, label="Conventional", alpha=0.7, color="red")
        
        ax.set_xlabel("Time")
        ax.set_ylabel("Generation (MW)")
        ax.set_title("Generation Stack Plot")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / filename, dpi=self.settings.output.figure_dpi, bbox_inches="tight")
        plt.close()
        
        logger.debug(f"Generated generation stack plot: {filename}")
    
    def plot_demand_vs_generation(
        self,
        results: Dict[str, Any],
        output_path: Path,
        filename: str = "demand_vs_generation.png",
    ) -> None:
        """Plot demand vs generation."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        time_index = pd.date_range("2025-01-01", periods=8760, freq="1H")
        
        demand = np.random.uniform(800, 1200, len(time_index))
        generation = demand + np.random.uniform(-50, 50, len(time_index))
        
        ax.plot(time_index, demand, label="Demand", linewidth=2, color="blue")
        ax.plot(time_index, generation, label="Generation", linewidth=2, color="green", linestyle="--")
        
        ax.set_xlabel("Time")
        ax.set_ylabel("Power (MW)")
        ax.set_title("Demand vs Generation")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / filename, dpi=self.settings.output.figure_dpi, bbox_inches="tight")
        plt.close()
        
        logger.debug(f"Generated demand vs generation plot: {filename}")
    
    def plot_storage_operation(
        self,
        results: Dict[str, Any],
        output_path: Path,
        filename: str = "storage_operation.png",
    ) -> None:
        """Plot storage state of charge."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        time_index = pd.date_range("2025-01-01", periods=8760, freq="1H")
        
        # Sample SOC data
        soc = 0.5 + 0.3 * np.sin(np.linspace(0, 4 * np.pi, len(time_index)))
        soc = np.clip(soc, 0, 1)
        
        ax.plot(time_index, soc * 100, linewidth=2, color="purple")
        ax.fill_between(time_index, 0, soc * 100, alpha=0.3, color="purple")
        
        ax.set_xlabel("Time")
        ax.set_ylabel("State of Charge (%)")
        ax.set_title("Storage State of Charge")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / filename, dpi=self.settings.output.figure_dpi, bbox_inches="tight")
        plt.close()
        
        logger.debug(f"Generated storage operation plot: {filename}")
    
    def plot_cost_breakdown(
        self,
        results: Dict[str, Any],
        output_path: Path,
        filename: str = "cost_breakdown.png",
    ) -> None:
        """Plot cost breakdown pie chart."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Sample cost data
        costs = {
            "Fixed Costs": 50000000,
            "Variable Costs": 30000000,
            "Transmission": 10000000,
            "Storage": 5000000,
        }
        
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"]
        wedges, texts, autotexts = ax.pie(
            costs.values(),
            labels=costs.keys(),
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
        )
        
        ax.set_title("Annual Cost Breakdown", fontsize=16, fontweight="bold")
        
        plt.tight_layout()
        plt.savefig(output_path / filename, dpi=self.settings.output.figure_dpi, bbox_inches="tight")
        plt.close()
        
        logger.debug(f"Generated cost breakdown plot: {filename}")
    
    def plot_emissions_comparison(
        self,
        results: Dict[str, Any],
        output_path: Path,
        filename: str = "emissions_comparison.png",
    ) -> None:
        """Plot emissions comparison bar chart."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scenarios = ["Baseline", "High RE", "Storage Heavy", "Grid Expansion"]
        emissions = [500000, 200000, 250000, 300000]  # kg CO2
        
        bars = ax.bar(scenarios, emissions, color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"])
        
        ax.set_ylabel("CO2 Emissions (kg)")
        ax.set_title("Emissions Comparison by Scenario")
        ax.grid(True, alpha=0.3, axis="y")
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height/1000:.0f}k",
                ha="center",
                va="bottom",
            )
        
        plt.tight_layout()
        plt.savefig(output_path / filename, dpi=self.settings.output.figure_dpi, bbox_inches="tight")
        plt.close()
        
        logger.debug(f"Generated emissions comparison plot: {filename}")
    
    def plot_network_map(
        self,
        system_data: Dict[str, Any],
        output_path: Path,
        filename: str = "network_map.html",
    ) -> None:
        """Generate interactive network map using Plotly."""
        # Create sample network visualization
        fig = go.Figure()
        
        # Sample nodes
        nodes = {
            "Node 1": (10, 20),
            "Node 2": (30, 25),
            "Node 3": (25, 15),
        }
        
        # Plot nodes
        for node, (x, y) in nodes.items():
            fig.add_trace(go.Scatter(
                x=[x],
                y=[y],
                mode="markers+text",
                marker=dict(size=20, color="blue"),
                text=node,
                textposition="top center",
                name=node,
            ))
        
        # Plot edges (transmission lines)
        edges = [("Node 1", "Node 2"), ("Node 2", "Node 3")]
        for from_node, to_node in edges:
            x0, y0 = nodes[from_node]
            x1, y1 = nodes[to_node]
            fig.add_trace(go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(width=3, color="gray"),
                showlegend=False,
            ))
        
        fig.update_layout(
            title="Energy Network Map",
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            showlegend=True,
        )
        
        output_file = output_path / filename
        fig.write_html(str(output_file))
        
        logger.debug(f"Generated network map: {filename}")

