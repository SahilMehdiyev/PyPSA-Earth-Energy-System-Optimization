#!/usr/bin/env python3
"""Demo script to run and display energy system optimization results."""

import json
from pathlib import Path
from main import simulate_command, optimize_command
import argparse

def run_demo():
    """Run a simple demo simulation."""
    print("=" * 60)
    print("PyPSA-Earth Energy System Optimization - Demo")
    print("=" * 60)
    print()
    
    args = argparse.Namespace(
        scenario="baseline",
        year=2025,
        region="global",
        output=None,
        parallel=False
    )
    
    print("Running baseline scenario simulation...")
    print("-" * 60)
    simulate_command(args)
    print()
    
    results_path = Path("results/simulations/baseline_results.json")
    if results_path.exists():
        print("=" * 60)
        print("SIMULATION RESULTS")
        print("=" * 60)
        with open(results_path, "r") as f:
            results = json.load(f)
        
        print(f"\nScenario: {results.get('scenario', 'N/A')}")
        print(f"Year: {results.get('year', 'N/A')}")
        print(f"Region: {results.get('region', 'N/A')}")
        print(f"\nStatus: {results.get('solve_status', 'N/A')}")
        print(f"Objective: {results.get('objective', 0):.2f}")
        print(f"\nGeneration:")
        print(f"  Total: {results.get('total_generation_mwh', 0):,.0f} MWh")
        print(f"  Renewable: {results.get('renewable_generation_mwh', 0):,.0f} MWh")
        print(f"  Conventional: {results.get('conventional_generation_mwh', 0):,.0f} MWh")
        print(f"  Renewable Penetration: {results.get('renewable_penetration', 0)*100:.1f}%")
        print(f"\nDemand:")
        print(f"  Total: {results.get('total_demand_mwh', 0):,.0f} MWh")
        print(f"\nCosts & Emissions:")
        print(f"  Total Cost: ${results.get('total_cost', 0):,.0f}")
        print(f"  Total CO2 Emissions: {results.get('total_emissions_kg_co2', 0):,.0f} kg")
        print(f"\nSimulation Time: {results.get('simulation_time_seconds', 0):.2f} seconds")
        print()
        print("=" * 60)
        print("Results saved to:", results_path)
        print("=" * 60)
    else:
        print("Warning: Results file not found!")

if __name__ == "__main__":
    run_demo()

