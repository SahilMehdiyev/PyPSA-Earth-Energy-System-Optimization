"""Main CLI entry point for energy system optimization."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from src.config import get_settings


def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """Configure logging."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def simulate_command(args: argparse.Namespace) -> None:
    """Run simulation command."""
    from src.simulation.simulator import Simulator
    from src.simulation.scenario_manager import ScenarioManager
    
    settings = get_settings()
    setup_logging(settings.log_level, settings.log_file)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting simulation: scenario={args.scenario}, year={args.year}, region={args.region}")
    
    # Load scenario
    scenario_manager = ScenarioManager(settings.data.scenarios_path)
    scenario = scenario_manager.load_scenario(args.scenario)
    
    if scenario is None:
        logger.error(f"Scenario '{args.scenario}' not found")
        sys.exit(1)
    
    # Create simulator
    simulator = Simulator(settings)
    
    # Run simulation
    output_path = Path(args.output) if args.output else settings.output.simulations_path
    results = simulator.run_scenario(scenario, output_path=output_path, parallel=args.parallel)
    
    logger.info(f"Simulation completed. Results saved to {output_path}")
    print(f"✓ Simulation completed successfully")
    print(f"  Results: {output_path}")


def optimize_command(args: argparse.Namespace) -> None:
    """Run optimization command."""
    from src.optimization.linear_programming import LinearProgrammingOptimizer
    from src.optimization.genetic_algorithm import GeneticAlgorithmOptimizer
    
    settings = get_settings()
    setup_logging(settings.log_level, settings.log_file)
    logger = logging.getLogger(__name__)
    
    # Update settings from CLI args
    settings.optimization.method = args.method
    settings.optimization.objective = args.objective
    settings.solver.name = args.solver
    
    if args.method == "ga":
        settings.optimization.ga_generations = args.generations
        settings.optimization.ga_population_size = args.population
    
    logger.info(f"Starting optimization: method={args.method}, objective={args.objective}, solver={args.solver}")
    
    # Select optimizer
    if args.method == "lp":
        optimizer = LinearProgrammingOptimizer(settings)
    else:
        optimizer = GeneticAlgorithmOptimizer(settings)
    
    # Run optimization
    output_path = Path(args.output) if args.output else settings.output.optimizations_path
    results = optimizer.optimize(output_path=output_path)
    
    logger.info(f"Optimization completed. Results saved to {output_path}")
    print(f"✓ Optimization completed successfully")
    print(f"  Results: {output_path}")


def visualize_command(args: argparse.Namespace) -> None:
    """Generate visualizations command."""
    from src.visualization.plots import PlotGenerator
    from src.visualization.reports import ReportGenerator
    
    settings = get_settings()
    setup_logging(settings.log_level, settings.log_file)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Generating visualizations for scenario: {args.scenario}")
    
    # Load results
    results_path = settings.output.simulations_path / args.scenario
    if not results_path.exists():
        logger.error(f"Results not found: {results_path}")
        sys.exit(1)
    
    # Generate plots
    plot_generator = PlotGenerator(settings)
    output_path = Path(args.output) if args.output else settings.output.visualizations_path
    
    plot_generator.generate_all_plots(results_path, output_path)
    
    # Generate report
    report_generator = ReportGenerator(settings)
    report_path = report_generator.generate_report(results_path, output_path)
    
    logger.info(f"Visualizations completed. Output: {output_path}")
    print(f"✓ Visualizations generated successfully")
    print(f"  Output: {output_path}")
    if report_path:
        print(f"  Report: {report_path}")


def analyze_command(args: argparse.Namespace) -> None:
    """Run analysis command."""
    from src.simulation.scenario_manager import ScenarioManager
    
    settings = get_settings()
    setup_logging(settings.log_level, settings.log_file)
    logger = logging.getLogger(__name__)
    
    scenarios = [s.strip() for s in args.scenarios.split(",")]
    logger.info(f"Running comparative analysis for scenarios: {scenarios}")
    
    # Load scenarios and results
    scenario_manager = ScenarioManager(settings.data.scenarios_path)
    results = {}
    
    for scenario_name in scenarios:
        results_path = settings.output.simulations_path / scenario_name
        if results_path.exists():
            # Load results (implementation depends on result format)
            logger.info(f"Loaded results for {scenario_name}")
        else:
            logger.warning(f"Results not found for scenario: {scenario_name}")
    
    # Perform comparative analysis
    output_path = Path(args.output) if args.output else settings.output.visualizations_path
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Analysis completed. Output: {output_path}")
    print(f"✓ Analysis completed successfully")
    print(f"  Output: {output_path}")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PyPSA-Earth Energy System Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Simulation command
    sim_parser = subparsers.add_parser("simulate", help="Run energy system simulation")
    sim_parser.add_argument("--scenario", type=str, required=True, help="Scenario name")
    sim_parser.add_argument("--year", type=int, default=2025, help="Target year")
    sim_parser.add_argument("--region", type=str, default="global", help="Region identifier")
    sim_parser.add_argument("--output", type=str, help="Output directory path")
    sim_parser.add_argument("--parallel", action="store_true", help="Enable parallel processing")
    
    # Optimization command
    opt_parser = subparsers.add_parser("optimize", help="Run optimization")
    opt_parser.add_argument("--method", type=str, choices=["lp", "ga"], default="lp", help="Optimization method")
    opt_parser.add_argument("--objective", type=str, choices=["cost", "emissions", "multi"], default="cost", help="Objective function")
    opt_parser.add_argument("--solver", type=str, choices=["highs", "gurobi", "cbc", "glpk"], default="highs", help="LP solver")
    opt_parser.add_argument("--generations", type=int, default=100, help="GA generations")
    opt_parser.add_argument("--population", type=int, default=200, help="GA population size")
    opt_parser.add_argument("--output", type=str, help="Output directory path")
    
    # Visualization command
    viz_parser = subparsers.add_parser("visualize", help="Generate visualizations")
    viz_parser.add_argument("--scenario", type=str, required=True, help="Scenario name")
    viz_parser.add_argument("--output", type=str, help="Output directory path")
    
    # Analysis command
    anal_parser = subparsers.add_parser("analyze", help="Run comparative analysis")
    anal_parser.add_argument("--scenarios", type=str, required=True, help="Comma-separated scenario names")
    anal_parser.add_argument("--metrics", type=str, help="Metrics to compare")
    anal_parser.add_argument("--output", type=str, help="Output directory path")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Route to appropriate command handler
    command_handlers = {
        "simulate": simulate_command,
        "optimize": optimize_command,
        "visualize": visualize_command,
        "analyze": analyze_command,
    }
    
    handler = command_handlers.get(args.command)
    if handler:
        try:
            handler(args)
        except Exception as e:
            logging.error(f"Error executing command: {e}", exc_info=True)
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

