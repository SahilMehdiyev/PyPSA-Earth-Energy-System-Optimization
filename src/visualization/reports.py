"""Report generation utilities."""

from pathlib import Path
from typing import Dict, Any, Optional
import logging

import pandas as pd
from jinja2 import Template

from src.config import Settings


logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate HTML and PDF reports."""
    
    def __init__(self, settings: Settings) -> None:
        """
        Initialize report generator.
        
        Args:
            settings: Configuration settings
        """
        self.settings = settings
    
    def generate_report(
        self,
        results_path: Path,
        output_path: Path,
        template_path: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Generate HTML report from results.
        
        Args:
            results_path: Path to simulation results
            output_path: Path to save report
            template_path: Optional custom template path
            
        Returns:
            Path to generated report
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load results
        import json
        results_file = None
        for file in Path(results_path).glob("*_results.json"):
            results_file = file
            break
        
        if not results_file:
            logger.warning("No results file found")
            return None
        
        with open(results_file, "r") as f:
            results = json.load(f)
        
        # Generate HTML report
        html_content = self._generate_html_content(results)
        
        report_file = output_path / "simulation_report.html"
        with open(report_file, "w") as f:
            f.write(html_content)
        
        logger.info(f"Report generated: {report_file}")
        
        return report_file
    
    def _generate_html_content(self, results: Dict[str, Any]) -> str:
        """Generate HTML content from results."""
        template_str = """
<!DOCTYPE html>
<html>
<head>
    <title>Energy System Simulation Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            margin-top: 30px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #3498db;
            color: white;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .metric {
            display: inline-block;
            margin: 10px 20px;
            padding: 15px;
            background-color: #ecf0f1;
            border-radius: 5px;
            min-width: 200px;
        }
        .metric-label {
            font-size: 14px;
            color: #7f8c8d;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Energy System Simulation Report</h1>
        
        <h2>Scenario Information</h2>
        <table>
            <tr>
                <th>Parameter</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Scenario Name</td>
                <td>{{ results.scenario }}</td>
            </tr>
            <tr>
                <td>Year</td>
                <td>{{ results.year }}</td>
            </tr>
            <tr>
                <td>Region</td>
                <td>{{ results.region }}</td>
            </tr>
        </table>
        
        <h2>Key Metrics</h2>
        <div class="metric">
            <div class="metric-label">Total Generation</div>
            <div class="metric-value">{{ "%.2f"|format(results.total_generation_mwh / 1e6) }} TWh</div>
        </div>
        <div class="metric">
            <div class="metric-label">Total Demand</div>
            <div class="metric-value">{{ "%.2f"|format(results.total_demand_mwh / 1e6) }} TWh</div>
        </div>
        <div class="metric">
            <div class="metric-label">Renewable Penetration</div>
            <div class="metric-value">{{ "%.1f"|format(results.renewable_penetration * 100) }}%</div>
        </div>
        <div class="metric">
            <div class="metric-label">Total Cost</div>
            <div class="metric-value">${{ "%.2f"|format(results.total_cost / 1e6) }}M</div>
        </div>
        <div class="metric">
            <div class="metric-label">CO2 Emissions</div>
            <div class="metric-value">{{ "%.2f"|format(results.total_emissions_kg_co2 / 1e6) }} Mt</div>
        </div>
        
        <h2>Generation Breakdown</h2>
        <table>
            <tr>
                <th>Type</th>
                <th>Generation (TWh)</th>
                <th>Percentage</th>
            </tr>
            <tr>
                <td>Renewable</td>
                <td>{{ "%.2f"|format(results.renewable_generation_mwh / 1e6) }}</td>
                <td>{{ "%.1f"|format((results.renewable_generation_mwh / results.total_generation_mwh * 100) if results.total_generation_mwh > 0 else 0) }}%</td>
            </tr>
            <tr>
                <td>Conventional</td>
                <td>{{ "%.2f"|format(results.conventional_generation_mwh / 1e6) }}</td>
                <td>{{ "%.1f"|format((results.conventional_generation_mwh / results.total_generation_mwh * 100) if results.total_generation_mwh > 0 else 0) }}%</td>
            </tr>
        </table>
        
        <h2>Simulation Status</h2>
        <table>
            <tr>
                <th>Status</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Solve Status</td>
                <td>{{ results.solve_status }}</td>
            </tr>
            <tr>
                <td>Objective Value</td>
                <td>{{ "%.2f"|format(results.objective) }}</td>
            </tr>
            <tr>
                <td>Simulation Time</td>
                <td>{{ "%.2f"|format(results.simulation_time_seconds) }} seconds</td>
            </tr>
        </table>
    </div>
</body>
</html>
        """
        
        template = Template(template_str)
        return template.render(results=results)
    
    def generate_comparative_table(
        self,
        scenarios_results: Dict[str, Dict[str, Any]],
        output_path: Path,
    ) -> Path:
        """Generate comparative table for multiple scenarios."""
        # Create DataFrame
        data = []
        for scenario_name, results in scenarios_results.items():
            data.append({
                "Scenario": scenario_name,
                "Year": results.get("year", "N/A"),
                "Total Generation (TWh)": results.get("total_generation_mwh", 0) / 1e6,
                "Renewable Penetration (%)": results.get("renewable_penetration", 0) * 100,
                "Total Cost (M$)": results.get("total_cost", 0) / 1e6,
                "Emissions (Mt CO2)": results.get("total_emissions_kg_co2", 0) / 1e6,
            })
        
        df = pd.DataFrame(data)
        
        # Save as CSV
        csv_file = output_path / "scenario_comparison.csv"
        df.to_csv(csv_file, index=False)
        
        # Save as LaTeX table
        latex_file = output_path / "scenario_comparison.tex"
        with open(latex_file, "w") as f:
            f.write(df.to_latex(index=False, float_format="%.2f"))
        
        logger.info(f"Comparative table saved to {csv_file} and {latex_file}")
        
        return csv_file

