.PHONY: help install demo test-all scenarios clean results summary

PYTHON = .venv/bin/python
VENV = .venv
UV = uv

.DEFAULT_GOAL := help

help:
	@echo "PyPSA-Earth Energy System Optimization - Makefile"
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:
	@echo "Installing dependencies..."
	$(UV) sync
	@echo "Installation complete!"

demo:
	@echo "Running demo script..."
	$(PYTHON) demo.py

scenario-baseline:
	@echo "Running Baseline Scenario (2025)..."
	$(PYTHON) main.py simulate --scenario baseline --year 2025 --region global

scenario-high-renewable:
	@echo "Running High Renewable Scenario (2030)..."
	$(PYTHON) main.py simulate --scenario high_renewable --year 2030 --region global

scenario-storage-heavy:
	@echo "Running Storage Heavy Scenario (2035)..."
	$(PYTHON) main.py simulate --scenario storage_heavy --year 2035 --region global

scenario-grid-expansion:
	@echo "Running Grid Expansion Scenario (2030)..."
	$(PYTHON) main.py simulate --scenario grid_expansion --year 2030 --region global

scenarios: scenario-baseline scenario-high-renewable scenario-storage-heavy scenario-grid-expansion
	@echo ""
	@echo "All scenarios completed!"

results:
	@echo "Simulation Results:"
	@ls -lh results/simulations/*.json 2>/dev/null || echo "No result files yet"

summary:
	@echo "Scenario Summaries:"
	@echo "==================="
	@for file in results/simulations/*_results.json; do \
		if [ -f "$$file" ]; then \
			echo ""; \
			echo "File: $$(basename $$file):"; \
			$(PYTHON) -c "import json, sys; data=json.load(open('$$file')); print(f\"  Scenario: {data.get('scenario', 'N/A')}\"); print(f\"  Year: {data.get('year', 'N/A')}\"); print(f\"  Renewable: {data.get('renewable_penetration', 0)*100:.1f}%\"); print(f\"  Total Generation: {data.get('total_generation_mwh', 0):,.0f} MWh\"); print(f\"  Total Cost: \$${data.get('total_cost', 0):,.0f}\"); print(f\"  CO2 Emissions: {data.get('total_emissions_kg_co2', 0):,.0f} kg\"); print(f\"  Simulation Time: {data.get('simulation_time_seconds', 0):.2f} s\");"; \
		fi; \
	done

test-all: scenarios summary
	@echo ""
	@echo "All tests completed!"

optimize-lp:
	@echo "Running Linear Programming optimization..."
	$(PYTHON) main.py optimize --method lp --objective cost --solver highs

optimize-ga:
	@echo "Running Genetic Algorithm optimization..."
	$(PYTHON) main.py optimize --method ga --generations 50 --population 100

clean:
	@echo "Cleaning result files..."
	@rm -f results/simulations/*.json
	@rm -f results/optimizations/*.json
	@rm -f results/visualizations/*
	@echo "Cleanup complete!"

clean-all: clean
	@echo "Cleaning all generated files..."
	@rm -rf .uv-cache
	@echo "Full cleanup complete!"

show-help:
	$(PYTHON) main.py --help

show-scenarios:
	@echo "Available Scenarios:"
	@for file in data/scenarios/*.yaml; do \
		[ -f "$$file" ] && basename "$$file" .yaml | sed 's/^/  - /'; \
	done

run-all: install demo scenarios summary
	@echo ""
	@echo "Full demo completed!"
