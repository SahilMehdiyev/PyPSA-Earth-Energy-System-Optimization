"""Setup script for PyPSA-Earth Energy System Optimization project."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pypsa-earth-optimization",
    version="0.1.0",
    author="Energy System Research Team",
    description="Energy system optimization using PyPSA-Earth framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pypsa-earth-optimization",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=[
        "pypsa>=0.25.0",
        "pyomo>=6.5.0",
        "pulp>=2.7.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "plotly>=5.14.0",
        "networkx>=3.0",
        "geopandas>=0.13.0",
        "scipy>=1.10.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0",
        "jinja2>=3.1.0",
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
    ],
    extras_require={
        "dev": [
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "isort>=5.12.0",
        ],
        "solvers": [
            "gurobipy>=10.0.0",  # Commercial solver
        ],
    },
    entry_points={
        "console_scripts": [
            "energy-optimizer=main:main",
        ],
    },
)

