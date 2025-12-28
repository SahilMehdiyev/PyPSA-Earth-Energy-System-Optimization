"""Data loading utilities for energy system data."""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

import pandas as pd
import numpy as np
import geopandas as gpd
import xarray as xr


logger = logging.getLogger(__name__)


class DataLoader:
    """Load energy system data from various formats."""
    
    def __init__(self, base_path: Path) -> None:
        """
        Initialize data loader.
        
        Args:
            base_path: Base path for data files
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def load_csv(
        self,
        file_path: Path,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            file_path: Path to CSV file
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            DataFrame with loaded data
        """
        full_path = self.base_path / file_path if not Path(file_path).is_absolute() else Path(file_path)
        
        if not full_path.exists():
            raise FileNotFoundError(f"CSV file not found: {full_path}")
        
        logger.info(f"Loading CSV: {full_path}")
        df = pd.read_csv(full_path, **kwargs)
        
        # Try to parse datetime index if 'time' or 'date' column exists
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_cols:
            df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
            df.set_index(date_cols[0], inplace=True)
        
        return df
    
    def load_json(self, file_path: Path) -> Dict[str, Any]:
        """
        Load data from JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Dictionary with loaded data
        """
        full_path = self.base_path / file_path if not Path(file_path).is_absolute() else Path(file_path)
        
        if not full_path.exists():
            raise FileNotFoundError(f"JSON file not found: {full_path}")
        
        logger.info(f"Loading JSON: {full_path}")
        with open(full_path, "r") as f:
            data = json.load(f)
        
        return data
    
    def load_netcdf(
        self,
        file_path: Path,
        variables: Optional[List[str]] = None,
    ) -> xr.Dataset:
        """
        Load data from NetCDF file.
        
        Args:
            file_path: Path to NetCDF file
            variables: Optional list of variables to load
            
        Returns:
            xarray Dataset
        """
        full_path = self.base_path / file_path if not Path(file_path).is_absolute() else Path(file_path)
        
        if not full_path.exists():
            raise FileNotFoundError(f"NetCDF file not found: {full_path}")
        
        logger.info(f"Loading NetCDF: {full_path}")
        ds = xr.open_dataset(full_path)
        
        if variables:
            ds = ds[variables]
        
        return ds
    
    def load_geojson(self, file_path: Path) -> gpd.GeoDataFrame:
        """
        Load geographic data from GeoJSON file.
        
        Args:
            file_path: Path to GeoJSON file
            
        Returns:
            GeoDataFrame with loaded data
        """
        full_path = self.base_path / file_path if not Path(file_path).is_absolute() else Path(file_path)
        
        if not full_path.exists():
            raise FileNotFoundError(f"GeoJSON file not found: {full_path}")
        
        logger.info(f"Loading GeoJSON: {full_path}")
        gdf = gpd.read_file(full_path)
        
        return gdf
    
    def load_renewable_profiles(
        self,
        file_path: Path,
        technology: str = "solar",
        region: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load renewable generation profiles.
        
        Args:
            file_path: Path to profile data file
            technology: Technology type (solar, wind_onshore, wind_offshore)
            region: Optional region filter
            
        Returns:
            DataFrame with time series profiles (columns: locations, index: time)
        """
        # Try CSV first
        try:
            df = self.load_csv(file_path)
            
            # Filter by technology if column exists
            if "technology" in df.columns:
                df = df[df["technology"] == technology]
            
            # Filter by region if specified
            if region and "region" in df.columns:
                df = df[df["region"] == region]
            
            # Pivot to time series format if needed
            if "time" in df.columns or df.index.dtype == "datetime64[ns]":
                if "location" in df.columns and "capacity_factor" in df.columns:
                    df = df.pivot_table(
                        index=df.index if df.index.dtype == "datetime64[ns]" else "time",
                        columns="location",
                        values="capacity_factor",
                    )
            
            return df
        
        except Exception as e:
            logger.warning(f"Error loading renewable profiles: {e}")
            # Return empty DataFrame with proper structure
            return pd.DataFrame()
    
    def load_demand_data(
        self,
        file_path: Path,
        region: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load electricity demand data.
        
        Args:
            file_path: Path to demand data file
            region: Optional region filter
            
        Returns:
            DataFrame with demand time series (columns: regions/nodes, index: time)
        """
        df = self.load_csv(file_path)
        
        # Filter by region if specified
        if region and "region" in df.columns:
            df = df[df["region"] == region]
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"])
                df.set_index("time", inplace=True)
            elif "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
        
        return df
    
    def load_transmission_network(
        self,
        file_path: Path,
    ) -> gpd.GeoDataFrame:
        """
        Load transmission network data.
        
        Args:
            file_path: Path to network data file (GeoJSON or CSV with coordinates)
            
        Returns:
            GeoDataFrame with transmission lines
        """
        # Try GeoJSON first
        if file_path.suffix == ".geojson" or file_path.suffix == ".json":
            try:
                return self.load_geojson(file_path)
            except Exception:
                pass
        
        # Fall back to CSV with coordinates
        df = self.load_csv(file_path)
        
        # Create GeoDataFrame if coordinates exist
        if "longitude" in df.columns and "latitude" in df.columns:
            gdf = gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df.longitude, df.latitude),
                crs="EPSG:4326",
            )
            return gdf
        
        return gpd.GeoDataFrame(df)
    
    def load_generation_capacity(
        self,
        file_path: Path,
        region: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load existing generation capacity data.
        
        Args:
            file_path: Path to capacity data file
            region: Optional region filter
            
        Returns:
            DataFrame with capacity data
        """
        df = self.load_csv(file_path)
        
        # Filter by region if specified
        if region and "region" in df.columns:
            df = df[df["region"] == region]
        
        # Ensure required columns
        required_cols = ["technology", "capacity_mw"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        return df
    
    def chunk_load(
        self,
        file_path: Path,
        chunk_size: int = 10000,
        **kwargs,
    ) -> pd.io.parsers.TextFileReader:
        """
        Load large CSV file in chunks.
        
        Args:
            file_path: Path to CSV file
            chunk_size: Number of rows per chunk
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            TextFileReader iterator
        """
        full_path = self.base_path / file_path if not Path(file_path).is_absolute() else Path(file_path)
        
        if not full_path.exists():
            raise FileNotFoundError(f"CSV file not found: {full_path}")
        
        logger.info(f"Loading CSV in chunks: {full_path}")
        return pd.read_csv(full_path, chunksize=chunk_size, **kwargs)

