"""Data preprocessing utilities for energy system data."""

from pathlib import Path
from typing import Optional, List, Tuple
import logging

import pandas as pd
import numpy as np
from scipy import interpolate
from scipy.stats import zscore


logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocess and clean energy system data."""
    
    def __init__(self) -> None:
        """Initialize preprocessor."""
        pass
    
    def clean_data(
        self,
        df: pd.DataFrame,
        remove_duplicates: bool = True,
        handle_missing: str = "interpolate",
    ) -> pd.DataFrame:
        """
        Clean data by removing duplicates and handling missing values.
        
        Args:
            df: Input DataFrame
            remove_duplicates: Whether to remove duplicate rows
            handle_missing: Strategy for missing values ('interpolate', 'forward_fill', 'backward_fill', 'drop')
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Remove duplicates
        if remove_duplicates:
            initial_len = len(df_clean)
            df_clean = df_clean.drop_duplicates()
            removed = initial_len - len(df_clean)
            if removed > 0:
                logger.info(f"Removed {removed} duplicate rows")
        
        # Handle missing values
        missing_count = df_clean.isnull().sum().sum()
        if missing_count > 0:
            logger.info(f"Handling {missing_count} missing values using strategy: {handle_missing}")
            
            if handle_missing == "interpolate":
                df_clean = df_clean.interpolate(method="time" if isinstance(df_clean.index, pd.DatetimeIndex) else "linear")
            elif handle_missing == "forward_fill":
                df_clean = df_clean.fillna(method="ffill")
            elif handle_missing == "backward_fill":
                df_clean = df_clean.fillna(method="bfill")
            elif handle_missing == "drop":
                df_clean = df_clean.dropna()
            else:
                raise ValueError(f"Unknown missing value strategy: {handle_missing}")
        
        return df_clean
    
    def normalize_data(
        self,
        df: pd.DataFrame,
        method: str = "min_max",
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Normalize data to [0, 1] range.
        
        Args:
            df: Input DataFrame
            method: Normalization method ('min_max', 'z_score')
            columns: Columns to normalize (None = all numeric columns)
            
        Returns:
            Normalized DataFrame
        """
        df_norm = df.copy()
        
        if columns is None:
            columns = df_norm.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col not in df_norm.columns:
                continue
            
            if method == "min_max":
                col_min = df_norm[col].min()
                col_max = df_norm[col].max()
                if col_max > col_min:
                    df_norm[col] = (df_norm[col] - col_min) / (col_max - col_min)
            
            elif method == "z_score":
                col_mean = df_norm[col].mean()
                col_std = df_norm[col].std()
                if col_std > 0:
                    df_norm[col] = (df_norm[col] - col_mean) / col_std
        
        return df_norm
    
    def detect_outliers(
        self,
        df: pd.DataFrame,
        method: str = "zscore",
        threshold: float = 3.0,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Detect outliers in data.
        
        Args:
            df: Input DataFrame
            method: Detection method ('zscore', 'iqr')
            threshold: Threshold for outlier detection
            columns: Columns to check (None = all numeric columns)
            
        Returns:
            DataFrame with boolean mask (True = outlier)
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
        
        for col in columns:
            if col not in df.columns:
                continue
            
            if method == "zscore":
                z_scores = np.abs(zscore(df[col].dropna()))
                outlier_mask[col] = z_scores > threshold
            
            elif method == "iqr":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outlier_mask[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
        
        return outlier_mask
    
    def handle_outliers(
        self,
        df: pd.DataFrame,
        outlier_mask: pd.DataFrame,
        method: str = "clip",
    ) -> pd.DataFrame:
        """
        Handle detected outliers.
        
        Args:
            df: Input DataFrame
            outlier_mask: Boolean mask from detect_outliers
            method: Handling method ('clip', 'remove', 'interpolate')
            
        Returns:
            DataFrame with outliers handled
        """
        df_clean = df.copy()
        
        for col in df_clean.columns:
            if col not in outlier_mask.columns:
                continue
            
            outliers = outlier_mask[col]
            n_outliers = outliers.sum()
            
            if n_outliers == 0:
                continue
            
            logger.info(f"Handling {n_outliers} outliers in column '{col}' using method: {method}")
            
            if method == "clip":
                # Clip to min/max of non-outlier values
                non_outlier_values = df_clean.loc[~outliers, col]
                if len(non_outlier_values) > 0:
                    df_clean.loc[outliers, col] = df_clean.loc[outliers, col].clip(
                        lower=non_outlier_values.min(),
                        upper=non_outlier_values.max(),
                    )
            
            elif method == "remove":
                df_clean = df_clean[~outliers]
            
            elif method == "interpolate":
                # Replace outliers with interpolated values
                df_clean.loc[outliers, col] = np.nan
                df_clean[col] = df_clean[col].interpolate(method="time" if isinstance(df_clean.index, pd.DatetimeIndex) else "linear")
        
        return df_clean
    
    def aggregate_data(
        self,
        df: pd.DataFrame,
        freq: str = "1H",
        method: str = "mean",
    ) -> pd.DataFrame:
        """
        Aggregate data to specified frequency.
        
        Args:
            df: Input DataFrame with datetime index
            freq: Target frequency (e.g., '1H', '1D', '1W')
            method: Aggregation method ('mean', 'sum', 'max', 'min')
            
        Returns:
            Aggregated DataFrame
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have datetime index for aggregation")
        
        if method == "mean":
            df_agg = df.resample(freq).mean()
        elif method == "sum":
            df_agg = df.resample(freq).sum()
        elif method == "max":
            df_agg = df.resample(freq).max()
        elif method == "min":
            df_agg = df.resample(freq).min()
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        logger.info(f"Aggregated data from {len(df)} to {len(df_agg)} timesteps ({freq})")
        
        return df_agg
    
    def create_renewable_profile(
        self,
        time_index: pd.DatetimeIndex,
        technology: str = "solar",
        location: Optional[Tuple[float, float]] = None,
        base_capacity_factor: float = 0.2,
    ) -> pd.Series:
        """
        Create synthetic renewable generation profile.
        
        Args:
            time_index: Time index for profile
            technology: Technology type (solar, wind_onshore, wind_offshore)
            location: Optional (lat, lon) for location-specific patterns
            base_capacity_factor: Base capacity factor
            
        Returns:
            Capacity factor time series (0-1)
        """
        profile = pd.Series(index=time_index, dtype=float)
        
        if technology == "solar":
            # Solar: high during day, zero at night, seasonal variation
            for timestamp in time_index:
                hour = timestamp.hour
                day_of_year = timestamp.timetuple().tm_yday
                
                # Diurnal pattern (sine wave for daylight hours)
                if 6 <= hour <= 18:
                    diurnal_factor = np.sin(np.pi * (hour - 6) / 12)
                else:
                    diurnal_factor = 0.0
                
                # Seasonal pattern (higher in summer)
                seasonal_factor = 0.7 + 0.3 * np.cos(2 * np.pi * (day_of_year - 172) / 365)
                
                profile[timestamp] = base_capacity_factor * diurnal_factor * seasonal_factor
        
        elif technology in ["wind_onshore", "wind_offshore"]:
            # Wind: more variable, less predictable
            np.random.seed(42)  # For reproducibility
            base = np.random.normal(base_capacity_factor, 0.15, len(time_index))
            
            # Add some autocorrelation (wind persists)
            for i in range(1, len(base)):
                base[i] = 0.7 * base[i-1] + 0.3 * base[i]
            
            # Seasonal variation (higher in winter)
            for i, timestamp in enumerate(time_index):
                day_of_year = timestamp.timetuple().tm_yday
                seasonal_factor = 0.8 + 0.4 * np.cos(2 * np.pi * (day_of_year - 355) / 365)
                base[i] *= seasonal_factor
            
            profile = pd.Series(base, index=time_index)
            profile = profile.clip(lower=0, upper=1)
        
        else:
            # Default: constant profile
            profile = pd.Series(base_capacity_factor, index=time_index)
        
        return profile
    
    def resample_time_series(
        self,
        series: pd.Series,
        target_freq: str = "1H",
        method: str = "interpolate",
    ) -> pd.Series:
        """
        Resample time series to target frequency.
        
        Args:
            series: Input time series
            target_freq: Target frequency
            method: Resampling method ('interpolate', 'forward_fill', 'mean', 'sum')
            
        Returns:
            Resampled time series
        """
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("Series must have datetime index")
        
        if method == "interpolate":
            resampled = series.resample(target_freq).interpolate(method="time")
        elif method == "forward_fill":
            resampled = series.resample(target_freq).ffill()
        elif method == "mean":
            resampled = series.resample(target_freq).mean()
        elif method == "sum":
            resampled = series.resample(target_freq).sum()
        else:
            raise ValueError(f"Unknown resampling method: {method}")
        
        return resampled

