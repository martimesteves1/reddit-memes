"""
Utility Module
==============

This module contains utility functions and classes that are used throughout the project.

Features:
---------
- Logging: Provides a simple logging interface TODO
- Progress Bar: Provides a simple progress bar TODO
- Timer: Provides a simple timer TODO
- Parallel Processing: Provides a simple interface for parallel processing
"""

import concurrent.futures
import os
from collections.abc import Iterable
from functools import wraps

import pandas as pd


def parallelize(max_workers: int = os.cpu_count()):
    """
    Decorator to parallelize a function that takes a list of arguments.

    Parameters:
    -----------
    max_workers : int
        The maximum number of workers to use for parallel processing.
        Defaults to the number of CPUs available on the system.

    Returns:
    --------
    decorator : function
        A decorator function that can be used to parallelize a function.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if args and isinstance(args[0], list | tuple):
                args_list = args[0]
                results = []
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_workers
                ) as executor:
                    futures = [
                        executor.submit(func, *args_item, **kwargs)
                        for args_item in args_list
                    ]
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            results.append(future.result())
                        except Exception:
                            results.append(None)
                return results
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


def remove_outliers_iqr(
    df: pd.DataFrame, columns: Iterable[str], iqr_factor: float = 1.5
) -> pd.DataFrame:
    """
    Removes outliers from specified columns in a DataFrame using the IQR method.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame from which to remove outliers.
    columns : list
        List of columns from which to remove outliers.
    iqr_factor : float
        Factor to multiply the IQR by to determine the lower and upper bounds.
        Defaults to 1.5.

    Returns:
    --------
    df_filtered : pd.DataFrame
        DataFrame with outliers removed.
    """
    df_filtered = df.copy()

    for col in columns:
        Q1 = df_filtered[col].quantile(0.25)
        Q3 = df_filtered[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - iqr_factor * IQR
        upper_bound = Q3 + iqr_factor * IQR
        initial_count = df_filtered.shape[0]

        df_filtered = df_filtered[
            (df_filtered[col] >= lower_bound)
            & (df_filtered[col] <= upper_bound)
        ]
        final_count = df_filtered.shape[0]

        print(
            f"Removed {initial_count - final_count} outliers from {col}"
        )  # Use logging here TODO

    return df_filtered
