"""
Utility Module
==============

This module contains utility functions and classes that are used throughout the project.

Features:
---------
- Logging: Provides a simple logging interface
- Progress Bar: Provides a simple progress bar TODO
- Parallel Processing: Provides a simple interface for parallel processing
"""

import logging
import os
import shutil

from typing import Iterable
from functools import wraps
from typing import Callable
from itertools import repeat

import concurrent.futures
from tqdm import tqdm
import pandas as pd
from PIL import Image


def parallelize(max_workers: int = os.cpu_count()) -> Callable:
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
                with concurrent.futures.ProcessPoolExecutor(
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

        logging.info(
            f"Removed {initial_count - final_count} outliers from {col}"
        )

    return df_filtered


def process_image(image_id: str, input_path: str, output_folder: str, output_resolution: tuple[int, int]) -> bool:
    """
    Processes a single image: resizes it and saves it to the output folder.

    Parameters:
    -----------
    image_id : str
        The filename of the image to process.
    input_path : str
        Path to the folder containing the images to resize.
    output_folder : str
        Path to the folder where the resized images will be saved.
    output_resolution : tuple
        Resolution to resize the images to.

    Returns:
    --------
    bool
        True if the image was processed successfully, False otherwise.
    """    
    img_path = os.path.join(input_path, image_id)
    save_path = os.path.join(output_folder, image_id)
    try:
        with Image.open(img_path) as img:
            img_resized = img.resize(output_resolution, Image.Resampling.LANCZOS)
            img_resized.save(save_path)
        logging.info(f"Resized and saved image: {save_path}")
        return True
    except:
        logging.error(f"Failed to resize image: {img_path}")
        return False


def resize_images(input_path: str, output_resolution: tuple[int, int], 
                  output_folder: str, num_workers: int=os.cpu_count()) -> None:
    """
    Resizes images in a folder to a specified resolution and saves them in a new folder.

    Parameters:
    -----------
    input_path : str
        Path to the folder containing the images to resize.
    output_resolution : tuple
        Resolution to resize the images to.
    output_folder : str
        Path to the folder where the resized images will be saved.
    num_workers : int
        Number of workers to use for parallel processing. Defaults to the number of CPUs available.
    """
    len_input = len(os.listdir(input_path))

    try:
        os.makedirs(output_folder)
    except FileExistsError:
        shutil.rmtree(output_folder)
        os.makedirs(output_folder)
        logging.info(f"Output folder already exists. Overwriting contents.")


    if input_path == output_folder:
        logging.error("Input and output folders cannot be the same.")
        raise ValueError()
       
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_image, 
                                         os.listdir(input_path),
                                         repeat(input_path),
                                         repeat(output_folder),
                                         repeat(output_resolution)),
                            total=len_input,
                            desc="Resizing Images",
                            unit="images")
)
    len_output = len(os.listdir(output_folder))
    logging.info(
        f"Resized {len_output} images out of {len_input}" +
        f" to {output_resolution}"
        )
    return None


def logger_init(log_file: None | str = None) -> None:
    """
    Initialize the logger.

    Parameters:
    -----------
    log_file : str
        Path to the log file. If no path is found, logs are not saved
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            h for h in [logging.FileHandler(log_file) if log_file else None,
                        logging.StreamHandler()]
            if h is not None
            ],
    )
    logging.info("Logger initialized")
