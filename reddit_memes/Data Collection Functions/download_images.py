"""
Image Downloader and Converter Module
=====================================

This module provides functionalities to download images from specified URLs, validate their formats,
convert them to a desired format (e.g., JPEG), and save them locally. It leverages Python's
`requests` library for HTTP operations, `Pillow` for image processing, and `concurrent.futures`
for parallel processing to enhance performance.

Features
--------
- **Download Images**: Fetch images from given URLs with robust error handling and retry mechanisms.
- **Validate Formats**: Ensure that downloaded images are in allowed formats to maintain consistency.
- **Convert Formats**: Convert images to a specified format, handling necessary mode changes (e.g., removing alpha channels for JPEG).
- **Save Locally**: Store processed images in a designated directory with unique filenames based on their URLs.
- **Parallel Processing**: Utilize multithreading to download and process multiple images concurrently, improving efficiency.

Usage
-----
To use this module, define a list of image URLs and specify the desired output directory and format.
Then, invoke the `process_images_parallel` function to handle the downloading and processing.

Example
-------
```python
from image_downloader import process_images_parallel

# List of image URLs to download
image_urls = [
    "https://example.com/image1.png",
    "https://example.com/image2.gif",
    "https://example.com/image3.jpg",
    # Add more URLs as needed
]

# Configuration parameters
output_directory = "downloaded_images"
desired_format = "JPEG"  # Desired image format
max_threads = 10  # Number of parallel threads

# Process images in parallel
process_images_parallel(
    urls=image_urls,
    output_dir=output_directory,
    target_format=desired_format,
    max_workers=max_threads,
)
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import pandas as pd
import requests
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_image(
    url: str, timeout: int = 10
) -> tuple[str, bytes | None, str | None]:
    """
    Download an image from a given URL.

    This function fetches the image content from the specified URL using an HTTP GET request.
    It returns the URL, the image content in bytes, and the Content-Type from the response headers.

    Parameters
    ----------
    post_id : str
        The unique identifier of the post associated with the image.
    url : str
        The URL of the image to download.
    timeout : int, optional
        The maximum time (in seconds) to wait for a response, by default 10.

    Returns
    -------
    Tuple[str, Optional[bytes], Optional[str]]
        A tuple containing:
        - The original URL.
        - The image content in bytes if the download was successful, else `None`.
        - The Content-Type header from the response if available, else `None`.

    Raises
    ------
    requests.RequestException
        If the HTTP request failed for any reason.
    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException:
        logger.exception(f"Error downloading {url}")
        return None, None
    else:
        content_type = response.headers.get("Content-Type", "").lower()
        return response.content, content_type


"""
Possible improvement to download code:

1. Add an url validator (Francisco was working on a version of it)
2. Add a retry mechanism using urllib3 Retry and HTTPAdapter:

from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

def download_image(url, timeout=10, retries=3, backoff_factor=0.3):
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=(500, 502, 504),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        content_type = response.headers.get('Content-Type', '').lower()
        return url, response.content, content_type
    except requests.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return url, None, None
"""


def validate_image(
    content: bytes, content_type: str, allowed_formats: list[str] | None = None
) -> Image.Image | None:
    """
    Validate the downloaded image content.

    This function checks whether the provided bytes represent a valid image and optionally
    verifies if the image format is within the allowed formats.

    Parameters
    ----------
    content : bytes
        The image content in bytes.
    content_type : str
        The MIME type of the image as obtained from the HTTP response headers.
    allowed_formats : list of str, optional
        A list of allowed image formats (e.g., ['jpeg', 'png']). If `None`, all formats are allowed.

    Returns
    -------
    Optional[Image.Image]
        A PIL Image object if the image is valid and its format is allowed; otherwise, `None`.

    Raises
    ------
    PIL.UnidentifiedImageError
        If the image cannot be identified or is corrupted.
    """
    if content is None:
        logger.error("No image content to validate")
        return None
    try:
        with Image.open(BytesIO(content)) as img:
            img.verify()
            image_format = img.format.lower()

        if allowed_formats and image_format not in allowed_formats:
            logger.error(f"Unsupported image format: {image_format}")
            return None

    except UnidentifiedImageError:
        logger.exception("Image validation failed: UnidentifiedImageError.")
        return None
    except Exception:
        logger.exception("Image validation failed")
        return None
    else:
        image = Image.open(BytesIO(content))
        return image


def convert_image(
    image: Image.Image, target_format: str
) -> Image.Image | None:
    """
    Convert a PIL Image to the specified target format.

    This function handles necessary conversions, such as removing alpha channels for formats
    that do not support transparency (e.g., JPEG).

    Parameters
    ----------
    image : PIL.Image.Image
        The PIL Image object to convert.
    target_format : str
        The desired image format (e.g., 'JPEG', 'PNG').

    Returns
    -------
    Optional[PIL.Image.Image]
        The converted PIL Image object if successful; otherwise, `None`.

    Raises
    ------
    ValueError
        If the target format is unsupported or conversion fails.
    """
    try:
        if target_format.upper() == "JPEG":
            if image.mode in ("RGBA", "LA") or (
                image.mode == "P" and "transparency" in image.info
            ):
                # Convert images with transparency to RGB with white background
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background
            else:
                image = image.convert("RGB")
        else:
            image = (
                image.convert("RGBA")
                if image.mode == "P"
                else image.convert(image.mode)
            )

    except Exception:
        logger.exception("Image conversion failed")
        return None
    else:
        return image


def save_image(
    post_id: str,
    url,
    image: Image.Image,
    output_dir: str,
    target_format: str,
) -> None:
    """
    Save the converted image to the specified directory with a unique filename.

    The filename is generated using an MD5 hash of the original URL to ensure uniqueness.

    Parameters
    ----------
    post_id : str
        The unique identifier of the post associated with the image.
    url : str
        The original URL of the image.
    image : PIL.Image.Image
        The PIL Image object to save.
    output_dir : str
        The directory where the image will be saved.
    target_format : str
        The format in which to save the image (e.g., 'JPEG').

    Returns
    -------
    None

    Raises
    ------
    OSError
        If the image cannot be saved due to filesystem errors.
    """
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filename = f"{post_id}.{target_format.lower()}"
        output_path = os.path.join(output_dir, filename)

        image.save(output_path, target_format.upper())

    except Exception:
        logger.exception(f"Failed to save image from {url}")
        return False
    else:
        logger.info(f"Image saved as {output_path}")
        return True


def process_image(
    post_id: str,
    url: str,
    output_dir: str,
    target_format: str = "JPEG",
    allowed_formats: list[str] | None = None,
) -> None:
    """
    Process a single image URL by downloading, validating, converting, and saving it.

    Parameters
    ----------
    post_id : str
        The unique identifier of the post associated with the image.
    url : str
        The URL of the image to process.
    output_dir : str
        The directory where the processed image will be saved.
    target_format : str, optional
        The desired format to convert the image to (default is 'JPEG').
    allowed_formats : list of str, optional
        A list of allowed image formats for validation (e.g., ['jpeg', 'png', 'gif']).
        If `None`, then ["jpeg", "png", "gif"] is used.

    Returns
    -------
    None
    """
    logger.info(f"Processing {url}")
    content, content_type = download_image(url)

    if allowed_formats is None:
        allowed_formats = ["jpeg", "png", "gif"]

    if allowed_formats and content_type:
        mime_to_format = {
            "image/jpeg": "jpeg",
            "image/png": "png",
            "image/gif": "gif",
        }
        inferred_format = mime_to_format.get(content_type)
        if inferred_format and inferred_format not in allowed_formats:
            logger.error(
                f"MIME type {content_type} not in allowed formats for {url}."
            )
            return

    image = validate_image(content, content_type, allowed_formats)
    if not image:
        logger.error(f"Skipping {url} due to validation failure.")
        return

    converted_image = convert_image(image, target_format)
    if not converted_image:
        logger.error(f"Skipping {url} due to conversion failure.")
        return

    save_image(post_id, url, converted_image, output_dir, target_format)


def process_images_parallel(
    post_ids: pd.Series,
    urls: pd.Series,
    output_dir: str,
    target_format: str = "JPEG",
    allowed_formats: list[str] | None = None,
    max_workers: int = 5,
) -> None:
    """
    Process multiple image URLs in parallel.

    This function utilizes a thread pool to download and process multiple images concurrently.

    Parameters
    ----------
    post_ids : pd.Series
        The unique identifiers of the posts associated with the images.
    urls : pd.Series
        A list of image URLs to process.
    output_dir : str
        The directory where the processed images will be saved.
    target_format : str, optional
        The desired format to convert the images to (default is 'JPEG').
    allowed_formats : list of str, optional
        A list of allowed image formats for validation (e.g., ['jpeg', 'png', 'gif']).
        If `None`, all formats are allowed.
    max_workers : int, optional
        The maximum number of threads to use for parallel processing (default is 5).

    Returns
    -------
    None
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {
            executor.submit(
                process_image,
                post_id,
                url,
                output_dir,
                target_format,
                allowed_formats,
            ): url
            for (post_id, url) in zip(post_ids, urls, strict=False)
        }

        for future in tqdm(
            as_completed(future_to_url),
            total=len(future_to_url),
            desc="Processing images",
        ):
            url = future_to_url[future]
            try:
                future.result()
            except Exception:
                logger.exception(f"Unhandled exception for {url}")


"""
Can improve the speeds using async and aiohttp for asyncronous requests and processing
so that while one image is downloading the same processor can do other operations,
like downloading the next image or processing and validating the previous one.

Would require major refactoring of the code, but could be a good improvement for the future.
Check aiohttp documentation for more information.
"""
