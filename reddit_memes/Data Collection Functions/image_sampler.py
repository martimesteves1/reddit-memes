"""
Image Sample Creator Module
===========================

This module provides functionalities to create a random sample of images from a folder containing
a collection of images. The user can specify the number of images to sample and the output folder
where the sampled images will be saved. The module also validates the images in the input folder
and ensures that only valid JPEG images are sampled. If the output folder already exists and has
content, the user will be prompted to confirm the deletion of the existing content.

Usage
-----
The module can be used as a standalone script or imported as a module. If used as a script, the
script assumes the input folder to be `downloaded_images/r-memes/2023` and the output folder to be
`downloaded_images/sample`, and a sample size of 2000. This can be altered in the future by using
argparse in the main block.

Example
-------
```python
from image_sampler import sample_images

input_folder = "downloaded_images/r-memes/2023"
output_folder = "downloaded_images/sample"
sample_size = 2000

sample_images(input_folder, output_folder, sample_size)
```
"""

import logging
import os
import random
import shutil
import signal

from PIL import Image

random.seed(42)
Image.MAX_IMAGE_PIXELS = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        # logging.FileHandler("image_sampler.log"),  # Log to a file
        logging.StreamHandler()  # Log to the console
    ],
)


def timeout_handler(signum, frame):
    """Handler function to raise a TimeoutError on signal."""
    raise TimeoutError


def get_user_input(prompt: str, timeout: int, default_value: str) -> str:
    """
    Gets user input with a timeout. Returns 'y' if the user doesn't
    respond in time.

    Parameters
    ----------
    prompt : str
        The prompt to display to the user.
    timeout : int
        The number of seconds before the input times out.
    default_value : str
        The default value to return if the input times out.

    Returns
    -------
    str
        The user input or the default value if timeout occurs.
    """
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)

    try:
        user_input = input(prompt)
        signal.alarm(0)

    except TimeoutError:
        logging.warning(
            f"Input timed out. Using default value: {default_value}"
        )
        return default_value

    else:
        return user_input


def validate_image(file_path: str) -> bool:
    """
    Validates if the given image is a JPEG and can be opened.

    Parameters
    ----------
    file_path : str
        The path to the image file.

    Returns
    -------
    bool
        True if the image is a valid JPEG and can be opened, False otherwise.
    """
    try:
        with Image.open(file_path) as img:
            img.verify()
            if img.format != "JPEG":
                logging.warning(f"Image '{file_path}' is not a JPEG.")
                return False

    except (OSError, SyntaxError) as e:
        logging.warning(
            f"Image '{file_path}' could not be opened or is invalid: {e}"
        )
        return False

    else:
        return True


def sample_images(
    input_folder: str, output_folder: str, sample_size: int
) -> None:
    """
    Creates a random sample of images from the input folder and copies
    them to the output folder. If the output folder already exists and
    as content, the user will be prompted to confirm folder deletion.

    Parameters
    ----------
    input_folder : str
        The path to the folder containing the original images.
    output_folder : str
        The path to the folder where the sampled images will be copied.
    sample_size : int
        The number of images to sample.

    Raises
    ------
    ValueError
        If the sample size is greater than the total number of images in the input folder.
    FileNotFoundError
        If the input folder does not exist.
    """
    logging.info(
        f"Starting to sample images from '{input_folder}' to '{output_folder}' with sample size {sample_size}."
    )

    if not os.path.exists(input_folder):
        logging.error(f"The input folder '{input_folder}' does not exist.")
        raise FileNotFoundError

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        logging.debug(f"Created the output folder '{output_folder}'.")

    else:
        if len(os.listdir(output_folder)) > 0:
            logging.warning(
                f"The output folder '{output_folder}' already has content."
            )
            user_input = get_user_input(
                "Do you want to delete the existing content in the output folder? ([y]/n): ",
                20,
                "y",
            )
            if user_input.lower() == "y" or user_input == "":
                shutil.rmtree(output_folder)
                os.makedirs(output_folder)
                logging.debug(
                    f"Deleted the existing content in the output folder '{output_folder}'."
                )
            else:
                logging.error("Aborted image sampling.")
                return

    image_files = os.listdir(input_folder)
    logging.info(f"Found {len(image_files)} images in the input folder.")

    valid_jpeg_images = [
        f for f in image_files if validate_image(os.path.join(input_folder, f))
    ]
    logging.info(
        f"Found {len(valid_jpeg_images)} \
          ({round(len(valid_jpeg_images) / len(image_files), 2) * 100}%) valid JPEG images."
    )

    if sample_size > len(valid_jpeg_images):
        logging.error(
            "Sample size cannot be greater than the total number of valid JPEG images in the input folder."
        )
        raise ValueError()

    sampled_images = random.sample(valid_jpeg_images, sample_size)
    logging.debug(f"Sampled images: {sampled_images}")

    for image in sampled_images:
        shutil.copy(
            os.path.join(input_folder, image),
            os.path.join(output_folder, image),
        )
        logging.info(f"Copied image: {image}")

    logging.info("Image sampling completed successfully.")


if __name__ == "__main__":
    input_path = os.path.join(
        os.path.dirname(__file__), "downloaded_images/r-memes/2023"
    )
    output_path = os.path.join(
        os.path.dirname(__file__), "downloaded_images/sample"
    )
    sample_images(input_path, output_path, 2000)
