import os
import urllib.parse

import pandas as pd
import requests
from tqdm import tqdm


def download_images(csv_file, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)

    # Function to get file extension from the response headers
    def get_extension_from_response(response):
        content_type = response.headers.get("content-type")
        if content_type and "image" in content_type:
            ext = content_type.split("/")[-1]
            return "." + ext
        return ".jpg"  # Default extension if none is found

    # Loop over the URLs and download the images
    for idx, url in enumerate(tqdm(df["url"])):
        try:
            response = requests.get(url, stream=True, timeout=10)
            if response.status_code == 200:
                # Extract filename from URL
                parsed_url = urllib.parse.urlparse(url)
                filename = os.path.basename(parsed_url.path)

                # If no filename is found, create one
                if not filename or "." not in filename:
                    ext = get_extension_from_response(response)
                    filename = f"image_{idx}{ext}"

                # Sanitize filename to remove invalid characters
                filename = "".join(
                    c for c in filename if c.isalnum() or c in (" ", ".", "_")
                ).rstrip()

                file_path = os.path.join(output_folder, filename)

                # Write the content to a file in chunks
                with open(file_path, "wb") as out_file:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            out_file.write(chunk)

                print(f"Downloaded {url} as {filename}")
            else:
                print(
                    f"Failed to download {url} (Status code: {response.status_code})"
                )
        except Exception as e:
            print(f"Error downloading {url}: {e}")
