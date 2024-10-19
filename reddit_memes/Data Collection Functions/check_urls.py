import os
import urllib.parse
import pandas as pd
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def check_url_status(csv_file):
    # Read the CSV file into a pandas DataFrame
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Check if 'url' column exists
    if 'url' not in df.columns:
        raise ValueError("The provided CSV file does not contain a 'url' column.")

    # Create a new column to store URL status
    df['status'] = ''

    # Use a Session object for connection pooling
    session = requests.Session()

    # Function to get URL status
    def get_url_status(url):
        try:
            time.sleep(0.1)  # Delay to reduce network load
            response = session.get(url, stream=True, timeout=10)
            response.close()  # Close the connection

            # Check if the URL is accessible and if it points to an image
            if response.status_code == 200:
                content_type = response.headers.get("Content-Type", "").lower()

                # Check if the content type is an image and not a GIF
                if "image" in content_type and "gif" not in content_type:
                    return "OK"
                else:
                    return "NOTOK"
            else:
                return f"NOTOK (Status code: {response.status_code})"
        except requests.RequestException as e:
            return f"NOTOK (Request error: {e})"
        except Exception as e:
            return "NOTOK (An unexpected error occurred)"

    # Use ThreadPoolExecutor to check URLs concurrently
    max_workers = 30
    urls = df['url'].tolist()
    statuses = [''] * len(urls)  # Placeholder for statuses

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(get_url_status, url): idx for idx, url in enumerate(urls)}

        for future in tqdm(as_completed(future_to_index), total=len(urls), desc="Checking URLs"):
            idx = future_to_index[future]
            try:
                status = future.result()
            except Exception as exc:
                status = f"NOTOK (Exception: {exc})"
            statuses[idx] = status

    # Assign the statuses back to the DataFrame
    df['status'] = statuses

    # Save the DataFrame with URL status to a new CSV file
    new_csv_file = os.path.join(os.path.dirname(os.path.abspath(csv_file)), 'url_status.csv')
    df.to_csv(new_csv_file, index=False)

    print(f"URL status saved to {new_csv_file}")
