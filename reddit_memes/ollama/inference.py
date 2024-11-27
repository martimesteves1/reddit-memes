"""
Inference Module
==============

This module contains a script to aid in the inference task using Ollama LVMs.
"""

import os
import logging
import traceback
from tqdm import tqdm
import ollama
import pandas as pd
import re
from collections import Counter


def infer(model: str, prompt: str, image_path: str, **options) -> dict:
    """
    Infer the prompt using the given model and image.

    Args:
        model (str): The path to the model file.
        prompt (str): The prompt to infer.
        image_path (str): The path to the image.
        **options: Additional options for the inference.
            - repeat_last_n
            - num_ctx
            - repeat_penalty
            - temperature
            - num_predict
            - others... https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
    Returns:
        dict: The inference result.
            - model
            - label
            - total_duration
            - options
    """
    res = ollama.chat(
        model = model,
        messages=[
        {
            "role": "user",
            "content": prompt,
            "images": [image_path]
        }
    ],
        **options
    )
    result = {
        "model": model,
        "label": res["message"]["content"],
        "total_duration": res["total_duration"] / 1_000_000_000,
        "options": {**options}
    }
    return result


def save_inference_results(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the inference results to a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame containing the inference results.
        output_path (str): The path to save the results.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pass


def label_validator(label: str) -> str:
    """
    Validates the labels inferred by the model, correcting cases where the
    model deviates from the expected labels.
    """
    label = label.lower().replace(" ", "")

    if label in ["screenshot", "text", "photo", "drawing", "emotional_reaction",
                 "template", "event_reaction", "macro", "situational", "comic",
                 "meme_character"]:
        return label

    patterns = {
        "screenshot": r".*screen.*",
        "text": r".*text.*",
        "photo": r".*photo.*",
        "drawing": r".*draw.*",
        "emotional_reaction": r".*emotio.*",
        "template": r".*temp.*",
        "event_reaction": r".*event.*",
        "macro": r".*macro.*",
        "situational": r".*situatio.*",
        "comic": r".*comic.*",
        "meme_character": r".*charact.*"
    }
    for key, value in patterns.items():
        if key == "photo":
            if re.match(".*photoshop.*", label):
                return "drawing"
        if re.match(value, label):
            for subkey, subvalue in patterns.items():
                if subkey == key:
                    continue
                if re.match(subvalue, label):
                    return "none"
            return key

    numbers = {
        "screenshot": "1",
        "text": "2",
        "photo": "3",
        "drawing": "4",
        "emotional_reaction": "5",
        "event_reaction": "6",
        "macro": "7",
        "situational": "8",
        "comic": "9",
        "meme_character": "10",
        "template": "11"
    }
    for key, value in numbers.items():
        if value == label:
            return key

    return "none"


def infer_from_df(df: pd.DataFrame, model: str, prompt: str, image_dir: str, output_path: str, **options) -> pd.DataFrame:
    """
    Infer the prompts in the DataFrame using the given model and images.

    Args:
        df (pd.DataFrame): The DataFrame containing the batch information.
        model (str): The path to the model file.
        prompt (str): The prompt passed to the model.
        image_dir (str): The directory containing the images.
        output_path (str): The path to save the results.
        **options: Additional options for the inference.

    Returns:
        pd.DataFrame: The DataFrame with the inference results.
            "id", "total_duration", "label", " label_clean", "meta", "stable", 
            "remixed", "labelled""stable", "remixed", "screenshot", "text", "photo",
            "drawing", "emotional_reaction", "template", "event_reaction",
            "macro", "situational", "comic", "meme_character", "labelled"
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_labelled = pd.DataFrame(columns=["id",
                                        "total_duration",
                                        "label",
                                        "label_clean",
                                        "meta",
                                        "stable",
                                        "remixed",
                                        "labelled"])
    
    stable_list = ["screenshot", "text", "photo", "drawing"]
    remixed_list = ["emotional_reaction", "template", "event_reaction", "macro",
                    "situational", "comic", "meme_character"]
    all_labels = stable_list + remixed_list + ["none"]
    
    for i, id in enumerate(tqdm(df["id"], desc="Infering prompts", unit="image"), start=1):
        image_path = os.path.join(image_dir, f"{id}.jpeg")
        try:
            result = infer(model, prompt, image_path, **options)
        except:
            logging.error(f"Failed to infer prompt for image: {id}")
            logging.debug(traceback.format_exc())
            continue
        
        result_formatted = {
            "id": id,
            "total_duration": result["total_duration"],
            "label": result.get("label", "None"),
            "label_clean": label_validator(result.get("label", "None")),
            "meta": {
                "prompt": prompt,
                "options": {**options}
            },
            "labelled": True,
            "stable": 0,
            "remixed": 0
        }
        df_labelled.loc[len(df_labelled)] = result_formatted

        if i % 1000 == 0:
            df_labelled.to_csv(output_path.replace(".csv", f"_partial_{i}.csv"), index=False)

    dummies = pd.get_dummies(df_labelled["label_clean"])
    dummies = dummies.reindex(columns=all_labels, fill_value=False)
    df_labelled = pd.concat([df_labelled, dummies], axis=1)

    df_labelled["stable"] = df_labelled["label_clean"].apply(lambda x: True if x in stable_list else False)
    df_labelled["remixed"] = df_labelled["label_clean"].apply(lambda x: True if x in remixed_list else False)
    
    df_labelled.to_csv(output_path, index=False)

    logging.info(f"Saved labelled DataFrame to: {output_path} ({df_labelled.shape[0]} rows)")

    return df_labelled
    

def infer_n_from_df(df: pd.DataFrame, model: str, prompt: str, image_dir: str, output_path: str, n_infer: int, **options) -> pd.DataFrame:
    """
    Infer the prompts in the DataFrame using the given model and images.
    Inference now runs 10 times per image and the most common label is selected.

    Args:
        df (pd.DataFrame): The DataFrame containing the batch information.
        model (str): The path to the model file.
        prompt (str): The prompt passed to the model.
        image_dir (str): The directory containing the images.
        output_path (str): The path to save the results.
        n_infer (int): The number of inferences to run per image.
        **options: Additional options for the inference.

    Returns:
        pd.DataFrame: The DataFrame with the inference results.
            "id", "total_duration", "label", " label_clean", "meta", "stable", 
            "remixed", "labelled""stable", "remixed", "screenshot", "text", "photo",
            "drawing", "emotional_reaction", "template", "event_reaction",
            "macro", "situational", "comic", "meme_character", "labelled"
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_labelled = pd.DataFrame(columns=["id",
                                        "total_duration",
                                        "label",
                                        "label_clean",
                                        "label_common",
                                        "meta",
                                        "stable",
                                        "remixed",
                                        "labelled"])
    
    stable_list = ["screenshot", "text", "photo", "drawing"]
    remixed_list = ["emotional_reaction", "template", "event_reaction", "macro",
                    "situational", "comic", "meme_character"]
    all_labels = stable_list + remixed_list + ["none"]
    
    for i, id in enumerate(tqdm(df["id"], desc="Infering prompts", unit="image"), start=1):
        image_path = os.path.join(image_dir, f"{id}.jpeg")
        results = []
        for n in range(n_infer):
            try:
                results.append(infer(model, prompt, image_path, **options))
            except:
                logging.error(f"Failed to infer prompt for image: {id}")
                logging.debug(traceback.format_exc())
                continue
        
        try:
            labels = [result["label"] for result in results]
            labels_clean = [label_validator(label) for label in labels]
            total_duration = sum([result["total_duration"] for result in results])
            label_common = Counter([label for label in labels_clean if label != "none"]).most_common(1)[0][0]
        except:
            logging.error(f"Unexpected Inference Result: {id}")
            logging.debug(traceback.format_exc())
            continue

        result_formatted = {
            "id": id,
            "total_duration": total_duration,
            "label": labels,
            "label_clean": labels_clean,
            "label_common": label_common,
            "meta": {
                "prompt": prompt,
                "n_infer": len(results),
                "options": {**options}
            },
            "labelled": True,
            "stable": 0,
            "remixed": 0
        }
        df_labelled.loc[len(df_labelled)] = result_formatted

        if i % 1000 == 0:
            df_labelled.to_csv(output_path.replace(".csv", f"_partial_{i}.csv"), index=False)

    dummies = pd.get_dummies(df_labelled["label_clean"])
    dummies = dummies.reindex(columns=all_labels, fill_value=False)
    df_labelled = pd.concat([df_labelled, dummies], axis=1)

    df_labelled["stable"] = df_labelled["label_clean"].apply(lambda x: True if x in stable_list else False)
    df_labelled["remixed"] = df_labelled["label_clean"].apply(lambda x: True if x in remixed_list else False)
    
    df_labelled.to_csv(output_path, index=False)

    logging.info(f"Saved labelled DataFrame to: {output_path} ({df_labelled.shape[0]} rows)")

    return df_labelled
"""

def load_data(manual_csv: str, model_csv: str) -> tuple[pd.DataFrame, pd.DataFrame]:


    Args:
        manual_csv (str): The path to the manual CSV file.
        model_csv (str): The path to the model CSV file.

    Returns:
        pd.DataFrame: The manual DataFrame.
        pd.DataFrame: The model DataFrame.
    try:
        manual_df = pd.read_csv(manual_csv)
    except:
        logging.error(f"Failed to load manual CSV: {manual_csv}")
        logging.debug(traceback.format_exc())
        raise
    try:
        model_df = pd.read_csv(model_csv)
    except:
        logging.error(f"Failed to load model CSV: {model_csv}")
        logging.debug(traceback.format_exc())
        raise
    logging.info("Loaded Data Successfully")

    return manual_df, model_df


def calculate_accuracy(manual_df, model_df, exclude_none=False):

    manual_df = manual_df.set_index('entry_id')
    model_df = model_df.set_index('entry_id')
    
    if exclude_none:
        model_df = model_df[model_df.ne('none').any(axis=1)]
    
    matched = manual_df & model_df
    accuracy = matched.sum().sum() / manual_df.sum().sum()
    
    return accuracy

def calculate_percentage_of_nones(model_df):
    none_count = model_df.isin(['none']).sum(axis=1).sum()
    total_entries = model_df.shape[0]
    return (none_count / total_entries) * 100

def visualize_results(manual_df, model_df, exclude_none=True):
    
    Visualize the results with and without 'none' entries.
    
    manual_df = manual_df.set_index('entry_id')
    model_df = model_df.set_index('entry_id')

    if exclude_none:
        model_df = model_df[model_df.ne('none').any(axis=1)]
    
    matched = manual_df & model_df
    category_match = matched.sum(axis=0)
    category_total = manual_df.sum(axis=0)
    accuracy_per_category = (category_match / category_total).fillna(0)

    # Plot accuracy per category
    plt.figure(figsize=(10, 6))
    sns.barplot(x=accuracy_per_category.index, y=accuracy_per_category.values)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Per Category')
    plt.show()

def main(manual_csv, model_csv):
    # Load data
    manual_df, model_df = load_data(manual_csv, model_csv)

    # Calculate and report overall accuracy
    accuracy_with_none = calculate_accuracy(manual_df, model_df, exclude_none=False)
    accuracy_without_none = calculate_accuracy(manual_df, model_df, exclude_none=True)
    print(f'Accuracy with None: {accuracy_with_none:.2%}')
    print(f'Accuracy without None: {accuracy_without_none:.2%}')

    # Calculate and report percentage of None entries
    none_percentage = calculate_percentage_of_nones(model_df)
    print(f'Percentage of None entries: {none_percentage:.2f}%')

    # Visualize results
    visualize_results(manual_df, model_df, exclude_none=True)

# Replace with your file paths
manual_csv_path = 'path_to_manual_labels.csv'
model_csv_path = 'path_to_model_inferences.csv'
main(manual_csv_path, model_csv_path)
"""