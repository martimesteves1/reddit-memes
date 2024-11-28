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
        messages = [
            {
                "role": "user",
                "content": prompt,
                "images": [image_path]
            }
        ],
        options = {**options}
    )
    result = {
        "model": model,
        "label": res["message"]["content"],
        "total_duration": res["total_duration"] / 1_000_000_000,
        "options": {**options}
    }
    return result


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
        
        else:
            try:
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
            except:
                logging.error(f"Unexpected Inference Result: {id}")
                logging.debug(traceback.format_exc())
                continue

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
        for _ in range(n_infer):
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
        else:
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


def calculate_match(df_manual, df_model, exclude_none=False) -> pd.DataFrame:
    """
    Calculate the label matches of the model against the manual labels.

    Args:
        df_manual (pd.DataFrame): The DataFrame containing the manual labels.
        df_model (pd.DataFrame): The DataFrame containing the model labels.
        exclude_none (bool): Whether to exclude 'none' entries.
    
    Returns:
        df_match (pd.DataFrame): The DataFrame containing the matches.
    """
    labels_1 = ["stable", "remixed"]
    labels_2 = ["screenshot", "text", "photo", "drawing", "emotional_reaction",
              "template", "event_reaction", "macro", "situational", "comic",
              "meme_character"]
    all_labels = labels_1 + labels_2

    if exclude_none:
        df_model = df_model[~df_model["none"]]

    df_model = df_model.set_index("id")[all_labels]
    df_manual = df_manual[df_manual["id"].isin(df_model.index)]
    df_manual = df_manual.set_index("id")[all_labels]

    df_match = df_manual & df_model
    
    return df_match, df_manual


def calculate_percentage_of_nones(df_model) -> tuple[int, float]:
    """
    Calculate the percentage of labels that are 'none'
    """
    none_count = len(df_model[df_model["none"]])
    total_entries = len((df_model))

    return none_count, none_count / total_entries


def evaluate_results(df_manual, df_model, exclude_none=False) -> dict:
    """
    Computes the accuracy of the infered labels, for the first level
    labels (stable + remixed), for the second level labels (screenshot,
    text, ...) and for each label individually.
    """
    labels_1 = ["stable", "remixed"]
    labels_2 = ["screenshot", "text", "photo", "drawing", "emotional_reaction",
              "template", "event_reaction", "macro", "situational", "comic",
              "meme_character"]
    all_labels = labels_1 + labels_2

    df_match, df_manual = calculate_match(df_manual, df_model, exclude_none)
    result = {}

    result["accuracy_1"] = df_match[labels_1].sum().sum() / len(df_manual)
    result["accuracy_2"] = df_match[labels_2].sum().sum() / len(df_manual)
    for label in all_labels:
        try:
            result[f"accuracy_{label}"] = df_match[label].sum() / df_manual[label].sum()
        except:
            result[f"accuracy_{label}"] = None

    n_nones, f_nones = calculate_percentage_of_nones(df_model)
    result["n_nones"] = n_nones
    result["f_nones"] = f_nones
    
    return result


def visualize_results(*results) -> None:
    """
    Visualizes the results of the evaluation of multiple models.
    """
    pass