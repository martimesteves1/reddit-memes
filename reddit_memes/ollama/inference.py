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
    result = {"model": model, "label": res["message"]["content"], "total_duration": res["total_duration"], "options": {**options}}
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
        'stable', 'remixed', 'screenshot', 'text', 'photo',
       'drawing', 'emotional_reaction', 'template', 'event_reaction', 'macro',
       'situational', 'comic', 'meme_character', 'labelled'
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_labelled = pd.DataFrame(columns=["id",
                                        "total_duration",
                                        "label",
                                        "meta",
                                        "stable",
                                        "remixed",
                                        "labelled"])
    
    stable_list = ["screenshot", "text", "photo", "drawing"]
    remixed_list = ["emotional_reaction", "template", "event_reaction", "macro",
                    "situational", "comic", "meme_character"]
    
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

    df_labelled = pd.concat([df_labelled, pd.get_dummies(df_labelled["label"])], axis=1)
    df_labelled["stable"] = df_labelled["label"].apply(lambda x: 1 if x in stable_list else 0)
    df_labelled["remixed"] = df_labelled["label"].apply(lambda x: 1 if x in remixed_list else 0)
    
    df_labelled.to_csv(output_path, index=False)

    logging.info(f"Saved labelled DataFrame to: {output_path} ({df_labelled.shape[0]} rows)")

    return df_labelled


