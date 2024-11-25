"""
Labelling Module
==============

This module contains a script to aid in the manual labelling of memes.
"""

import os
import sys
import logging
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

sys.path.append(os.path.abspath("../"))
import utils

utils.logger_init()


def save_csv(df, folder):
    """
    Save the current labelling progress to a CSV file.
    """
    logging.info(f"Exiting.")
    os.makedirs(folder, exist_ok=True)
    df.to_csv(f"{folder}/{folder}_{len(os.listdir(folder))}.csv", index=False)
    save_current(df)
    logging.info("Saved to folder.")


def save_current(df):
    """
    Save the current labelling progress to a CSV file.
    """
    df.to_csv("current_labelling.csv", index=False)
    logging.info("Saved current labelling progress.")


def show_image_in_window(image_path, title):
    """
    Display an image in a window.
    """
    img = mpimg.imread(image_path)
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.show()


def input_level_1():
    """
    Stable vs. Remixed
    """
    print(
        "1 - Stable Image;\n" +
        "2 - Remixed Image;\n" +
        "[3] - Skip;\n" +
        "4 - Exit"
        )
    response = input(
        "1 - Stable Image;\n" +
        "2 - Remixed Image;\n" +
        "[3] - Skip;\n" +
        "4 - Exit"
    )
                    
    if response == "":
        return "Skip"
    response = int(response)
    if response == 1:
        return "Stable"
    elif response == 2:
        return "Remixed"
    elif response == 3 or None:
        return "Skip"
    elif response == 4:
        return "Exit"
    else:
        logging.error("Invalid input, try again.")
        return input_level_1()


def input_level_2(df, id, multiple=False):
    """
    Stable Image Subcategories - Screenshots vs. Text vs. Photos vs. Drawings
    """
    if not multiple:
        print(
            "1 - Screenshot;\n" +
            "2 - Text;\n" +
            "3 - Photo / IRL;\n" +
            "4 - Drawing / Photoshop;\n" +
            "[5] - Skip;\n" +
            "6 - Exit"
            )
        response = input(
            "1 - Screenshot;\n" +
            "2 - Text;\n" +
            "3 - Photo / IRL;\n" +
            "4 - Drawing / Photoshop;\n" +
            "5 - Multiple;\n" +
            "[6] - Skip;\n" +
            "7 - Exit"
        )
    else:
        print(
            "1 - Screenshot;\n" +
            "2 - Text;\n" +
            "3 - Photo / IRL;\n" +
            "4 - Drawing / Photoshop;\n" +
            "0 - Done"
            )
        response = input(
            "1 - Screenshot;\n" +
            "2 - Text;\n" +
            "3 - Photo / IRL;\n" +
            "4 - Drawing / Photoshop;\n" +
            "0 - Done"
        )
    
    if response == "" and not multiple:
        return "Skip"
    response = int(response)

    if response == 1:
        logging.info(f"{id} - Screenshot.")
        df.loc[df["id"]==id, "stable"] = 1
        df.loc[df["id"]==id, "screenshot"] = 1
        df.loc[df["id"]==id, "labelled"] = True
        if multiple:
            df = input_level_2(df, id, multiple=True)
        return df
    elif response == 2:
        logging.info(f"{id} - Text.")
        df.loc[df["id"]==id, "stable"] = 1
        df.loc[df["id"]==id, "text"] = 1
        df.loc[df["id"]==id, "labelled"] = True
        if multiple:    
            df = input_level_2(df, id, multiple=True)
        return df
    elif response == 3:
        logging.info(f"{id} - Photo / IRL.")
        df.loc[df["id"]==id, "stable"] = 1
        df.loc[df["id"]==id, "photo"] = 1
        df.loc[df["id"]==id, "labelled"] = True
        if multiple:    
            df = input_level_2(df, id, multiple=True)
        return df
    elif response == 4:
        logging.info(f"{id} - Drawing / Photoshop.")
        df.loc[df["id"]==id, "stable"] = 1
        df.loc[df["id"]==id, "drawing"] = 1
        df.loc[df["id"]==id, "labelled"] = True
        if multiple:    
            df = input_level_2(df, id, multiple=True)
        return df
    
    elif response == 5:
        return input_level_2(df, id, multiple=True)
    elif response == 0 and multiple:
        return df

    elif response == 6:
        return "Skip"
    elif response == 7:
        return "Exit"
    else:
        logging.error("Invalid input, try again.")
        return input_level_2()


def input_level_3(df, id, multiple=False):
    """
    Remixed Image Subcategories - Comics vs. Reactions vs. Macros vs. 
    Templates vs. Demotivationals vs. Stacked Stills
    """
    if not multiple:
        print(
            "1- Emotional Reaction;\n" +
            "2 - Template;\n" +
            "3 - Event Reaction;\n" +
            "4 - Macro\n" +
            "5 - Situational;\n" +
            "6 - Comic / Stacked Images;\n" +
            "7 - Meme Character;\n" +
            "8 - Multiple;\n"
            "[9] - Skip;\n" +
            "10 - Exit"
            )
        response = input(
            "1- Emotional Reaction;\n" +
            "2 - Template;\n" +
            "3 - Event Reaction;\n" +
            "4 - Macro\n" +
            "5 - Situational;\n" +
            "6 - Comic / Stacked Images;\n" +
            "7 - Meme Character;\n" +
            "8 - Multiple;\n"
            "[9] - Skip;\n" +
            "10 - Exit"
            )
    else:
        print(
            "1- Emotional Reaction;\n" +
            "2 - Template;\n" +
            "3 - Event Reaction;\n" +
            "4 - Macro\n" +
            "5 - Situational;\n" +
            "6 - Comic / Stacked Images;\n" +
            "7 - Meme Character;\n" +
            "0 - Done"
            )
        response = input(
            "1- Emotional Reaction;\n" +
            "2 - Template;\n" +
            "3 - Event Reaction;\n" +
            "4 - Macro\n" +
            "5 - Situational;\n" +
            "6 - Comic / Stacked Images;\n" +
            "7 - Meme Character;\n" +
            "0 - Done"
            )
    if response == "" and not multiple:
        return "Skip"

    response = int(response)
    if response == 1:
        logging.info(f"{id} - Emotional Reaction.")
        df.loc[df["id"]==id, "remixed"] = 1
        df.loc[df["id"]==id, "emotional_reaction"] = 1
        df.loc[df["id"]==id, "labelled"] = True
        if multiple:
            df = input_level_3(df, id, multiple=True)
        return df
    elif response == 2:
        logging.info(f"{id} - Template.")
        df.loc[df["id"]==id, "remixed"] = 1
        df.loc[df["id"]==id, "template"] = 1
        df.loc[df["id"]==id, "labelled"] = True
        if multiple:
            df = input_level_3(df, id, multiple=True)
        return df
    elif response == 3:
        logging.info(f"{id} - Event Reaction.")
        df.loc[df["id"]==id, "remixed"] = 1
        df.loc[df["id"]==id, "event_reaction"] = 1
        df.loc[df["id"]==id, "labelled"] = True
        if multiple:
            df = input_level_3(df, id, multiple=True)
        return df
    elif response == 4:
        logging.info(f"{id} - Macro.")
        df.loc[df["id"]==id, "remixed"] = 1
        df.loc[df["id"]==id, "macro"] = 1
        df.loc[df["id"]==id, "labelled"] = True
        if multiple:
            df = input_level_3(df, id, multiple=True)
        return df
    elif response == 5:
        logging.info(f"{id} - Situational.")
        df.loc[df["id"]==id, "remixed"] = 1
        df.loc[df["id"]==id, "situational"] = 1
        df.loc[df["id"]==id, "labelled"] = True
        if multiple:
            df = input_level_3(df, id, multiple=True)
        return df
    elif response == 6:
        logging.info(f"{id} - Comic / Stacked Images.")
        df.loc[df["id"]==id, "remixed"] = 1
        df.loc[df["id"]==id, "comic"] = 1
        df.loc[df["id"]==id, "labelled"] = True
        if multiple:
            df = input_level_3(df, id, multiple=True)
        return df
    elif response == 7:
        logging.info(f"{id} - Meme Character.")
        df.loc[df["id"]==id, "remixed"] = 1
        df.loc[df["id"]==id, "meme_character"] = 1
        df.loc[df["id"]==id, "labelled"] = True
        if multiple:
            df = input_level_3(df, id, multiple=True)
        return df
    
    elif response == 8:
        return input_level_3(df, id, multiple=True)
    elif response == 0 and multiple:
        return df
    
    elif response == 9:
        return "Skip"
    elif response == 10:
        return "Exit"

    else:
        logging.error("Invalid input, try again.")
        return input_level_3(df, id, multiple=True)


def relabel_image(df, id, image_folder=r"../Data Collection Functions/downloaded_images/sample"):
    """
    Resets a previously labelled image.
    """
    image_path = os.path.join(image_folder, str(id) + '.jpeg')
    base_url = "https://www.reddit.com"
    permalink = base_url + df.loc[df["id"]==id, "permalink"].values[0]
    print("*"*150 + f"\nID {id}; {permalink}\n" + "*"*150)

    show_image_in_window(image_path, f"ID {id}")

    for column in df.columns[-12:-1]:
        if df.loc[df["id"]==id, column].values[0] == 1:
            logging.info(f"{id} is labelled as {column.upper()}.")
            print(f"Enter to reset {column.upper()} label, or anything else to skip.")
            
            response = input(f"1 to reset {column.upper()} label, or anything else to skip.")
            if response == "":
                logging.info(f"{id} - {column.upper()} Label NOT Reset.")
            elif int(response) == 1:
                df.loc[df["id"]==id, column] = 0
                logging.info(f"{id} - {column.upper()} Label Reset.")
            else:
                logging.info(f"{id} - {column.upper()} Label NOT Reset.")

    if all(df.loc[df["id"]==id, df.columns[-12:-1]].values[0] == 0):
        df.loc[df["id"]==id, "labelled"] = False
        df.loc[df["id"]==id, "stable"] = 0
        df.loc[df["id"]==id, "remixed"] = 0
        logging.info(f"{id} - All Labels Reset.")

    save_current(df)
    return df.loc[df["id"]==id, df.columns[-14:]]


def main():
    """
    Main function for the labelling script.
    """
    logging.info("Starting labelling script.")

    df = pd.read_csv('current_labelling.csv')
    image_folder = r"../Data Collection Functions/downloaded_images/sample"
    base_url = "https://www.reddit.com"

    for i, row in enumerate(df.loc[~df["labelled"], ["id", "permalink"]].values):

        if i % 100 == 0 and i != 0:
            save_csv(df, "labelling_saves")
            logging.info("Saved current labelling progress (100 iterations).")

        id, permalink = row
        permalink = base_url + permalink
        print("*"*150 + f"\nIteration {i} - ID {id}; {permalink}\n" + "*"*150)
        image_path = os.path.join(image_folder, str(id) + '.jpeg')

        try:
            show_image_in_window(image_path, f"ID {id}")
        except FileNotFoundError:
            logging.error(f"Image {id} not found.")
            continue
        except Exception as e:
            logging.error(f"Error loading image {id}: {e}")
            continue

        level_1 = input_level_1()
        if level_1 == "Exit":
            logging.info(f"{id} - Exiting.")
            save_csv(df, "labelling_saves")
            logging.info("Saved current labelling progress.")
            break

        elif level_1 == "Stable":
            logging.info(f"{id} - Stable Image.")
            df_2 = input_level_2(df, id)

            if isinstance(df_2, pd.DataFrame):
                df = df_2
                save_current(df)
                continue
            elif df_2 == "Exit":
                save_csv(df, "labelling_saves")
                break
            elif df_2 == "Skip":
                logging.info("{id} - Skipping to next image.")
                continue


        elif level_1 == "Remixed":
            logging.info(f"{id} - Remixed Image.")
            df_2 = input_level_3(df, id)

            if isinstance(df_2, pd.DataFrame):
                df = df_2
                save_current(df)
                continue
            elif df_2 == "Exit":
                save_csv(df, "labelling_saves")
                break
            elif df_2 == "Skip":
                logging.info("{id} - Skipping to next image.")
                continue

        elif level_1 == "Skip":
            logging.info("{id} - Skipping to next image.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(e)
        logging.error("An error occurred, exiting.")
        sys.exit(1)
    else:
        sys.exit(0)