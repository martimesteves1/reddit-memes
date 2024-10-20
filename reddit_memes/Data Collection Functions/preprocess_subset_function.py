import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from sklearn.model_selection import StratifiedShuffleSplit
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def load_jsonl_to_dataframe(jsonl_file_path):
    """
    Load a JSONL file into a pandas DataFrame.

    Parameters:
    jsonl_file_path (str): The file path to the JSONL file.

    Returns:
    pd.DataFrame: The loaded DataFrame.
    """
    try:
        with open(jsonl_file_path, 'r') as f:
            data = [json.loads(line) for line in f]
        df = pd.DataFrame(data)
        logging.info("Data loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Error loading JSONL file: {e}")
        raise

def clean_data(df):
    """
    Clean the DataFrame by dropping columns with all missing values,
    removing video posts, and removing posts with author_cakeday.

    Parameters:
    df (pd.DataFrame): The DataFrame to clean.

    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """
    df.dropna(axis=1, how='all', inplace=True)

    if 'is_video' in df.columns:
        df = df[df['is_video'] == False]
    else:
        logging.warning("Column 'is_video' not found in DataFrame.")

    if 'author_cakeday' in df.columns:
        df = df[df['author_cakeday'].isna()]
    else:
        logging.warning("Column 'author_cakeday' not found in DataFrame.")

    logging.info("Data cleaned successfully.")
    return df

def stratify_data(df):
    """
    Stratify the DataFrame based on number of comments, month of publishing,
    time of publishing, and score.

    Parameters:
    df (pd.DataFrame): The DataFrame to stratify.

    Returns:
    pd.DataFrame: The stratified DataFrame with new columns for stratification.
    """
    # Stratify by number of comments
    if 'num_comments' in df.columns:
        comment_bins = [-np.inf, 0, 10, 50, 100, np.inf]
        comment_labels = ['No comments', 'Very Low [1-10]', 'Medium [11-50]', 'High [51-100]', 'Very High [101+]']
        df['comment_category'] = pd.cut(df['num_comments'], bins=comment_bins, labels=comment_labels)
    else:
        logging.warning("Column 'num_comments' not found in DataFrame.")

    # Stratify by month of publishing
    if 'created_utc' in df.columns:
        df['created_month'] = pd.to_datetime(df['created_utc'], unit='s').dt.month_name()
    else:
        logging.warning("Column 'created_utc' not found in DataFrame.")

    # Stratify by time of publishing
    if 'created_utc' in df.columns:
        df['created_hour'] = pd.to_datetime(df['created_utc'], unit='s').dt.hour
        time_bins = [0, 6, 12, 18, 24]
        time_labels = ['Night [0am-6am]', 'Morning ]6am-12am]', 'Afternoon ]12am-6pm]', 'Evening ]6pm-12pm]']
        df['time_of_day'] = pd.cut(df['created_hour'], bins=time_bins, labels=time_labels, right=False)
    else:
        logging.warning("Column 'created_utc' not found in DataFrame.")

    # Stratify by score
    if 'score' in df.columns:
        score_bins = [-np.inf, 0, 50, 100, 500, np.inf]
        score_labels = ['Zero', 'Low [1-50]', 'Medium [51-100]', 'High [101-500]', 'Very High [501+]']
        df['score_category'] = pd.cut(df['score'], bins=score_bins, labels=score_labels)
    else:
        logging.warning("Column 'score' not found in DataFrame.")

    logging.info("Data stratified successfully.")
    return df

def analyze_distributions(df):
    """
    Analyze the distributions of the stratification categories, plot them, and display the numbers.

    Parameters:
    df (pd.DataFrame): The DataFrame to analyze.

    Returns:
    None
    """
    import matplotlib.pyplot as plt

    # Percentage of posts per month
    if 'created_month' in df.columns:
        monthly_counts = df['created_month'].value_counts()
        monthly_percentage = df['created_month'].value_counts(normalize=True) * 100
        print("Distribution of Posts per Month:")
        print(pd.concat([monthly_counts.rename('Count'), monthly_percentage.rename('Percentage')], axis=1))
        monthly_percentage.sort_index().plot(kind='bar', title='Percentage of Posts per Month')
        plt.ylabel('% of Posts')
        plt.show()
    else:
        logging.warning("Column 'created_month' not found in DataFrame.")

    # Percentage of over_18 posts
    if 'over_18' in df.columns:
        over_18_counts = df['over_18'].value_counts()
        over_18_percentage = df['over_18'].value_counts(normalize=True) * 100
        print("\nDistribution of Over 18 Posts:")
        print(pd.concat([over_18_counts.rename('Count'), over_18_percentage.rename('Percentage')], axis=1))
        over_18_percentage.plot(kind='bar', title='Percentage of Over 18 Posts')
        plt.ylabel('% of Posts')
        plt.xticks(rotation=0)
        plt.show()
    else:
        logging.warning("Column 'over_18' not found in DataFrame.")

    # Percentage of posts with number of comments
    if 'comment_category' in df.columns:
        comment_counts = df['comment_category'].value_counts()
        comment_percentage = df['comment_category'].value_counts(normalize=True) * 100
        print("\nDistribution of Posts by Number of Comments:")
        print(pd.concat([comment_counts.rename('Count'), comment_percentage.rename('Percentage')], axis=1))
        comment_percentage.plot(kind='bar', title='Percentage of Posts by Number of Comments')
        plt.ylabel('% of Posts')
        plt.xticks(rotation=0)
        plt.show()
    else:
        logging.warning("Column 'comment_category' not found in DataFrame.")

    # Percentage of posts for each time of day
    if 'time_of_day' in df.columns:
        time_counts = df['time_of_day'].value_counts()
        time_percentage = df['time_of_day'].value_counts(normalize=True) * 100
        print("\nDistribution of Posts by Time of Day:")
        print(pd.concat([time_counts.rename('Count'), time_percentage.rename('Percentage')], axis=1))
        time_percentage.plot(kind='bar', title='Percentage of Posts by Time of Day')
        plt.ylabel('% of Posts')
        plt.xticks(rotation=0)
        plt.show()
    else:
        logging.warning("Column 'time_of_day' not found in DataFrame.")

    # Percentage of posts by score
    if 'score_category' in df.columns:
        score_counts = df['score_category'].value_counts()
        score_percentage = df['score_category'].value_counts(normalize=True) * 100
        print("\nDistribution of Posts by Score:")
        print(pd.concat([score_counts.rename('Count'), score_percentage.rename('Percentage')], axis=1))
        score_percentage.plot(kind='bar', title='Percentage of Posts by Score')
        plt.ylabel('% of Posts')
        plt.xticks(rotation=0)
        plt.show()
    else:
        logging.warning("Column 'score_category' not found in DataFrame.")

    logging.info("Data distributions analyzed.")


def create_subset(df):
    """
    Create a subset of the DataFrame by sampling 20% of the original data
    while maintaining the distribution of stratification categories.
    Strata with fewer than two samples are removed.

    Parameters:
    df (pd.DataFrame): The DataFrame to sample from.

    Returns:
    pd.DataFrame: The subset DataFrame.
    """
    try:
        # Create a stratification key
        strat_cols = ['created_month', 'over_18', 'comment_category', 'time_of_day', 'score_category']
        missing_cols = [col for col in strat_cols if col not in df.columns]
        if missing_cols:
            logging.error(f"Missing columns for stratification: {missing_cols}")
            raise KeyError(f"Missing columns for stratification: {missing_cols}")

        df['strata'] = df[strat_cols].astype(str).agg('-'.join, axis=1)
        df = df.dropna(subset=['strata']).copy()

        # Check strata counts before splitting
        strata_counts = df['strata'].value_counts()
        logging.info("Strata counts before filtering:")
        logging.info(strata_counts)

        # Remove strata with fewer than 2 samples
        valid_strata = strata_counts[strata_counts >= 2].index
        df = df[df['strata'].isin(valid_strata)].copy()

        # Check strata counts after filtering
        strata_counts = df['strata'].value_counts()
        logging.info("Strata counts after filtering:")
        logging.info(strata_counts)

        # Proceed with StratifiedShuffleSplit
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for _, subset_index in split.split(df, df['strata']):
            subset_df = df.iloc[subset_index].copy()

        subset_df.drop('strata', axis=1, inplace=True)
        logging.info("Subset created successfully after removing strata with fewer than 2 samples.")
        return subset_df
    except Exception as e:
        logging.error(f"Error creating subset: {e}")
        raise

def main(jsonl_file_path):
    """
    Execute the data loading, cleaning, stratification, analysis, and subset creation.

    Parameters:
    jsonl_file_path (str): The file path to the JSONL dataset.

    Returns:
    None
    """
    # Load dataset
    df = load_jsonl_to_dataframe(jsonl_file_path)

    # Clean data
    df_cleaned = clean_data(df)

    # Stratify data
    df_stratified = stratify_data(df_cleaned)

    # Analyze distributions
    analyze_distributions(df_stratified)

    # Create subset
    subset_df = create_subset(df_stratified)

    # Save the subset to a CSV file
    subset_file_path = os.path.splitext(jsonl_file_path)[0] + '_subset.csv'
    subset_df.to_csv(subset_file_path, index=False)
    logging.info(f"Subset saved to {subset_file_path}")

if __name__ == "__main__":
    jsonl_file_path = "your_dataset.jsonl"  # Replace with your JSONL file path
    main(jsonl_file_path)
