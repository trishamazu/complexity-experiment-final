import os
import pandas as pd
import numpy as np
from scipy.stats import entropy
import csv

def order_embedding_file(input_csv):
    """Sort rows alphabetically by the image column and zero out negative values."""
    df = pd.read_csv(input_csv)
    df.iloc[:, 1:] = df.iloc[:, 1:].applymap(lambda x: max(x, 0))  # Zero out negatives
    df = df.sort_values(by='image')  # Sort alphabetically by 'image'
    output_csv = os.path.join(os.path.dirname(input_csv), f"ordered_{os.path.basename(input_csv)}")
    df.to_csv(output_csv, index=False)
    return output_csv

def calculate_entropy(ordered_csv):
    """Normalize rows to probability distribution and calculate entropy."""
    df = pd.read_csv(ordered_csv)
    for index, row in df.iterrows():
        row_values = row.values[1:]
        row_sum = np.sum(row_values)
        if row_sum > 0:
            probability_distribution = row_values / row_sum
            df.iloc[index, 1:] = probability_distribution
    df['Entropy'] = df.iloc[:, 1:].apply(lambda x: entropy(x), axis=1)
    entropy_csv = os.path.join(os.path.dirname(ordered_csv), f"entropy_{os.path.basename(ordered_csv)}")
    df[['image', 'Entropy']].to_csv(entropy_csv, index=False)
    return entropy_csv

def filter_and_normalize_complexity(ordered_csv, experiment_csv):
    """Filter ordered file by experiment images and normalize complexity scores."""
    df1 = pd.read_csv(ordered_csv)
    df2 = pd.read_csv(experiment_csv)
    if 'image' in df1.columns and 'image' in df2.columns:
        set_of_image_names_df2 = set(df2['image'])
        filtered_df1 = df1[df1['image'].isin(set_of_image_names_df2)]
        third_ordering_cols = [col for col in df2.columns if col.endswith('Third Ordering')]
        df2['Average Third Ordering'] = df2[third_ordering_cols].mean(axis=1)
        df2['Normalized Average'] = 1 - ((df2['Average Third Ordering'] - 1) / (50 - 1))
        return filtered_df1, df2[['image', 'Normalized Average']]
    else:
        raise ValueError("'image' column missing in one or both datasets.")

def multiply_scores(filtered_df, normalized_df, output_csv):
    """Multiply normalized complexity scores by embedding values."""
    normalized_dict = normalized_df.set_index('image')['Normalized Average'].to_dict()
    for index, row in filtered_df.iterrows():
        image_name = row['image']
        norm_value = normalized_dict.get(image_name, 1)
        for col in filtered_df.columns[1:]:  # Exclude 'image' column
            filtered_df.at[index, col] *= norm_value
    filtered_df.to_csv(output_csv, index=False)

def assign_weights(filtered_df, output_csv):
    """Calculate weights based on distance from the average."""
    columns_to_sum = [str(i) for i in range(49)]
    sums = filtered_df[columns_to_sum].sum()
    total_sum = sums.sum()
    probability_distribution = sums / total_sum
    average_probability = probability_distribution.mean()
    max_distance = abs(probability_distribution - average_probability).max()
    weights = (probability_distribution - average_probability) / max_distance
    weight_df = pd.DataFrame({'Weight': weights.values})
    weight_df.to_csv(output_csv, index=False)
    return output_csv

def main():
    input_csv = input("Enter the path to the input CSV file: ").strip()
    experiment_csv = '/home/wallacelab/complexity-final/ExperimentData/filtered_hebart_ranking_results.csv'
    
    # Step 1: Order and zero negative values
    ordered_csv = order_embedding_file(input_csv)
    print(f"Ordered file saved as: {ordered_csv}")
    
    # Step 2: Calculate entropy
    entropy_csv = calculate_entropy(ordered_csv)
    print(f"Entropy file saved as: {entropy_csv}")
    
    # Step 3: Filter and normalize complexity scores
    filtered_df, normalized_df = filter_and_normalize_complexity(ordered_csv, experiment_csv)
    filtered_csv = os.path.join(os.path.dirname(input_csv), f"filtered_{os.path.basename(input_csv)}")
    filtered_df.to_csv(filtered_csv, index=False)
    print(f"Filtered file saved as: {filtered_csv}")
    
    # Step 4: Multiply normalized complexity scores
    multiplied_csv = os.path.join(os.path.dirname(input_csv), f"multiplied_{os.path.basename(input_csv)}")
    multiply_scores(filtered_df, normalized_df, multiplied_csv)
    print(f"Multiplied file saved as: {multiplied_csv}")
    
    # Step 5: Assign weights
    weights_csv = os.path.join(os.path.dirname(input_csv), f"weights_{os.path.basename(input_csv)}")
    assign_weights(filtered_df, weights_csv)
    print(f"Weights file saved as: {weights_csv}")

if __name__ == "__main__":
    main()