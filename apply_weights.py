import os
import pandas as pd
import numpy as np
from scipy.stats import entropy, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

def zero_negative_values(df, columns):
    """Zero out negative values in specified columns."""
    for col in columns:
        df[col] = df[col].clip(lower=0)
    return df

def calculate_entropy(df, columns):
    """Calculate entropy for each row in the specified columns."""
    df = df.copy()  # Avoid SettingWithCopyWarning
    for index, row in df.iterrows():
        row_values = row[columns].values
        sum_of_row = np.sum(row_values)
        if sum_of_row > 0:
            probability_distribution = row_values / sum_of_row
            for i, col in enumerate(columns):
                df.at[index, col] = probability_distribution[i]

    df['Entropy'] = df[columns].apply(lambda x: entropy(x), axis=1)
    return df[['image', 'Entropy']]

def compute_dot_products(matrix_df, weights):
    """Compute dot products of weights with matrix."""
    # Select only numeric columns (0-48)
    matrix = matrix_df.loc[:, [str(i) for i in range(49)]].values
    dot_products = np.dot(matrix, weights)
    return dot_products

def calculate_pairwise_differences(df, columns_to_diff, image_column='image'):
    """
    Calculate pairwise differences for paired images based on specified columns.
    The 'image' column is preserved for reference.
    """
    # Initialize a list to store differences
    diffs = []

    # Iterate over rows in pairs
    for i in range(0, len(df), 2):
        if i + 1 < len(df):  # Ensure there is a next row
            row1 = df.iloc[i]
            row2 = df.iloc[i + 1]

            # Determine base and comparison rows based on naming convention
            if '_c' in row1[image_column]:
                base_row, comp_row = row2, row1
            else:
                base_row, comp_row = row1, row2

            # Calculate differences for the specified columns
            diff = {col: base_row[col] - comp_row[col] for col in columns_to_diff}
            diff[image_column] = base_row[image_column].replace('_c', '')  # Retain base image name

            # Append the differences to the list
            diffs.append(diff)

    # Convert differences to a DataFrame
    return pd.DataFrame(diffs)

def calculate_spearman_correlation(data, x_col, y_col, save_path):
    """
    Calculate Spearman correlation between two difference columns and plot the results.
    """
    # Remove NaN values (if any)
    data = data.dropna(subset=[x_col, y_col])

    # Calculate Spearman correlation
    rho, p_value = spearmanr(data[x_col], data[y_col])

    # Plot scatter plot with best-fit line
    plt.figure(figsize=(10, 8))
    sns.regplot(
        x=x_col, y=y_col, data=data,
        scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'}
    )
    plt.title(f"Spearman Correlation\nrho={rho:.2f}, p={p_value:.2g}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path)
    plt.close()
    print(f"Saved correlation plot to {save_path}")
    return rho, p_value

def main():
    # Step 1: Input file paths
    embeddings_path = input("Enter the path to the embeddings file: ")
    weights_path = input("Enter the path to the weights file: ")
    experiment_results_path = input("Enter the path to the experiment results file: ")

    # Step 2: Load data
    embeddings_df = pd.read_csv(embeddings_path)
    weights_df = pd.read_csv(weights_path)
    experiment_df = pd.read_csv(experiment_results_path)

    # Step 3: Ensure correct columns in embeddings file
    embeddings_df = embeddings_df.loc[:, ['image'] + [str(i) for i in range(49)]]

    # Step 4: Zero negative values
    embeddings_df = zero_negative_values(embeddings_df, [str(i) for i in range(49)])
    embeddings_dir, embeddings_file = os.path.split(embeddings_path)
    zeroed_file_path = os.path.join(embeddings_dir, f"zeroed_{embeddings_file}")
    embeddings_df.to_csv(zeroed_file_path, index=False)
    print(f"Zeroed embeddings saved to {zeroed_file_path}")

    # Step 5: Calculate entropy
    entropy_df = calculate_entropy(embeddings_df, [str(i) for i in range(49)])
    entropy_file_path = os.path.join(embeddings_dir, f"entropy_{embeddings_file}")
    entropy_df.to_csv(entropy_file_path, index=False)
    print(f"Entropy file saved to {entropy_file_path}")

    # Step 6: Compute dot products
    weights = weights_df.iloc[:, 0].values  # Assuming weights are in the first column
    entropy_change = compute_dot_products(embeddings_df, weights)

    # Add entropy change and weighted entropy
    entropy_df = entropy_df.copy()  # Avoid SettingWithCopyWarning
    entropy_df['Entropy Change'] = entropy_change
    entropy_df['Weighted Entropy'] = entropy_df['Entropy'] + entropy_df['Entropy Change']
    updated_entropy_file_path = os.path.join(embeddings_dir, f"weighted_{embeddings_file}")
    entropy_df.to_csv(updated_entropy_file_path, index=False)
    print(f"Updated entropy file saved to {updated_entropy_file_path}")

    # Step 7: Calculate pairwise differences
    difference_columns = ['Entropy', 'Entropy Change', 'Weighted Entropy']
    pairwise_diffs = calculate_pairwise_differences(entropy_df, difference_columns)

    # Step 8: Merge with experiment results
    if 'image' not in pairwise_diffs.columns:
        raise KeyError("'image' column not found in pairwise_diffs")
    if 'image' not in experiment_df.columns:
        raise KeyError("'image' column not found in experiment_df")

    # Ensure consistent formatting of 'image' columns
    pairwise_diffs['image'] = pairwise_diffs['image'].str.strip()
    experiment_df['image'] = experiment_df['image'].str.strip()

    # Merge the data
    pairwise_diffs = pairwise_diffs.merge(experiment_df, on='image', how='inner')

    # Step 9: Correlate differences with experimental results
    correlation_dir = os.path.join(embeddings_dir, "correlations")
    os.makedirs(correlation_dir, exist_ok=True)

    for col in difference_columns:
        # Correlate the differences
        x_col = col
        y_col = 'Proportion of Participant Choice'
        save_path = os.path.join(correlation_dir, f"{x_col}_vs_{y_col}.png")

        # Perform the correlation and plot
        calculate_spearman_correlation(pairwise_diffs, x_col, y_col, save_path)

    print("Analysis completed. Correlation plots saved.")

if __name__ == "__main__":
    main()