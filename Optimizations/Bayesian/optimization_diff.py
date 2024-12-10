import numpy as np
import pandas as pd
import os
from scipy.stats import spearmanr
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from datetime import datetime

def bayesian_optimization_fit(df1, df2, predictor_columns, task_name, n_calls=50, tol=1e-4, n_iter_no_change=10,
                              n_repeats=10, cross_validate=False, n_splits=5):
    # Ensure that df1 and df2 are aligned on 'image'
    if 'image' in df1.columns and 'image' in df2.columns:
        df_combined = pd.merge(df1, df2, on='image')
    else:
        raise ValueError("Both df1 and df2 must contain 'image' column for alignment.")

    # Process the images to compute differences between pairs
    # Extract base image names (without '_c' and file extension)
    df_combined['base_image'] = df_combined['image'].apply(lambda x: x.replace('_c', '').rsplit('.', 1)[0])

    # Group by 'base_image' to find pairs
    grouped = df_combined.groupby('base_image')

    # Initialize lists to store differences
    y_differences = []
    X_differences = []
    valid_base_images = []

    for name, group in grouped:
        if group.shape[0] != 2:
            print(f"Warning: base_image {name} does not have exactly 2 images. Skipping this pair.")
            continue

        # Identify images without '_c' and with '_c'
        idx_no_c = group['image'].str.contains('_c') == False
        idx_c = group['image'].str.contains('_c') == True

        if idx_no_c.sum() != 1 or idx_c.sum() != 1:
            print(f"Warning: base_image {name} does not have one image with '_c' and one without '_c'. Skipping this pair.")
            continue

        # Get 'Proportion of Participant Choice' values and compute the difference
        y_no_c = group.loc[idx_no_c, 'Proportion of Participant Choice'].values[0]
        y_c = group.loc[idx_c, 'Proportion of Participant Choice'].values[0]
        y_diff = y_no_c - y_c
        y_differences.append(y_diff)

        # Get predictor values and compute the difference
        X_no_c = group.loc[idx_no_c, predictor_columns].values[0]
        X_c = group.loc[idx_c, predictor_columns].values[0]
        X_diff = X_no_c - X_c
        X_differences.append(X_diff)

        valid_base_images.append(name)

    # Convert lists to arrays
    y_true = np.array(y_differences)  # Differences in 'Proportion of Participant Choice'
    X = np.array(X_differences)       # Differences in predictors
    n_predictors = X.shape[1]

    # Create a unique directory for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    opt_dir = f'./optimization_results_{timestamp}'
    os.makedirs(opt_dir, exist_ok=True)
    print(f"Results will be saved in: {opt_dir}")

    if cross_validate:
        print("Starting cross-validation...")
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        weights_list = []
        rho_list = []
        fold = 1
        for train_index, test_index in kf.split(X):
            print(f"\nProcessing fold {fold}/{n_splits}")
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y_true[train_index], y_true[test_index]
            best_theta, best_rho = optimize_weights(X_train, y_train, predictor_columns,
                                                    task_name + f"_fold{fold}", n_calls, tol,
                                                    n_iter_no_change, n_repeats, opt_dir)
            # Evaluate on the test set
            y_pred = X_test @ best_theta
            rho_test = spearmanr(y_pred, y_test)[0]
            print(f"Fold {fold}: Test Spearman rho = {rho_test}")
            weights_list.append(best_theta)
            rho_list.append(rho_test)
            fold += 1

        # Compute mean and standard error
        weights_array = np.array(weights_list)
        mean_weights = np.mean(weights_array, axis=0)
        std_weights = np.std(weights_array, axis=0, ddof=1)
        se_weights = std_weights / np.sqrt(n_splits)
        mean_rho = np.mean(rho_list)
        std_rho = np.std(rho_list, ddof=1)
        se_rho = std_rho / np.sqrt(n_splits)

        # Output results
        print("\nCross-validation results:")
        for i, predictor in enumerate(predictor_columns):
            print(f"{predictor}: Mean weight = {mean_weights[i]:.4f}, SE = {se_weights[i]:.4f}")
        print(f"Mean Spearman rho = {mean_rho:.4f}, SE = {se_rho:.4f}")

        # Save mean weights and SE to CSV
        results_df = pd.DataFrame({
            'Predictor': predictor_columns,
            'Mean Weight': mean_weights,
            'SE Weight': se_weights
        })
        results_csv_path = os.path.join(opt_dir, f'mean_weights_{task_name}.csv')
        results_df.to_csv(results_csv_path, index=False)
        print(f"Mean weights and SE saved to {results_csv_path}")

        # Generate bar plot of weights with standard error
        plt.figure(figsize=(10, 6))
        plt.bar(range(n_predictors), mean_weights, yerr=se_weights, capsize=5, tick_label=predictor_columns)
        plt.ylabel('Mean Weight')
        plt.title('Mean Weights with Standard Error')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        # Save the plot as a PNG file
        plot_filename = os.path.join(opt_dir, f'weights_bar_plot_{task_name}.png')
        plt.savefig(plot_filename)
        plt.close()
        print(f"Bar plot saved to {plot_filename}")

        return mean_weights, se_weights, mean_rho, se_rho
    else:
        print("Starting optimization without cross-validation...")
        best_theta, best_rho = optimize_weights(X, y_true, predictor_columns, task_name, n_calls, tol,
                                                n_iter_no_change, n_repeats, opt_dir)
        print("\nOptimization results:")
        for i, predictor in enumerate(predictor_columns):
            print(f"{predictor}: Weight = {best_theta[i]:.4f}")
        print(f"Best Spearman rho = {best_rho:.4f}")

        # Save final weights to CSV
        final_weights_df = pd.DataFrame({
            'Predictor': predictor_columns,
            'Weight': best_theta
        })
        final_weights_csv_path = os.path.join(opt_dir, f'final_weights_{task_name}.csv')
        final_weights_df.to_csv(final_weights_csv_path, index=False)
        print(f"Final weights saved to {final_weights_csv_path}")

        return best_theta, best_rho

def optimize_weights(X, y_true, predictor_columns, task_name, n_calls, tol, n_iter_no_change, n_repeats, opt_dir):
    n_predictors = X.shape[1]
    initial_theta = np.zeros(n_predictors)
    rhos = np.zeros(n_predictors)

    print("Calculating initial Spearman correlations for each predictor...")
    # Calculate initial Spearman correlations for each predictor
    for i in range(n_predictors):
        weights = np.zeros(n_predictors)
        weights[i] = 1
        y_pred = X @ weights  # Linear combination of predictor differences
        rho = spearmanr(y_pred, y_true)[0]
        rhos[i] = 0 if np.isnan(rho) else rho
        print(f"Predictor {predictor_columns[i]}: Initial Spearman rho = {rhos[i]}")

    # Determine the initial weights
    max_rho_index = np.argmax(rhos)
    initial_theta[max_rho_index] = 1  # Set the weight of the best predictor to 1
    initial_rho = rhos[max_rho_index]
    print(f"\nInitial weights set to predictor {predictor_columns[max_rho_index]} with weight 1. Initial Spearman rho = {initial_rho}")

    # File for logging
    filename = os.path.join(opt_dir, f'spearman_correlations_{task_name}.txt')
    with open(filename, 'w') as f:
        f.write(f"Task: {task_name}\nInitial weights:\n")
        f.write(f'{predictor_columns[max_rho_index]}: 1\n')
        f.write(f'Initial Spearman rho = {initial_rho}\n')

    best_overall_theta = initial_theta.copy()
    best_overall_rho = initial_rho

    for repeat in range(n_repeats):
        print(f"\nStarting repeat {repeat+1}/{n_repeats}...")
        best_rho = initial_rho
        best_theta = initial_theta.copy()

        def objective(**kwargs):
            theta = np.array([kwargs.get(f'theta{i}', 0) for i in range(n_predictors)])
            if theta.sum() == 0:
                return -1e10
            theta /= theta.sum()
            y_pred = X @ theta
            rho = spearmanr(y_pred, y_true)[0]
            return rho if not np.isnan(rho) else -1e10

        bounds = {f'theta{i}': (0, 1) for i in range(n_predictors)}
        opt = BayesianOptimization(objective, pbounds=bounds, random_state=None, verbose=0)
        opt.maximize(init_points=min(5, n_calls), n_iter=max(n_calls - 5, 0))

        if opt.max['target'] > best_rho:
            best_theta = np.array([opt.max['params'][f'theta{i}'] for i in range(n_predictors)])
            if best_theta.sum() != 0:
                best_theta /= best_theta.sum()
            best_rho = opt.max['target']

        print(f"Repeat {repeat+1} best rho: {best_rho}")
        if best_rho > best_overall_rho:
            best_overall_theta = best_theta
            best_overall_rho = best_rho
            print(f"Updated best overall rho to {best_overall_rho}")

    # Save final results
    final_weights_df = pd.DataFrame({
        'Predictor': predictor_columns,
        'Weight': best_overall_theta
    })
    final_weights_csv_path = os.path.join(opt_dir, f'final_weights_{task_name}.csv')
    final_weights_df.to_csv(final_weights_csv_path, index=False)
    print(f"Final weights saved to {final_weights_csv_path}")

    with open(filename, 'a') as f:
        f.write(f'\nBest overall Spearman rho = {best_overall_rho}\n')
        f.write('Best overall weights:\n')
        for i, weight in enumerate(best_overall_theta):
            f.write(f'{predictor_columns[i]}: {weight}\n')

    return best_overall_theta, best_overall_rho

def main():
    # Example usage (use actual file paths and variable names in practice):
    cross_validate = False
    df1 = pd.read_csv('/home/wallacelab/complexity-experiment-paper/Complexity Scores/ranking_complexity_scores.csv')
    df2 = pd.read_csv('/home/wallacelab/complexity-experiment-paper/Embeddings/CLIP-CBA/filtered_things_complexity_embedding.csv')
    predictor_columns = [col for col in df2.columns if col != 'image']

    if cross_validate:
        bayesian_optimization_fit(df1, df2, predictor_columns, task_name='complexity_prediction', cross_validate=True)
    else:
        best_weights, best_rho = bayesian_optimization_fit(df1, df2, predictor_columns, task_name='complexity_prediction', cross_validate=False)
        print(f"Best Spearman rho: {best_rho}")

if __name__ == "__main__":
    main()
