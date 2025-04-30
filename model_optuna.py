import pandas as pd
import numpy as np
import textdistance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


df_oldest = pd.read_json("dataset_2016.json", lines=True)
df_old = pd.read_json("dataset_2017.json", lines=True)
df_new = pd.read_json("dataset_2018.json", lines=True)
df_new["indoklas"].fillna("", inplace=True)
df_oldest["indoklas"].fillna("", inplace=True)
df_old["indoklas"].fillna("", inplace=True)
df_new.fillna(0, inplace=True)
df_oldest.fillna(0, inplace=True)
df_old.fillna(0, inplace=True)


def get_accuracy(y_true, y_pred):
    """
    Calculate the accuracy of the predictions.
    """
    return np.sum(y_true == y_pred) / len(y_true)


def weighted_function_classifier(
    row,
    df_old,
    weights=None,
    name_threshold=0.85,
    fid_threshold=0.85,
    indoklas_threshold=0.85,
    function_column=None,
):
    """
    Assign a function to a budget item using a weighted voting classifier.

    Args:
        row: The row from the new dataset
        df_old: The old dataset containing known functions

    Returns:
        tuple: A tuple containing (match_type, function)
    """
    name = row["name"]
    ahtt = row["ÁHT-T"]
    fid = row["fid"]

    # Define weights for different match types
    if weights is None:
        # Default weights for different match types
        weights = {
            "ahtt_exact": 10.0,
            "name_exact": 8.0,
            "fid_exact": 7.0,
            "name_fuzzy": 6.0,
            "fid_fuzzy": 5.0,
            "indoklas_fuzzy": 4.0,
        }

    # Dictionary to store scores for each candidate function
    function_scores = {}

    # Track match types for reporting
    match_types = {}

    # 1. ÁHT-T exact match (highest weight)
    ahtt_matches = df_old[df_old["ÁHT-T"] == ahtt]
    for i, match in ahtt_matches.iterrows():
        if function_column is not None:
            function = function_column.iloc[i]
        else:
            function = match["function"]
        if function not in function_scores:
            function_scores[function] = 0
            match_types[function] = []
        function_scores[function] += weights["ahtt_exact"]
        match_types[function].append("ahtt_exact")

    # 2. Name exact match
    name_matches = df_old[df_old["name"] == name]
    for i, match in name_matches.iterrows():
        if function_column is not None:
            function = function_column.iloc[i]
        else:
            function = match["function"]
        if function not in function_scores:
            function_scores[function] = 0
            match_types[function] = []
        function_scores[function] += weights["name_exact"]
        match_types[function].append("name_exact")

    # 3. FID exact match
    fid_matches = df_old[df_old["fid"] == fid]
    for i, match in fid_matches.iterrows():
        if function_column is not None:
            function = function_column.iloc[i]
        else:
            function = match["function"]
        if function not in function_scores:
            function_scores[function] = 0
            match_types[function] = []
        function_scores[function] += weights["fid_exact"]
        match_types[function].append("fid_exact")

    # 4. Fuzzy name matching
    name_similarities = []
    for i, old_row in df_old.iterrows():
        old_name = old_row["name"]
        similarity = textdistance.jaro_winkler(name, old_name)
        if similarity > name_threshold:  # Only consider significant matches
            name_similarities.append((i, similarity))

    # Sort by similarity score and take top 5
    name_similarities.sort(key=lambda x: x[1], reverse=True)
    for i, similarity in name_similarities[:5]:
        match = df_old.iloc[i]
        if function_column is not None:
            function = function_column.iloc[i]
        else:
            function = match["function"]
        if function not in function_scores:
            function_scores[function] = 0
            match_types[function] = []
        function_scores[function] += weights["name_fuzzy"] * similarity
        match_types[function].append(f"name_fuzzy_{similarity:.2f}")

    # 5. Fuzzy FID matching
    fid_similarities = []
    for i, old_row in df_old.iterrows():
        old_fid = old_row["fid"]
        similarity = textdistance.jaro_winkler(fid, old_fid)
        if similarity > fid_threshold:  # Only consider significant matches
            fid_similarities.append((i, similarity))

    # Sort by similarity score and take top 5
    fid_similarities.sort(key=lambda x: x[1], reverse=True)
    for i, similarity in fid_similarities[:5]:
        match = df_old.iloc[i]
        if function_column is not None:
            function = function_column.iloc[i]
        else:
            function = match["function"]
        if function not in function_scores:
            function_scores[function] = 0
            match_types[function] = []
        function_scores[function] += weights["fid_fuzzy"] * similarity
        match_types[function].append(f"fid_fuzzy_{similarity:.2f}")

    # 6. Fuzzy indoklas matching
    fid_similarities = []
    for i, old_row in df_old.iterrows():
        old_fid = old_row["indoklas"]
        similarity = textdistance.jaro_winkler(fid, old_fid)
        if similarity > indoklas_threshold:  # Only consider significant matches
            fid_similarities.append((i, similarity))

    # Sort by similarity score and take top 5
    fid_similarities.sort(key=lambda x: x[1], reverse=True)
    for i, similarity in fid_similarities[:5]:
        match = df_old.iloc[i]
        if function_column is not None:
            function = function_column.iloc[i]
        else:
            function = match["function"]
        if function not in function_scores:
            function_scores[function] = 0
            match_types[function] = []
        function_scores[function] += weights["indoklas_fuzzy"] * similarity
        match_types[function].append(f"indoklas_fuzzy{similarity:.2f}")

    # Find the function with the highest score
    if function_scores:
        best_function = max(function_scores.items(), key=lambda x: x[1])
        match_type = f"weighted ({best_function[1]:.2f}): {'+'.join(match_types[best_function[0]])}"
        return match_type, best_function[0]

    # No match found
    return None, None


y = df_new["function"]
X = df_new.drop(columns=["function"])

# Apply the weighted function classifier
y_pred = X.apply(weighted_function_classifier, axis=1, df_old=df_old)

# Extract only the function predictions (second element of each tuple)
y_pred_functions = y_pred.apply(lambda x: x[1] if x is not None else None)

# Calculate accuracy and coverage
accuracy = get_accuracy(y, y_pred_functions)
coverage = y_pred_functions.notnull().sum() / len(y_pred_functions)

print(f"Accuracy: {accuracy:.4f}, Coverage: {coverage:.4f}")


y = df_old["function"]
X = df_old.drop(columns=["function"])


# Apply the weighted function classifier
y_pred = X.apply(weighted_function_classifier, axis=1, df_old=df_oldest)

# Extract only the function predictions (second element of each tuple)
y_pred_functions = y_pred.apply(lambda x: x[1] if x is not None else None)

# Calculate accuracy and coverage
accuracy = get_accuracy(y, y_pred_functions)
coverage = y_pred_functions.notnull().sum() / len(y_pred_functions)

print(f"Accuracy: {accuracy:.4f}, Coverage: {coverage:.4f}")


# Analyze the prediction method used for each item
match_types = y_pred.apply(lambda x: x[0] if x is not None else None)
match_types.value_counts().head(10)

import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Prepare datasets for optimization
X_train = df_oldest.drop(columns=["function"])  # Training features
y_train = df_oldest["function"]  # Training targets

X_val = df_old.drop(columns=["function"])  # Validation features
y_val = df_old["function"]  # Validation targets


def objective(trial):
    """
    Objective function for Optuna optimization.

    Args:
        trial: Optuna trial object

    Returns:
        float: Accuracy score to be maximized
    """
    # Suggest values for weights
    weights = {
        "ahtt_exact": trial.suggest_float("ahtt_exact", 5.0, 20.0, log=True),
        "name_exact": trial.suggest_float("name_exact", 4.0, 20.0, log=True),
        "fid_exact": trial.suggest_float("fid_exact", 3.0, 20.0, log=True),
        "name_fuzzy": trial.suggest_float("name_fuzzy", 2.0, 15.0, log=True),
        "fid_fuzzy": trial.suggest_float("fid_fuzzy", 1.0, 15.0, log=True),
        "indoklas_fuzzy": trial.suggest_float("indoklas_fuzzy", 1.0, 10.0, log=True),
    }

    # Suggest threshold values
    name_threshold = trial.suggest_float("name_threshold", 0.5, 0.95)
    fid_threshold = trial.suggest_float("fid_threshold", 0.5, 0.95)

    # Apply weighted classifier with suggested weights
    y_pred = X_val.apply(
        lambda row: weighted_function_classifier(
            row,
            X_train,
            weights=weights,
            name_threshold=name_threshold,
            fid_threshold=fid_threshold,
            function_column=y_train,
        )[
            1
        ],  # Extract only the function prediction
        axis=1,
    )

    # Calculate accuracy
    accuracy = get_accuracy(y_val, y_pred)

    return accuracy


# Create and run the optimization study
study = optuna.create_study(direction="maximize", study_name="weight_optimization")
study.optimize(objective, n_trials=50)  # Adjust n_trials as needed

# Print optimization results
print("Number of finished trials:", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("  Value: {:.4f}".format(trial.value))
print("  Params:")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# Visualize the optimization results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plot_optimization_history(study)

plt.subplot(1, 2, 2)
plot_param_importances(study)

plt.tight_layout()
plt.show()

# Extract the best parameters
best_params = study.best_params

# Format the best weights for the function classifier
best_weights = {
    "ahtt_exact": best_params["ahtt_exact"],
    "name_exact": best_params["name_exact"],
    "fid_exact": best_params["fid_exact"],
    "name_fuzzy": best_params["name_fuzzy"],
    "fid_fuzzy": best_params["fid_fuzzy"],
    "indoklas_fuzzy": best_params["indoklas_fuzzy"],
}
best_name_threshold = best_params["name_threshold"]
best_fid_threshold = best_params["fid_threshold"]

print("Best weights:")
for key, value in best_weights.items():
    print(f"  {key}: {value:.4f}")
print(f"Best name threshold: {best_name_threshold:.4f}")
print(f"Best FID threshold: {best_fid_threshold:.4f}")

# Test with default weights on df_new
default_y_pred = X.apply(weighted_function_classifier, axis=1, df_old=df_old)
default_y_pred_functions = default_y_pred.apply(
    lambda x: x[1] if x is not None else None
)
default_accuracy = get_accuracy(y, default_y_pred_functions)
default_coverage = default_y_pred_functions.notnull().sum() / len(
    default_y_pred_functions
)

print(
    f"Default parameters - Accuracy: {default_accuracy:.4f}, Coverage: {default_coverage:.4f}"
)

# Test with optimized parameters on df_new
optimized_y_pred = X.apply(
    lambda row: weighted_function_classifier(
        row,
        df_old,
        weights=best_weights,
        name_threshold=best_name_threshold,
        fid_threshold=best_fid_threshold,
    )[1],
    axis=1,
)

optimized_accuracy = get_accuracy(y, optimized_y_pred)
optimized_coverage = optimized_y_pred.notnull().sum() / len(optimized_y_pred)

print(
    f"Optimized parameters - Accuracy: {optimized_accuracy:.4f}, Coverage: {optimized_coverage:.4f}"
)
print(f"Improvement: {(optimized_accuracy - default_accuracy) * 100:.2f}%")

# Create a DataFrame to compare predictions
comparison_df = pd.DataFrame(
    {
        "true_function": y,
        "default_prediction": default_y_pred_functions,
        "optimized_prediction": optimized_y_pred,
    }
)

# Add columns for correct/incorrect predictions
comparison_df["default_correct"] = (
    comparison_df["true_function"] == comparison_df["default_prediction"]
)
comparison_df["optimized_correct"] = (
    comparison_df["true_function"] == comparison_df["optimized_prediction"]
)

# Find cases where optimization helped
improved = comparison_df[
    ~comparison_df["default_correct"] & comparison_df["optimized_correct"]
]
print(f"Items with improved prediction: {len(improved)}")

# Find cases where optimization made things worse
worsened = comparison_df[
    comparison_df["default_correct"] & ~comparison_df["optimized_correct"]
]
print(f"Items with worsened prediction: {len(worsened)}")

# Show examples of improvements
if len(improved) > 0:
    print("\nSample of improved predictions:")
    sample_improved = improved.sample(min(5, len(improved)))
    for i, row in sample_improved.iterrows():
        print(f"Item {i}: {X.loc[i, 'name']}")
        print(
            f"  True: {row['true_function']}, Default predicted: {row['default_prediction']}, Optimized predicted: {row['optimized_prediction']}"
        )
        print()
