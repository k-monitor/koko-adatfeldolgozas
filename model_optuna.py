import pandas as pd
import numpy as np
import textdistance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score


# Add text preprocessing to improve indoklas matching
def preprocess_text(text):
    # Handle Hungarian language
    try:
        hungarian_stopwords = stopwords.words("hungarian")
    except:
        # Download stopwords if not available
        import nltk

        nltk.download("stopwords")
        hungarian_stopwords = stopwords.words("hungarian")

    # Create stemmer
    stemmer = SnowballStemmer("hungarian")

    # Lowercase, stem and remove stopwords
    if isinstance(text, str):
        words = text.lower().split()
        return " ".join(
            [stemmer.stem(word) for word in words if word not in hungarian_stopwords]
        )
    return ""


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


def precompute_tfidf(documents):
    vectorizer = TfidfVectorizer(stop_words=None, min_df=1)
    tfidf_matrix = vectorizer.fit_transform(documents)
    return vectorizer, tfidf_matrix


def weighted_function_classifier(
    row,
    df_old,
    weights=None,
    name_threshold=0.85,
    indoklas_threshold=0.85,
    function_column=None,
    tfidf_data=None,  # Add parameter for precomputed TF-IDF data
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
    indoklas = row["indoklas"]

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
        search_fid = ".".join(fid.split(".")[:-1])
        if old_row["fid"].startswith(search_fid):
            similarity = fid.count(".") / old_row["fid"].count(".")
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

    # 6. TF-IDF vectorization and cosine similarity for indoklas matching
    if indoklas and not df_old["indoklas"].isna().all():
        if tfidf_data is not None:
            # Use precomputed TF-IDF data
            vectorizer, old_tfidf_matrix = tfidf_data

            # Process and transform the current indoklas
            processed_indoklas = preprocess_text(indoklas)
            indoklas_vector = vectorizer.transform([processed_indoklas])

            # Compute cosine similarities with all documents in old dataset
            cosine_similarities = cosine_similarity(indoklas_vector, old_tfidf_matrix)[
                0
            ]

            # Find significant matches
            indoklas_similarities = []
            for i, sim in enumerate(cosine_similarities):
                if sim > indoklas_threshold:  # Only consider significant matches
                    indoklas_similarities.append((df_old.index[i], sim))

            # Sort by similarity score and take top 5
            indoklas_similarities.sort(key=lambda x: x[1], reverse=True)
            for i, similarity in indoklas_similarities[:5]:
                match = df_old.loc[i]
                if function_column is not None:
                    function = function_column.loc[i]
                else:
                    function = match["function"]
                if function not in function_scores:
                    function_scores[function] = 0
                    match_types[function] = []
                function_scores[function] += weights["indoklas_fuzzy"] * similarity
                match_types[function].append(f"indoklas_tfidf_{similarity:.2f}")
        else:
            # Prepare corpus for TF-IDF
            corpus = list(df_old["indoklas"].fillna("").apply(preprocess_text))
            if indoklas:
                processed_indoklas = preprocess_text(indoklas)
                corpus.append(processed_indoklas)

            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(stop_words=None, min_df=1)
            tfidf_matrix = vectorizer.fit_transform(corpus)

            # Get vector for current indoklas (last item in corpus)
            indoklas_vector = tfidf_matrix[-1]

            # Compute cosine similarities with all documents in old dataset
            corpus_vectors = tfidf_matrix[:-1]  # All except the current indoklas
            cosine_similarities = cosine_similarity(indoklas_vector, corpus_vectors)[0]

            # Find significant matches
            indoklas_similarities = []
            for i, sim in enumerate(cosine_similarities):
                if sim > indoklas_threshold:  # Only consider significant matches
                    indoklas_similarities.append((df_old.index[i], sim))

            # Sort by similarity score and take top 5
            indoklas_similarities.sort(key=lambda x: x[1], reverse=True)
            for i, similarity in indoklas_similarities[:5]:
                match = df_old.loc[i]
                if function_column is not None:
                    function = function_column.loc[i]
                else:
                    function = match["function"]
                if function not in function_scores:
                    function_scores[function] = 0
                    match_types[function] = []
                function_scores[function] += weights["indoklas_fuzzy"] * similarity
                match_types[function].append(f"indoklas_tfidf_{similarity:.2f}")

    # Find the function with the highest score
    if function_scores:
        best_function = max(function_scores.items(), key=lambda x: x[1])
        match_type = f"weighted ({best_function[1]:.2f}): {'+'.join(match_types[best_function[0]])}"
        return match_type, best_function[0]

    # No match found
    return None, None


# Prepare datasets for optimization
X_train = df_oldest.drop(columns=["function"])  # Training features
y_train = df_oldest["function"]  # Training targets

X_val = df_old.drop(columns=["function"])  # Validation features
y_val = df_old["function"]  # Validation targets

X = df_new.drop(columns=["function"])  # New dataset features
y = df_new["function"]  # New dataset targets

# Precompute TF-IDF for the training data
processed_indoklas_train = X_train["indoklas"].fillna("").apply(preprocess_text)
tfidf_data_train = precompute_tfidf(processed_indoklas_train)


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
    indoklas_threshold = trial.suggest_float("indoklas_threshold", 0.1, 0.95)

    # Apply weighted classifier with suggested weights
    y_pred = X_val.apply(
        lambda row: weighted_function_classifier(
            row,
            X_train,
            weights=weights,
            name_threshold=name_threshold,
            indoklas_threshold=indoklas_threshold,
            function_column=y_train,
            tfidf_data=tfidf_data_train,  # Pass precomputed TF-IDF data
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
study.optimize(objective, n_trials=100)  # Adjust n_trials as needed

# Print optimization results
print("Number of finished trials:", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("  Value: {:.4f}".format(trial.value))
print("  Params:")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

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
best_indoklas_threshold = best_params["indoklas_threshold"]

print("Best weights:")
for key, value in best_weights.items():
    print(f"  {key}: {value:.4f}")
print(f"Best name threshold: {best_name_threshold:.4f}")
print(f"Best FID threshold: {best_indoklas_threshold:.4f}")

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


processed_indoklas_old = df_old["indoklas"].fillna("").apply(preprocess_text)
tfidf_data_old = precompute_tfidf(processed_indoklas_old)

# Test with optimized parameters on df_new
optimized_y_pred = X.apply(
    lambda row: weighted_function_classifier(
        row,
        df_old,
        weights=best_weights,
        name_threshold=best_name_threshold,
        indoklas_threshold=best_indoklas_threshold,  # Add this parameter
        tfidf_data=tfidf_data_old,  # Pass precomputed TF-IDF data
    )[1],
    axis=1,
)

optimized_accuracy = get_accuracy(y, optimized_y_pred)
optimized_coverage = optimized_y_pred.notnull().sum() / len(optimized_y_pred)

print(
    f"Optimized parameters - Accuracy: {optimized_accuracy:.4f}, Coverage: {optimized_coverage:.4f}"
)
print(f"Improvement: {(optimized_accuracy - default_accuracy) * 100:.2f}%")

# After computing predictions
print("Classification report for default model:")
print(classification_report(y, default_y_pred_functions, zero_division=0))

print("\nClassification report for optimized model:")
print(classification_report(y, optimized_y_pred, zero_division=0))

# F1 scores
default_f1 = f1_score(y, default_y_pred_functions, average="weighted", zero_division=0)
optimized_f1 = f1_score(y, optimized_y_pred, average="weighted", zero_division=0)
print(f"Default F1-score: {default_f1:.4f}")
print(f"Optimized F1-score: {optimized_f1:.4f}")

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


# [I 2025-04-30 14:03:35,273] Trial 44 finished with value: 0.9266480965645311 and parameters: {'ahtt_exact': 18.24249077777863, 'name_exact': 9.646733805783061, 'fid_exact': 8.156735046769706, 'name_fuzzy': 2.9577721252455733, 'fid_fuzzy': 2.1705892451434217, 'indoklas_fuzzy': 6.487926301039365, 'name_threshold': 0.8407713991836503, 'fid_threshold': 0.5694381067962929, 'indoklas_threshold': 0.7633828580168956}. Best is trial 44 with value: 0.9266480965645311.
