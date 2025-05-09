import pandas as pd
import numpy as np
import textdistance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from collections import defaultdict


df_oldest = pd.read_json("dataset_2016.json", lines=True)
df_old = pd.read_json("dataset_2017.json", lines=True)
df_new = pd.read_json("dataset_2018.json", lines=True)
df_new["indoklas"].fillna("", inplace=True)
df_oldest["indoklas"].fillna("", inplace=True)
df_old["indoklas"].fillna("", inplace=True)
df_new.fillna(0, inplace=True)
df_oldest.fillna(0, inplace=True)
df_old.fillna(0, inplace=True)

df_oldest["sum"] = (
    df_oldest["spending"]
    + df_oldest["income"]
    + df_oldest["support"]
    + df_oldest["accumulated_spending"]
    + df_oldest["accumulated_income"]
)
df_old["sum"] = (
    df_old["spending"]
    + df_old["income"]
    + df_old["support"]
    + df_old["accumulated_spending"]
    + df_old["accumulated_income"]
)
df_new["sum"] = (
    df_new["spending"]
    + df_new["income"]
    + df_new["support"]
    + df_new["accumulated_spending"]
    + df_new["accumulated_income"]
)


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


processed_indoklas_old = df_old["indoklas"].fillna("").apply(preprocess_text)
processed_indoklas_new = df_new["indoklas"].fillna("").apply(preprocess_text)


def precompute_tfidf(documents, functions=None):
    """
    Compute TF-IDF representations with awareness of function categories.

    Args:
        documents (pd.Series): Series containing processed text documents
        functions (pd.Series, optional): Series containing function labels for each document

    Returns:
        tuple: (vectorizer, tfidf_matrix, function_indices)
    """
    vectorizer = TfidfVectorizer(stop_words=None, min_df=1, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(documents)

    function_indices = None
    if functions is not None:
        # Create a dictionary mapping functions to document indices
        function_indices = defaultdict(list)
        for i, func in enumerate(functions):
            function_indices[func].append(i)

    return vectorizer, tfidf_matrix, function_indices


# Include function column for better differentiation
tfidf_data_old = precompute_tfidf(processed_indoklas_old, df_old["function"])


# Function to find similar documents with function awareness
def find_similar_with_function(
    query_text, vectorizer, tfidf_matrix, function_indices, threshold=0.3
):
    """
    Find similar documents considering function categories.

    Args:
        query_text (str): The processed query text
        vectorizer: TF-IDF vectorizer
        tfidf_matrix: TF-IDF matrix of reference documents
        function_indices: Dict mapping function labels to document indices
        threshold (float): Minimum similarity score

    Returns:
        list: List of (index, similarity, function) tuples
    """
    query_vector = vectorizer.transform([query_text])
    similarities = cosine_similarity(query_vector, tfidf_matrix)[0]

    # Get results with function information
    results = []
    for function, indices in function_indices.items():
        # Consider only documents of this function
        for idx in indices:
            if similarities[idx] > threshold:
                results.append((idx, similarities[idx], function))

    # Sort by similarity score
    return sorted(results, key=lambda x: x[1], reverse=True)


import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

y = df_new["function"]
y

X = df_new.drop(columns=["function"])
X


def get_accuracy(y_true, y_pred, count_none=True):
    """
    Calculate the accuracy of the predictions.

    Args:
        y_true (pd.Series): True labels (str).
        y_pred (pd.Series): Predicted labels (str).
        count_none (bool): Whether to count None values in the accuracy calculation.

    Returns:
        float: Accuracy score between 0.0 and 1.0
    """
    if count_none:
        # Include None values in calculation
        accuracy = np.sum(y_true == y_pred) / len(y_true)
    else:
        # Exclude None values from calculation
        mask = y_pred.notnull()
        if mask.sum() > 0:
            accuracy = np.sum((y_true[mask] == y_pred[mask])) / mask.sum()
        else:
            accuracy = 0.0  # All predictions are None

    return accuracy


get_accuracy(y, y)

excel_file = "xlsx/Elfogadott költségvetések.xlsx"

years = [
    {
        "excel_sheet": "2016",
        "pdf_file": "javaslatok/2016 összefűzött javaslat.pdf",
        "name_column": "NEV",
    },
    {
        "excel_sheet": "2017",
        "pdf_file": "javaslatok/2017 összefűzött javaslat.pdf",
        "name_column": "MEGNEVEZÉS",
    },
    {
        "excel_sheet": "2018",
        "pdf_file": "javaslatok/2018 összefűzött javaslat.pdf",
        "name_column": "MEGNEVEZÉS",
    },
]

excel_dfs = {}

for year in years:
    name_column = year["name_column"]
    excel_sheet = year["excel_sheet"]
    pdf_file = year["pdf_file"]

    df = pd.read_excel(excel_file, sheet_name=excel_sheet)
    df.columns = df.iloc[0]
    df = df[1:]

    df = df[df["FEJEZET"].notna()]

    df["fid"] = (
        df["FEJEZET"].astype(int).astype(str)
        + "."
        + df["CIM"].astype(int).astype(str)
        + "."
        + df["ALCIM"].astype(int).astype(str)
        + "."
        + df["JOGCIM1"].astype(int).astype(str)
        + "."
        + df["JOGCIM2"].astype(int).astype(str)
    )

    df["fid"] = (
        df["fid"]
        .str.replace(r"\.0$", "", regex=True)
        .replace(r"\.0$", "", regex=True)
        .replace(r"\.0$", "", regex=True)
        .replace(r"\.0$", "", regex=True)
        .replace(r"\.0$", "", regex=True)
    )

    df["name"] = df[name_column]

    excel_dfs[excel_sheet] = df


def getsub_names(fid, year):
    edf = excel_dfs[year]
    sub_names = edf[edf.fid.apply(lambda x: x.startswith(fid))]
    return "\n".join(sub_names["name"].tolist())


getsub_names("1.1.1", "2018")


def sample_examples(df, n=5, year="2016"):
    df = df.sample(n=n)
    df = df[["indoklas", "function", "name", "fid"]]

    examples = []
    for i, row in df.iterrows():
        indoklas = row["indoklas"]
        function = row["function"]
        name = row["name"]
        fid = row["fid"]
        names = getsub_names(fid, year)

        examples.append(
            f"Tétel: {names}\n" f"Indoklás: {indoklas}\n" f"Funkció: {function}\n"
        )
    return examples


def generate(item, description, examples):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.5-flash-preview-04-17"
    prompt = f"""
Egy költségvetési tételt szeretnék besorolni a következő funkciók valamelyikébe.

Funkciók leírása:
**ÁLLAMI MŰKÖDÉSI FUNKCIÓK**

*   **F01: Általános közösségi szolgáltatások**
    *   F01.a: Törvényhozó és végrehajtó szervek
    *   F01.b: Pénzügyi és költségvetési tevékenységek és szolgáltatások
    *   F01.c: Külügyek
    *   F01.d: Alapkutatás
    *   F01.e: Műszaki fejlesztés
    *   F01.f: Egyéb általános közösségi szolgáltatások
*   **F02: Védelem**
*   **F03: Rendvédelem és közbiztonság**
    *   F03.a: Igazságszolgáltatás
    *   F03.b: Rend- és közbiztonság
    *   F03.c: Tűzvédelem
    *   F03.d: Büntetésvégrehajtási igazgatás és működtetés

**JÓLÉTI FUNKCIÓK**

*   **F04: Oktatási tevékenységek és szolgáltatások**
    *   F04.a: Iskolai előkészítés és alapfokú oktatás
    *   F04.b: Középfokú oktatás
    *   F04.c: Felsőfokú oktatás
    *   F04.d: Egyéb oktatás
*   **F05: Egészségügy**
    *   F05.a: Kórházi tevékenységek és szolgáltatások
    *   F05.b: Háziorvosi és gyermekorvosi szolgálat
    *   F05.c: Rendelői, orvosi, fogorvosi ellátás
    *   F05.d: Közegészségügyi tevékenységek és szolgáltatások
    *   F05.e: Egyéb egészségügy
*   **F06: Társadalombiztosítási és jóléti szolgáltatások**
    *   F06.a: Táppénz, anyasági vagy ideiglenes rokkantsági juttatások
    *   F06.b: Nyugellátások
    *   F06.c: Egyéb társadalombiztosítási ellátások
    *   F06.d: Munkanélküli ellátások
    *   F06.e: Családi pótlékok és gyermekeknek járó juttatások
    *   F06.f: Egyéb szociális támogatások
    *   F06.g: Szociális és jóléti intézményi szolgáltatások
*   **F07: Lakásügyek, települési és közösségi tevékenységek és szolgáltatások**
*   **F08: Szórakoztató, kulturális, vallási tevékenységek és szolgáltatások**
    *   F08.a: Sport és szabadidős tevékenységek és szolgáltatások
    *   F08.b: Kulturális tevékenységek és szolgáltatások
    *   F08.c: Műsorszórási és kiadói tevékenységek és szolgáltatások
    *   F08.d: Hitéleti tevékenységek
    *   F08.e: Párttevékenységek
    *   F08.f: Egyéb közösségi és kulturális tevékenységek

**GAZDASÁGI FUNKCIÓK**

*   **F09: Tüzelő- és üzemanyag, valamint energiaellátási feladatok**
*   **F10: Mező-, erdő-, hal- és vadgazdálkodás**
*   **F11: Bányászat és ipar**
*   **F12: Közlekedési és távközlési tevékenységek és szolgáltatások**
    *   F12.a: Közúti közlekedési tevékenységek
    *   F12.b: Vasúti közlekedésügyek és szolgáltatások
    *   F12.c: Távközlés
    *   F12.d: Egyéb közlekedés és szállítás
*   **F13: Egyéb gazdasági tevékenységek és szolgáltatások**
    *   F13.a: Többcélú fejlesztési témák tevékenységei és szolgáltatásai
    *   F13.b: Egyéb gazdasági tevékenységek és szolgáltatások
*   **F14: Környezetvédelem**

**ÁLLAMADÓSSÁG-KEZELÉS**

*   **F15: Államadósság-kezelés, államháztartás**

**FUNKCIÓBA NEM SOROLHATÓ TÉTELEK**

*   **F16: A főcsoportokba nem sorolható tételek**


Példák korábbi költségvetési tételek besorolására:
{examples}


Sorold be a korábban bemutatott összes funkció valamelyikébe a következő tételt: 
```
{item}
```

A tétel indoklása:
```
{description}
```"""
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=0,
        thinking_config=types.ThinkingConfig(
            thinking_budget=2048,
        ),
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type=genai.types.Type.OBJECT,
            required=["funkció kód"],
            properties={
                "funkció kód": genai.types.Schema(
                    type=genai.types.Type.STRING,
                    enum=possible_functions,
                    description="A funkció kódja, amely a költségvetési tételhez tartozik.",
                ),
            },
        ),
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )

    return response


def classify_llm(row, year, df):
    sleep(10)
    item = getsub_names(row["fid"], year)
    description = row["indoklas"]
    examples = sample_examples(df, n=5, year=year)
    try:
        result = generate(item, description, examples)
        function_code = result.parsed["funkció kód"]
        if function_code in possible_functions:
            return function_code
        else:
            print(f"Function code '{function_code}' not in possible functions.")
            return None
    except Exception as e:
        print(f"Error: {e}")
        sleep(30)
        return None


def weighted_function_classifier(
    row,
    df_old,
    name_threshold=0.84,
    indoklas_threshold=0.45,  # Lowered threshold for function-aware matching
    function_column=None,
    tfidf_data=None,
):
    name = row["name"]
    ahtt = row["ÁHT-T"]
    fid = row["fid"]
    indoklas = row["indoklas"]

    # Store best matches for each method
    method_matches = {
        "ahtt_exact": None,
        "name_exact": None,
        "fid_exact": None,
        "name_fuzzy": None,
        "fid_fuzzy": None,
        "indoklas_fuzzy": None,
        "name_fuzzy_fallback": None,
    }

    # 1. ÁHT-T exact match (highest weight)
    ahtt_matches = df_old[df_old["ÁHT-T"] == ahtt]
    for _, match in ahtt_matches.iterrows():
        function = match["function"]
        method_matches["ahtt_exact"] = function

    # 2. Name exact match
    name_matches = df_old[df_old["name"].str.lower() == name.lower()]
    for _, match in name_matches.iterrows():
        function = match["function"]
        method_matches["name_exact"] = function

    # 3. FID exact match
    fid_matches = df_old[df_old["fid"] == fid]
    for _, match in fid_matches.iterrows():
        function = match["function"]
        method_matches["fid_exact"] = function

    # 4. Fuzzy name matching
    name_similarities = []
    for i, old_row in df_old.iterrows():
        old_name = old_row["name"]
        similarity = textdistance.jaro_winkler(name.lower(), old_name.lower())
        if similarity > name_threshold:  # Only consider significant matches
            name_similarities.append((i, similarity))

    name_similarities.sort(key=lambda x: x[1], reverse=True)
    for i, similarity in name_similarities[:1]:
        match = df_old.iloc[i]
        function = match["function"]
        method_matches["name_fuzzy"] = function

    # 5. Fuzzy FID matching
    fid_similarities = []
    for i, old_row in df_old.iterrows():
        search_fid = ".".join(fid.split(".")[:-1])
        if old_row["fid"].startswith(search_fid):
            similarity = fid.count(".") / old_row["fid"].count(".")
            fid_similarities.append((i, similarity))

    fid_similarities.sort(key=lambda x: x[1], reverse=True)
    for i, similarity in fid_similarities[:1]:
        match = df_old.iloc[i]
        function = match["function"]
        method_matches["fid_fuzzy"] = function

    # 6. Fuzzy indoklas matching with function awareness
    vectorizer, old_tfidf_matrix, function_indices = tfidf_data

    # Process and transform the current indoklas
    processed_indoklas = preprocess_text(indoklas)
    indoklas_vector = vectorizer.transform([processed_indoklas])

    # Compute cosine similarities with all documents in old dataset
    cosine_similarities = cosine_similarity(indoklas_vector, old_tfidf_matrix)[0]

    # Find significant matches
    indoklas_similarities = []
    for i, sim in enumerate(cosine_similarities):
        if sim > indoklas_threshold:  # Only consider significant matches
            indoklas_similarities.append((df_old.index[i], sim))

    # Sort by similarity score and take top 5
    indoklas_similarities.sort(key=lambda x: x[1], reverse=True)
    for i, similarity in indoklas_similarities[:1]:
        match = df_old.loc[i]
        if function_column is not None:
            function = function_column.loc[i]
        else:
            function = match["function"]
        method_matches["indoklas_fuzzy"] = function

    # 7. Fuzzy name matching fallback
    name_similarities_fallback = []
    for i, old_row in df_old.iterrows():
        old_name = old_row["name"]
        similarity = textdistance.jaro_winkler(name.lower(), old_name.lower())
        if similarity > 0.5:  # Only consider significant matches
            name_similarities_fallback.append((i, similarity))

    name_similarities_fallback.sort(key=lambda x: x[1], reverse=True)
    for i, similarity in name_similarities_fallback[:1]:
        match = df_old.iloc[i]
        function = match["function"]
        method_matches["name_fuzzy_fallback"] = function

    if (
        method_matches["ahtt_exact"] is None
        and method_matches["name_exact"] is None
        and method_matches["fid_exact"] is None
        and method_matches["name_fuzzy"]
    ):
        method_matches["llm"] = classify_llm(row, year="2018", df=df_old)

    return {
        "ahtt_exact_match": method_matches["ahtt_exact"],
        "name_exact_match": method_matches["name_exact"],
        "fid_exact_match": method_matches["fid_exact"],
        "name_fuzzy_match": method_matches["name_fuzzy"],
        "fid_fuzzy_match": method_matches["fid_fuzzy"],
        "indoklas_fuzzy": method_matches["indoklas_fuzzy"],
        "name_fuzzy_fallback": method_matches["name_fuzzy_fallback"],
        "oldrow": row.to_dict(),
        "llm": method_matches["llm"],
        "predicted_function": None,
    }


detailed_predictions = X.apply(
    weighted_function_classifier, axis=1, df_old=df_old, tfidf_data=tfidf_data_old
)


def process_row(row):
    if row["ahtt_exact_match"]:
        row["predicted_function"] = row["ahtt_exact_match"]
        row["prediction_function"] = "ahtt_exact_match"
    elif row["name_exact_match"]:
        row["predicted_function"] = row["name_exact_match"]
        row["prediction_function"] = "name_exact_match"
    elif row["fid_exact_match"]:
        row["predicted_function"] = row["fid_exact_match"]
        row["prediction_function"] = "fid_exact_match"
    elif row["name_fuzzy_match"]:
        row["predicted_function"] = row["name_fuzzy_match"]
        row["prediction_function"] = "name_fuzzy_match"
    elif row["fid_fuzzy_match"]:
        row["predicted_function"] = row["fid_fuzzy_match"]
        row["prediction_function"] = "fid_fuzzy_match"
    elif row["indoklas_fuzzy"]:
        row["predicted_function"] = row["indoklas_fuzzy"]
        row["prediction_function"] = "indoklas_fuzzy"
    elif row["name_fuzzy_fallback"]:
        row["predicted_function"] = row["name_fuzzy_fallback"]
        row["prediction_function"] = "name_fuzzy_fallback"
    else:
        row["predicted_function"] = None
        row["prediction_function"] = None

    return row


detailed_predictions = detailed_predictions.apply(lambda row: process_row(row))

matches_df = pd.DataFrame(
    {
        "predicted_function": detailed_predictions.apply(
            lambda x: x["predicted_function"]
        ),
        "prediction_function": detailed_predictions.apply(
            lambda x: x["prediction_function"]
        ),
        "ahtt_exact_match": detailed_predictions.apply(lambda x: x["ahtt_exact_match"]),
        "name_exact_match": detailed_predictions.apply(lambda x: x["name_exact_match"]),
        "fid_exact_match": detailed_predictions.apply(lambda x: x["fid_exact_match"]),
        "name_fuzzy_match": detailed_predictions.apply(lambda x: x["name_fuzzy_match"]),
        "fid_fuzzy_match": detailed_predictions.apply(lambda x: x["fid_fuzzy_match"]),
        "indoklas_fuzzy": detailed_predictions.apply(lambda x: x["indoklas_fuzzy"]),
        "name_fuzzy_fallback": detailed_predictions.apply(
            lambda x: x["name_fuzzy_fallback"]
        ),
        "section_name": detailed_predictions.apply(
            lambda x: x["oldrow"]["section_name"]
        ),
        "name": detailed_predictions.apply(lambda x: x["oldrow"]["name"]),
        "fid": detailed_predictions.apply(lambda x: x["oldrow"]["fid"]),
        "indoklas": detailed_predictions.apply(lambda x: x["oldrow"]["indoklas"]),
        "ÁHT-T": detailed_predictions.apply(lambda x: x["oldrow"]["ÁHT-T"]),
        "sum": detailed_predictions.apply(lambda x: x["oldrow"]["sum"]),
    }
)


# Add true function and evaluate accuracy
matches_df["true_function"] = y
matches_df["is_correct"] = (
    matches_df["predicted_function"] == matches_df["true_function"]
)

# Analyze accuracy by match type
print(f"Overall accuracy: {matches_df['is_correct'].mean():.4f}")
print(f"Coverage: {matches_df['predicted_function'].notnull().mean():.4f}")

# Show individual methods' accuracy
for method in [
    "ahtt_exact_match",
    "name_exact_match",
    "fid_exact_match",
    "name_fuzzy_match",
    "fid_fuzzy_match",
    "indoklas_fuzzy",
    "name_fuzzy_fallback",
]:
    mask = matches_df[method].notnull()
    if mask.sum() > 0:
        accuracy = (
            matches_df[method] == matches_df["true_function"]
        ).sum() / mask.sum()
        coverage = mask.sum() / len(matches_df)
        print(f"{method}: Accuracy = {accuracy:.4f}, Coverage = {coverage:.4f}")

# Display a sample of the results
matches_df.head(10)

# Compare the indoklas_fuzzy match counts before and after function-aware matching
indoklas_matches_count = matches_df["indoklas_fuzzy"].notna().sum()
print(f"Function-aware indoklas matches: {indoklas_matches_count}")
print(f"Percentage of total: {indoklas_matches_count / len(matches_df):.2%}")

# Analyze how many indoklas fuzzy matches are correct
indoklas_mask = matches_df["indoklas_fuzzy"].notna()
if indoklas_mask.sum() > 0:
    indoklas_accuracy = (
        matches_df.loc[indoklas_mask, "indoklas_fuzzy"]
        == matches_df.loc[indoklas_mask, "true_function"]
    ).mean()
    print(f"Indoklas fuzzy match accuracy: {indoklas_accuracy:.2%}")

matches_df["y"] = y

tutifilter = matches_df["prediction_function"].apply(
    lambda x: x in ["fid_fuzzy_match", "indoklas_fuzzy", None, "name_fuzzy_fallback"]
)
matches_df_tuti = matches_df[~tutifilter]
matches_df_nemtuti = matches_df[tutifilter]


tuti_accuracy = get_accuracy(
    matches_df_tuti["y"], matches_df_tuti["predicted_function"], count_none=False
)
tuti_coverage = matches_df_tuti["predicted_function"].notna().sum() / len(
    detailed_predictions
)

print(f"Tuti accuracy: {tuti_accuracy:.4f}")
print(f"Tuti coverage: {tuti_coverage:.4f}")

accuracy_sum = (
    matches_df_tuti[matches_df_tuti["predicted_function"] == matches_df_tuti["y"]][
        "sum"
    ].sum()
    / matches_df["sum"].sum()
)

print("tuti accuracy in percentage of the total sum: ", accuracy_sum)

total_accuracy_sum = (
    matches_df[matches_df["predicted_function"] == matches_df["y"]]["sum"].sum()
    / matches_df["sum"].sum()
)

print("total accuracy in percentage of the total sum: ", total_accuracy_sum)

cumulative_accuracy_by_sum = (
    (
        matches_df_nemtuti["sum"].sort_values(ascending=False).cumsum()
        + matches_df_tuti[
            matches_df_tuti["predicted_function"] == matches_df_tuti["y"]
        ]["sum"].sum()
    )
    / matches_df["sum"].sum()
).head(20)

# Analyze the performance by function category
function_performance = {}

for func in y.unique():
    mask = matches_df["true_function"] == func
    if mask.sum() > 0:
        accuracy = matches_df.loc[mask, "is_correct"].mean()
        count = mask.sum()
        function_performance[func] = {"accuracy": accuracy, "count": count}

# Convert to DataFrame for better visualization
func_perf_df = pd.DataFrame.from_dict(function_performance, orient="index")
func_perf_df = func_perf_df.sort_values("count", ascending=False)

print("Performance by most common function categories:")
print(func_perf_df.head(20))
