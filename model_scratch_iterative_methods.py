from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")
from google import genai
from google.genai import types
import os
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
from time import sleep
import json
from sklearn import metrics

with open("n2f.json", "r") as f:
    n2f = json.load(f)

df_2016 = pd.read_json("dataset_2016.json", lines=True)
df_2017 = pd.read_json("dataset_2017.json", lines=True)
df_2018 = pd.read_json("dataset_2018.json", lines=True)
df_2019 = pd.read_json("dataset_2019.json", lines=True)
df_2020 = pd.read_json("dataset_2020.json", lines=True)
df_2021 = pd.read_json("dataset_2021.json", lines=True)


def preprocess_df(df):
    df["indoklas"].fillna("", inplace=True)
    df.fillna(0, inplace=True)
    df["sum"] = (
        df["spending"]
        # + df["income"]
        # + df["support"]
        + df["accumulated_spending"]
        # + df["accumulated_income"]
    )
    return df


df_2016 = preprocess_df(df_2016)
df_2017 = preprocess_df(df_2017)
df_2018 = preprocess_df(df_2018)
df_2019 = preprocess_df(df_2019)
df_2020 = preprocess_df(df_2020)
df_2021 = preprocess_df(df_2021)

df_old = pd.concat(
    [
        df_2019,
        df_2018,
        df_2017,
        df_2016,
    ]
)
df_new = df_2020
searchyear = "df_2016"
testyear = "df_2020"

# df_new["ÁHT-T"] = None

df_new = df_new[df_new["sum"] > 0]

print(df_2018.head(10))

possible_functions = [
    "F01.a",
    "F01.b",
    "F01.c",
    "F01.d",
    "F01.e",
    "F01.f",
    "F02",
    "F03.a",
    "F03.b",
    "F03.c",
    "F03.d",
    "F04.a",
    "F04.b",
    "F04.c",
    "F04.d",
    "F05.a",
    "F05.b",
    "F05.c",
    "F05.d",
    "F05.e",
    "F06.a",
    "F06.b",
    "F06.c",
    "F06.d",
    "F06.e",
    "F06.f",
    "F06.g",
    "F07",
    "F08.a",
    "F08.b",
    "F08.c",
    "F08.d",
    "F08.e",
    "F08.f",
    "F09",
    "F10",
    "F11",
    "F12.a",
    "F12.b",
    "F12.c",
    "F12.d",
    "F13.a",
    "F13.b",
    "F14",
    "F15",
    "F16",
]

import nltk

stemmer = SnowballStemmer("hungarian")


def stem(text):
    words = text.lower().split()
    return " ".join([stemmer.stem(word) for word in words])


def preprocess_text(text):
    try:
        hungarian_stopwords = stopwords.words("hungarian")
    except:
        nltk.download("stopwords")
        hungarian_stopwords = stopwords.words("hungarian")
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


y = df_new["function"]
y

X = df_new.drop(columns=["function"])
X

## CTFIDF

# processed_indoklas_old = df_old["indoklas"].fillna("").apply(preprocess_text)
# processed_indoklas_new = df_new["indoklas"].fillna("").apply(preprocess_text)


# def precompute_tfidf(documents, functions=None):
#     """
#     Compute TF-IDF representations with awareness of function categories.

#     Args:
#         documents (pd.Series): Series containing processed text documents
#         functions (pd.Series, optional): Series containing function labels for each document

#     Returns:
#         tuple: (vectorizer, tfidf_matrix, function_indices)
#     """
#     vectorizer = TfidfVectorizer(stop_words=None, min_df=1, ngram_range=(1, 2))
#     tfidf_matrix = vectorizer.fit_transform(documents)

#     function_indices = None
#     if functions is not None:
#         # Create a dictionary mapping functions to document indices
#         function_indices = defaultdict(list)
#         for i, func in enumerate(functions):
#             function_indices[func].append(i)

#     return vectorizer, tfidf_matrix, function_indices


# # Include function column for better differentiation
# vectorizer, tfidf_matrix, function_indices = precompute_tfidf(processed_indoklas_old, df_old["function"])
# tfidf_matrix_new = vectorizer.transform(processed_indoklas_new)

# # Function to find similar documents with function awareness
# def find_similar_with_function(query_text, vectorizer, tfidf_matrix, function_indices, threshold=0.3):
#     """
#     Find similar documents considering function categories.

#     Args:
#         query_text (str): The processed query text
#         vectorizer: TF-IDF vectorizer
#         tfidf_matrix: TF-IDF matrix of reference documents
#         function_indices: Dict mapping function labels to document indices
#         threshold (float): Minimum similarity score

#     Returns:
#         list: List of (index, similarity, function) tuples
#     """
#     query_vector = vectorizer.transform([query_text])
#     similarities = cosine_similarity(query_vector, tfidf_matrix)[0]

#     # Get results with function information
#     results = []
#     for function, indices in function_indices.items():
#         # Consider only documents of this function
#         for idx in indices:
#             if similarities[idx] > threshold:
#                 results.append((idx, similarities[idx], function))

#     # Sort by similarity score
#     return sorted(results, key=lambda x: x[1], reverse=True)

import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer


class CTFIDFVectorizer(TfidfTransformer):
    def __init__(self, *args, **kwargs):
        super(CTFIDFVectorizer, self).__init__(*args, **kwargs)

    def fit(self, X: sp.csr_matrix, n_samples: int):
        """Learn the idf vector (global term weights)"""
        _, n_features = X.shape
        df = np.squeeze(np.asarray(X.sum(axis=0)))
        idf = np.log(n_samples / df)
        self._idf_diag = sp.diags(
            idf,
            offsets=0,
            shape=(n_features, n_features),
            format="csr",
            dtype=np.float64,
        )
        return self

    def transform(self, X: sp.csr_matrix) -> sp.csr_matrix:
        """Transform a count-based matrix to c-TF-IDF"""
        X = X * self._idf_diag
        X = normalize(X, axis=1, norm="l1", copy=False)
        return X


def f2c(f):
    return possible_functions.index(f)


def c2f(c):
    return possible_functions[c]


# df_old["f"] = df_old["function"].apply(f2c)
# df_new["f"] = df_new["function"].apply(f2c)

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn import metrics
# from sentence_transformers import SentenceTransformer
# from sklearn.decomposition import PCA

# model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
# # tfidf_vect = TfidfVectorizer()
# pca = PCA(n_components=16,)
# rf = RandomForestClassifier(n_estimators=100, random_state=42)

# # X_train = tfidf_vect.fit_transform(df_old["indoklas"])
# embeddings = model.encode(df_old["indoklas"].values.ravel())
# X_embedded = pca.fit_transform(embeddings)
# y_train = df_old["f"]

# rf.fit(X_embedded, y_train.values.ravel())

# # X_test = tfidf_vect.transform(df_new["indoklas"])
# embeddings_test = model.encode(df_new["indoklas"].values.ravel())
# X_embedded_test = pca.transform(embeddings_test)
# y_test = df_new["f"]

# prediction = rf.predict(X_embedded_test)

# print(metrics.classification_report(y_test, prediction))


df_old["f"] = df_old["function"].apply(f2c)
print(df_new["function"].head(10))
if df_new[df_new["function"] == 0.0].empty:
    df_new["f"] = df_new["function"].apply(f2c)

docs_per_class = df_old.groupby(["f"], as_index=False).agg({"indoklas": " ".join})

count_vectorizer = CountVectorizer(
    ngram_range=(1, 2), stop_words=stopwords.words("hungarian")
).fit(docs_per_class.indoklas)
count = count_vectorizer.transform(docs_per_class.indoklas)
ctfidf_vectorizer = CTFIDFVectorizer().fit(count, n_samples=len(df_old))
ctfidf = ctfidf_vectorizer.transform(count)

if "f" in df_new.columns:
    test = df_new
    count = count_vectorizer.transform(test.indoklas)
    vector = ctfidf_vectorizer.transform(count)
    distances = cosine_similarity(vector, ctfidf)
    prediction = np.argmax(distances, 1)
    print(
        metrics.classification_report(
            test.f, prediction, target_names=possible_functions
        )
    )


def classify(indoklas):
    count = count_vectorizer.transform([indoklas])
    vector = ctfidf_vectorizer.transform(count)
    distances = cosine_similarity(vector, ctfidf)
    prediction = np.argmax(distances, 1)
    return c2f(prediction[0])


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
    {
        "excel_sheet": "2019",
        "pdf_file": "javaslatok/2019 összefűzött javaslat.pdf",
        "name_column": "MEGNEVEZÉS",
    },
]

excel_dfs = {}


def to_numbers(row):
    """
    Convert a row of strings to numbers.
    """
    return [int(x) if str(x).isdigit() else x for x in row]


for year in years:
    name_column = year["name_column"]
    excel_sheet = year["excel_sheet"]
    pdf_file = year["pdf_file"]

    df = pd.read_excel(excel_file, sheet_name=excel_sheet)
    df.columns = df.iloc[0]
    df = df[1:]

    df = df[df["FEJEZET"].notna()]
    df["CIM"].fillna(0, inplace=True)
    df["ALCIM"].fillna(0, inplace=True)
    df["JOGCIM1"].fillna(0, inplace=True)
    df["JOGCIM2"].fillna(0, inplace=True)

    numbered_rows = [
        to_numbers(row)
        for row in df[["FEJEZET", "CIM", "ALCIM", "JOGCIM1", "JOGCIM2"]].itertuples(
            index=False, name=None
        )
    ]

    print("Original rows:")
    print(numbered_rows[:10])  # Print first 10 for debugging

    filled_rows = []
    prev_filled_row = None

    for current_row_sparse in numbered_rows:
        new_filled_row = [0] * len(current_row_sparse)

        if prev_filled_row is None:
            # For the first row, the filled row is just a copy of the sparse row
            new_filled_row = current_row_sparse.copy()
        else:
            # This flag tracks if a non-zero value in current_row_sparse has differed
            # from prev_filled_row, thereby establishing a new "context".
            context_established_by_current_sparse_row = False
            for i in range(len(current_row_sparse)):
                if context_established_by_current_sparse_row:
                    # If context has changed, subsequent values are taken directly from current_row_sparse.
                    # If current_row_sparse[i] is 0, new_filled_row[i] will be 0.
                    new_filled_row[i] = current_row_sparse[i]
                else:
                    # Context not yet changed by a differing non-zero value in current_row_sparse.
                    if current_row_sparse[i] != 0:
                        new_filled_row[i] = current_row_sparse[i]
                        # Check if this non-zero value differs from prev_filled_row, thus changing context.
                        if prev_filled_row[i] != current_row_sparse[i]:
                            context_established_by_current_sparse_row = True
                    else:  # current_row_sparse[i] is 0
                        # Inherit from prev_filled_row as context hasn't changed at this point.
                        new_filled_row[i] = prev_filled_row[i]

        filled_rows.append(new_filled_row)
        prev_filled_row = (
            new_filled_row  # Update prev_filled_row for the next iteration
        )

    print("Filled rows:")
    print(filled_rows[:10])  # Print first 10 for debugging

    numbered_rows = filled_rows

    fids = [
        ".".join([str(i) for i in l])
        .replace(".0", "")
        .replace(".0", "")
        .replace(".0", "")
        .replace(".0", "")
        for l in numbered_rows
    ]

    # concat 'FEJEZET', 'CIM', 'ALCIM', 'JOGCIM1'
    df["fid"] = fids

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


# print(getsub_names("1.1.1", testyear))


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

    model = "gemini-2.5-flash-preview-05-20"
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
        "llm": None,
        "ctfidf": None,
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

    if not name_similarities:
        for name_key in n2f.keys():
            if textdistance.jaro_winkler(stem(name), name_key) > 0.84:
                method_matches["name_fuzzy"] = n2f[name_key]
                break

    name_similarities.sort(key=lambda x: x[1], reverse=True)
    for i, similarity in name_similarities[:1]:
        match = df_old.iloc[i]
        function = match["function"]
        method_matches["name_fuzzy"] = function

    if indoklas:
        method_matches["ctfidf"] = classify(indoklas)

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
        if type(match["function"]) == pd.Series:
            function = match["function"].mode().values[0]
        else:
            function = match["function"]
        # if function_column is not None:
        #     function = function_column.loc[i]
        # else:
        #     function = match["function"]
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
        and method_matches["name_fuzzy"] is None
    ):
        pass
        # method_matches["llm"] = classify_llm(row, year="2019", df=df_old)

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
        "ctfidf": method_matches["ctfidf"],
        "predicted_function": None,
    }


detailed_predictions = X.apply(
    weighted_function_classifier,
    axis=1,
    df_old=df_old,
    tfidf_data=tfidf_data_old,
    name_threshold=0.95,
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
    elif row["indoklas_fuzzy"]:
        row["predicted_function"] = row["indoklas_fuzzy"]
        row["prediction_function"] = "indoklas_fuzzy"
    elif row["llm"]:
        row["predicted_function"] = row["llm"]
        row["prediction_function"] = "llm"
    elif row["fid_fuzzy_match"]:
        row["predicted_function"] = row["fid_fuzzy_match"]
        row["prediction_function"] = "fid_fuzzy_match"
    elif row["name_fuzzy_fallback"]:
        row["predicted_function"] = row["name_fuzzy_fallback"]
        row["prediction_function"] = "name_fuzzy_fallback"
    elif row["ctfidf"]:
        row["predicted_function"] = row["ctfidf"]
        row["prediction_function"] = "ctfidf"
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
        "ctfidf": detailed_predictions.apply(lambda x: x["ctfidf"]),
        "llm": detailed_predictions.apply(lambda x: x["llm"]),
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

print(f"Coverage: {matches_df['predicted_function'].notnull().mean():.4f}")

# Add true function and evaluate accuracy
matches_df["true_function"] = y
matches_df["is_correct"] = (
    matches_df["predicted_function"] == matches_df["true_function"]
)

# Analyze accuracy by match type
print(f"Overall accuracy: {matches_df['is_correct'].mean():.4f}")
print(f"Coverage: {matches_df['predicted_function'].notnull().mean():.4f}")

# Show individual methods' accuracy
db_coverages = []
sum_coverages = []
db_accuracys = []
sum_accuracys = []
for method in [
    "ahtt_exact_match",
    "name_exact_match",
    "fid_exact_match",
    "name_fuzzy_match",
    "fid_fuzzy_match",
    "indoklas_fuzzy",
    "name_fuzzy_fallback",
    # "llm",
    "ctfidf",
]:
    mask = matches_df[method].notnull()
    accuracy = (matches_df[method] == matches_df["true_function"]).sum() / mask.sum()
    coverage = mask.sum() / len(matches_df)
    print(f"darab: {method}: Accuracy = {accuracy:.4f}, Coverage = {coverage:.4f}")
    db_coverages.append(str(coverage))
    db_accuracys.append(str(accuracy))
    sum_accuracy = (
        matches_df[matches_df[method] == matches_df["true_function"]]["sum"].sum()
        / matches_df[mask]["sum"].sum()
    )
    sum_coverage = matches_df[mask]["sum"].sum() / matches_df["sum"].sum()
    sum_coverages.append(str(sum_coverage))
    sum_accuracys.append(str(sum_accuracy))
    print(
        f"összeg: {method}: Accuracy = {sum_accuracy:.4f}, Coverage = {sum_coverage:.4f}"
    )

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
    lambda x: x
    in ["fid_fuzzy_match", "indoklas_fuzzy", None, "name_fuzzy_fallback", "ctfidf"]
)
matches_df["tuti"] = ~tutifilter
matches_df.to_excel(f"matches_df_{testyear}.xlsx", index=False)
matches_df_tuti = matches_df[~tutifilter]
matches_df_nemtuti = matches_df[tutifilter]


tuti_accuracy = get_accuracy(
    matches_df_tuti["y"], matches_df_tuti["predicted_function"], count_none=False
)
tuti_coverage = matches_df_tuti["predicted_function"].notna().sum() / len(
    detailed_predictions
)
nemtuti_accuracy = get_accuracy(
    matches_df_nemtuti["y"], matches_df_nemtuti["predicted_function"], count_none=False
)
nemtuti_coverage = matches_df_nemtuti["predicted_function"].notna().sum() / len(
    detailed_predictions
)

print(f"Tuti accuracy: {tuti_accuracy:.4f}")
print(f"Tuti coverage: {tuti_coverage:.4f}")

tuti_accuracy_sum = (
    matches_df_tuti[matches_df_tuti["predicted_function"] == matches_df_tuti["y"]][
        "sum"
    ].sum()
    / matches_df["sum"].sum()
)

print("tuti accuracy in percentage of the total sum: ", tuti_accuracy_sum)

tuti_coverage_sum = (
    matches_df_tuti[matches_df_tuti["predicted_function"].notna()]["sum"].sum()
    / matches_df["sum"].sum()
)
print("tuti coverage in percentage of the total sum: ", tuti_coverage)

nemtuti_coverage_sum = (
    matches_df_nemtuti[matches_df_nemtuti["predicted_function"].notna()]["sum"].sum()
    / matches_df["sum"].sum()
)
nemtuti_accuracy_sum = (
    matches_df_nemtuti[
        matches_df_nemtuti["predicted_function"] == matches_df_nemtuti["y"]
    ]["sum"].sum()
    / matches_df["sum"].sum()
)

sum_coverages.append(str(tuti_coverage_sum))
sum_accuracys.append(str(tuti_accuracy_sum))
db_coverages.append(str(tuti_coverage))
db_accuracys.append(str(tuti_accuracy))

sum_coverages.append(str(nemtuti_coverage_sum))
sum_accuracys.append(str(nemtuti_accuracy_sum))
db_coverages.append(str(nemtuti_coverage))
db_accuracys.append(str(nemtuti_accuracy))

print("db_coverage\n", "\n".join(db_coverages))
print("db_accuracy\n", "\n".join(db_accuracys))
print("sum_coverage\n", "\n".join(sum_coverages))
print("sum_accuracy\n", "\n".join(sum_accuracys))


number_of_tuti = matches_df_tuti["predicted_function"].notna().sum()
print(f"number of tuti: {number_of_tuti}")

total_rows = matches_df.shape[0]
print(f"total rows: {total_rows}")

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

totalsum = matches_df["sum"].sum()
matches_df_nemtuti["relative_sum"] = matches_df_nemtuti["sum"] / totalsum
print(
    matches_df_nemtuti.sort_values("sum", ascending=False)[
        ["name", "sum", "relative_sum"]
    ].head(20)
)
matches_df_nemtuti.sort_values("sum", ascending=False)[
    ["name", "sum", "relative_sum"]
].to_excel(f"matches_df_nemtuti_{testyear}.xlsx", index=False)

print("Cumulative accuracy by sum:")
print(cumulative_accuracy_by_sum)

cumulative_accuracy_alt = (
    (
        matches_df_nemtuti["sum"].sort_values(ascending=False).cumsum()
        + matches_df_tuti[matches_df_tuti["predicted_function"].notna()]["sum"].sum()
    )
    / matches_df["sum"].sum()
).head(20)

print("Cumulative accuracy alt:")
print(cumulative_accuracy_alt)


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
