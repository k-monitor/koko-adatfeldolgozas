import json
import pandas as pd
import numpy as np
import textdistance
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer
import scipy.sparse as sp
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from collections import defaultdict, Counter
from dotenv import load_dotenv
import nltk
from generate_name_indexes import generate_name_indexes
from zarszamadas.pdf_name_matcher import (
    extract_text_from_pdf,
    extract_names_by_function,
    filter_names,
    match_name_with_dataset,
)

# Load environment variables
load_dotenv(dotenv_path=".env")

# Constants
POSSIBLE_FUNCTIONS = [
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

FUNCTION_YEARS = ["2016", "2017", "2018", "2019"]


def generate_name_indexes_filtered(test_year):
    filtered_years = [year for year in FUNCTION_YEARS if year < str(test_year)]
    generate_name_indexes(years=filtered_years, excel_file="adatok/koltsegvetesek.xlsx")


USE_N2F = False

# Method classification
PRECISE_METHODS = [
    "ahtt_exact",
    "name_exact",
    "fid_exact",
    "name_fuzzy",
    "zarszam_name",
    "indoklas_fuzzy",
    "ctfidf",
]
IMPRECISE_METHODS = [
    "fid_fuzzy",
    "name_fuzzy_fallback",
    "ctfidf_atnezendo",
]


def get_zarszam_names(year):
    pdf_path = f"zarszamok/{year}.pdf"
    
    print("PDF NAME MATCHER - EXAMPLE USAGE")
    print("=" * 60)
    
    # Step 1: Extract text from PDF
    print("1. Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)
    if not text:
        print("Error: Could not extract text from PDF")
        return
    print(f"   Extracted {len(text)} characters")
    
    # Step 2: Extract names by function
    print("2. Extracting names by function...")
    names_by_function = extract_names_by_function(text)
    total_names = sum(len(names) for names in names_by_function.values())
    print(f"   Extracted {total_names} names from {len(names_by_function)} functions")
    
    # Show sample results
    if names_by_function:
        print("   Sample functions:")
        for i, (func_code, names) in enumerate(list(names_by_function.items())[:3]):
            print(f"     {func_code}: {len(names)} names")
            if names:
                print(f"       Example: {names[0]}")
    
    # Step 3: Filter names
    print("3. Filtering names...")
    filtered_names, name_functions = filter_names(names_by_function, min_length=10)
    filtered_total = sum(len(names) for names in filtered_names.values())
    print(f"   Kept {filtered_total} names after filtering")

    return filtered_names
    

class CTFIDFVectorizer(TfidfTransformer):
    """Custom TF-IDF vectorizer for class-based term frequency analysis."""

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


class DataLoader:
    """Handle loading and preprocessing of budget data."""

    @staticmethod
    def load_budget_data():
        """Load all budget datasets."""
        datasets = {}
        for year in range(2016, 2026+1):
            df = pd.read_json(f"dataset/{year}.json", lines=True)
            datasets[year] = DataLoader.preprocess_df(df)
        return datasets

    @staticmethod
    def preprocess_df(df):
        """Preprocess a budget dataframe."""
        df = df.copy()  # Create a copy to avoid SettingWithCopyWarning
        df["indoklas"] = df["indoklas"].fillna("")
        df = df.fillna(0)
        df["sum"] = df["spending"] + df["accumulated_spending"]
        df["income_sum"] = df["income"] + df["accumulated_income"]
        return df

    @staticmethod
    def load_excel_data():
        """Load and process Excel data for all years."""
        excel_file = "adatok/koltsegvetesek.xlsx"
        excel_dfs = {}

        for year in FUNCTION_YEARS:
            df = pd.read_excel(excel_file, sheet_name=year)
            df = DataLoader.process_excel_df(df, "MEGNEVEZÉS")
            excel_dfs[year] = df

        return excel_dfs

    @staticmethod
    def process_excel_df(df, name_column):
        """Process individual Excel dataframe."""
        df = df[df["FEJEZET"].notna()].copy()  # Create explicit copy to avoid warnings
        df = df.fillna({"CIM": 0, "ALCIM": 0, "JOGCIM1": 0, "JOGCIM2": 0})

        # Generate FIDs
        numbered_rows = [
            DataLoader.to_numbers(row)
            for row in df[["FEJEZET", "CIM", "ALCIM", "JOGCIM1", "JOGCIM2"]].itertuples(
                index=False, name=None
            )
        ]

        filled_rows = DataLoader.fill_hierarchical_ids(numbered_rows)
        fids = [DataLoader.create_fid(row) for row in filled_rows]

        df["fid"] = fids
        df["name"] = df[name_column]

        return df

    @staticmethod
    def to_numbers(row):
        """Convert a row of strings to numbers."""
        return [int(x) if str(x).isdigit() else x for x in row]

    @staticmethod
    def fill_hierarchical_ids(numbered_rows):
        """Fill hierarchical structure in numbered rows."""
        filled_rows = []
        prev_filled_row = None

        for current_row_sparse in numbered_rows:
            new_filled_row = [0] * len(current_row_sparse)

            if prev_filled_row is None:
                new_filled_row = current_row_sparse.copy()
            else:
                context_established = False
                for i in range(len(current_row_sparse)):
                    if context_established:
                        new_filled_row[i] = current_row_sparse[i]
                    else:
                        if current_row_sparse[i] != 0:
                            new_filled_row[i] = current_row_sparse[i]
                            if prev_filled_row[i] != current_row_sparse[i]:
                                context_established = True
                        else:
                            new_filled_row[i] = prev_filled_row[i]

            filled_rows.append(new_filled_row)
            prev_filled_row = new_filled_row

        return filled_rows

    @staticmethod
    def create_fid(row):
        """Create FID string from row."""
        fid = ".".join([str(i) for i in row])
        for _ in range(4):  # Remove trailing .0s
            fid = fid.replace(".0", "")
        return fid


class TextProcessor:
    """Handle text preprocessing and similarity calculations."""

    def __init__(self):
        self.stemmer = SnowballStemmer("hungarian")
        try:
            self.hungarian_stopwords = stopwords.words("hungarian")
        except:
            nltk.download("stopwords")
            self.hungarian_stopwords = stopwords.words("hungarian")

    def stem(self, text):
        """Stem text."""
        words = text.lower().split()
        return " ".join([self.stemmer.stem(word) for word in words])

    def preprocess_text(self, text):
        """Preprocess text for analysis."""
        if isinstance(text, str):
            words = text.lower().split()
            return " ".join(
                [
                    self.stemmer.stem(word)
                    for word in words
                    if word not in self.hungarian_stopwords
                ]
            )
        return ""

    def precompute_tfidf(self, documents, functions=None):
        """Compute TF-IDF representations with function awareness."""
        vectorizer = TfidfVectorizer(stop_words=None, min_df=1, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(documents)

        function_indices = None
        if functions is not None:
            function_indices = defaultdict(list)
            for i, func in enumerate(functions):
                function_indices[func].append(i)

        return vectorizer, tfidf_matrix, function_indices


class FunctionClassifier:
    """Main classifier for budget function prediction."""

    def __init__(self):
        self.text_processor = TextProcessor()
        self.n2f = self.load_name_mappings()
        self.excel_dfs = DataLoader.load_excel_data()

        # Initialize CTFIDF components
        self.count_vectorizer = None
        self.ctfidf_vectorizer = None
        self.ctfidf_matrix = None

    def load_name_mappings(self):
        """Load name to function mappings."""
        with open("n2f.json", "r") as f:
            return json.load(f)

    def setup_ctfidf_classifier(self, df_old):
        """Setup CTFIDF classifier."""
        df_old["f"] = df_old["function"].apply(self.f2c)

        docs_per_class = df_old.groupby(["f"], as_index=False).agg(
            {"indoklas": " ".join}
        )

        self.count_vectorizer = CountVectorizer(
            ngram_range=(1, 2), stop_words=self.text_processor.hungarian_stopwords
        ).fit(docs_per_class.indoklas)

        count = self.count_vectorizer.transform(docs_per_class.indoklas)
        self.ctfidf_vectorizer = CTFIDFVectorizer().fit(count, n_samples=len(df_old))
        self.ctfidf_matrix = self.ctfidf_vectorizer.transform(count)

    def f2c(self, f):
        """Convert function to class index."""
        return POSSIBLE_FUNCTIONS.index(f)

    def c2f(self, c):
        """Convert class index to function."""
        return POSSIBLE_FUNCTIONS[c]

    def classify_ctfidf(self, indoklas):
        """Classify using CTFIDF."""
        count = self.count_vectorizer.transform([indoklas])
        vector = self.ctfidf_vectorizer.transform(count)
        distances = cosine_similarity(vector, self.ctfidf_matrix)
        prediction = np.argmax(distances, 1)
        prediction_distance = distances[0][prediction[0]]
        if len(prediction) == 0:
            return None, 0
        # print(distances)
        # print(prediction, prediction_distance)
        return self.c2f(prediction[0]), prediction_distance

    def get_sub_names(self, fid, year):
        """Get sub-names for a given FID and year."""
        edf = self.excel_dfs[year]
        sub_names = edf[edf.fid.apply(lambda x: x.startswith(fid))]
        return "\n".join(sub_names["name"].tolist())

    def weighted_function_classifier(
        self, row, df_old, tfidf_data, names_list, indoklas_threshold=0.45
    ):
        """Classify function using weighted approach."""
        name = row["name"]
        ahtt = row["ÁHT-T"]
        fid = row["fid"]
        indoklas = row["indoklas"]

        method_matches = {
            "ahtt_exact": None,
            "name_exact": None,
            "fid_exact": None,
            "name_fuzzy": None,
            "fid_fuzzy": None,
            "indoklas_fuzzy": None,
            "name_fuzzy_fallback": None,
            "ctfidf": None,
            "ctfidf_atnezendo": None,
            "zarszam_name": None,
        }

        # 1. ÁHT-T exact match
        ahtt_matches = df_old[df_old["ÁHT-T"] == ahtt]
        if not ahtt_matches.empty:
            method_matches["ahtt_exact"] = ahtt_matches.iloc[0]["function"]

        # 2. Name exact match
        name_matches = df_old[df_old["name"].str.lower() == name.lower()]
        if not name_matches.empty:
            method_matches["name_exact"] = name_matches.iloc[0]["function"]

        # 3. FID exact match
        fid_matches = df_old[df_old["fid"] == fid]
        if not fid_matches.empty:
            method_matches["fid_exact"] = fid_matches.iloc[0]["function"]

        # 4. Fuzzy name matching
        method_matches["name_fuzzy"] = self._fuzzy_name_match(name, df_old)

        # 5. CTFIDF classification
        ctfidf_distance = None
        if indoklas:
            method_matches["ctfidf_atnezendo"], ctfidf_distance = self.classify_ctfidf(indoklas)
            if ctfidf_distance >= 0.18:
                method_matches["ctfidf"] = method_matches["ctfidf_atnezendo"]

        # 6. Fuzzy FID matching
        method_matches["fid_fuzzy"] = self._fuzzy_fid_match(fid, df_old)

        # 7. Fuzzy indoklas matching
        method_matches["indoklas_fuzzy"] = self._fuzzy_indoklas_match(
            indoklas, df_old, tfidf_data, indoklas_threshold
        )

        # 8. Fallback name matching
        method_matches["name_fuzzy_fallback"] = self._fallback_name_match(name, df_old)

        # 9. Zarszam names matching
        if names_list:
            for names_dict in reversed(names_list):
                match_info = match_name_with_dataset(
                    names_dict, row, similarity_threshold=0.9
                )
                if match_info and match_info["assigned_function"]:
                    method_matches["zarszam_name"] = match_info["assigned_function"]


        return {**method_matches, "oldrow": row.to_dict(), "predicted_function": None, "ctfidf_distance": ctfidf_distance}

    def _fuzzy_name_match(self, name, df_old):
        """Perform fuzzy name matching."""
        name_similarities = []
        for i, old_row in df_old.iterrows():
            old_name = old_row["name"]
            similarity = textdistance.algorithms.levenshtein.normalized_similarity(
                name.lower(), old_name.lower()
            )
            if similarity > 0.95:
                name_similarities.append((i, similarity))

        if not name_similarities and USE_N2F:
            for name_key in self.n2f.keys():
                if (
                    textdistance.algorithms.levenshtein.normalized_similarity(
                        self.text_processor.stem(name), name_key
                    )
                    > 0.95
                ):
                    return self.n2f[name_key]

        if name_similarities:
            name_similarities.sort(key=lambda x: x[1], reverse=True)
            i, _ = name_similarities[0]
            return df_old.iloc[i]["function"]

        return None

    def _fuzzy_fid_match(self, fid, df_old):
        """Perform fuzzy FID matching."""
        fid_similarities = []
        search_fid = ".".join(fid.split(".")[:-1]) + "."

        for i, old_row in df_old.iterrows():
            if old_row["fid"].startswith(search_fid):
                similarity = fid.count(".") / old_row["fid"].count(".")
                fid_similarities.append((i, similarity))

        if fid_similarities:
            cnt = Counter([df_old.iloc[i]["function"] for i, s in fid_similarities])
            if cnt.most_common(1):
                return cnt.most_common(1)[0][0]

        return None

    def _fuzzy_indoklas_match(self, indoklas, df_old, tfidf_data, threshold):
        """Perform fuzzy indoklas matching."""
        vectorizer, old_tfidf_matrix, function_indices = tfidf_data

        processed_indoklas = self.text_processor.preprocess_text(indoklas)
        indoklas_vector = vectorizer.transform([processed_indoklas])

        cosine_similarities = cosine_similarity(indoklas_vector, old_tfidf_matrix)[0]

        indoklas_similarities = [
            (df_old.index[i], sim)
            for i, sim in enumerate(cosine_similarities)
            if sim > threshold
        ]

        if indoklas_similarities:
            indoklas_similarities.sort(key=lambda x: x[1], reverse=True)
            i, _ = indoklas_similarities[0]
            match = df_old.loc[i]

            if isinstance(match["function"], pd.Series):
                return match["function"].mode().values[0]
            else:
                return match["function"]

        return None

    def _fallback_name_match(self, name, df_old):
        """Perform fallback name matching with lower threshold."""
        name_similarities = []
        for i, old_row in df_old.iterrows():
            old_name = old_row["name"]
            similarity = textdistance.algorithms.levenshtein.normalized_similarity(
                name.lower(), old_name.lower()
            )
            if similarity > 0.2:
                name_similarities.append((i, similarity))

        if name_similarities:
            name_similarities.sort(key=lambda x: x[1], reverse=True)
            i, _ = name_similarities[0]
            return df_old.iloc[i]["function"]

        return None


class ResultAnalyzer:
    """Analyze and report classification results."""

    @staticmethod
    def process_predictions(detailed_predictions):
        """Process predictions to determine final function."""

        def process_row(row):
            if row["ahtt_exact"]:
                row["predicted_function"] = row["ahtt_exact"]
                row["prediction_function"] = "ahtt_exact"
            elif row["name_exact"]:
                row["predicted_function"] = row["name_exact"]
                row["prediction_function"] = "name_exact"
            elif row["fid_exact"]:
                row["predicted_function"] = row["fid_exact"]
                row["prediction_function"] = "fid_exact"
            elif row["name_fuzzy"]:
                row["predicted_function"] = row["name_fuzzy"]
                row["prediction_function"] = "name_fuzzy"
            elif row["zarszam_name"]:
                row["predicted_function"] = row["zarszam_name"]
                row["prediction_function"] = "zarszam_name"
            elif row["fid_fuzzy"]:
                row["predicted_function"] = row["fid_fuzzy"]
                row["prediction_function"] = "fid_fuzzy"
            elif row["ctfidf"]:
                row["predicted_function"] = row["ctfidf"]
                row["prediction_function"] = "ctfidf"
            elif row["ctfidf_atnezendo"]:
                row["predicted_function"] = row["ctfidf_atnezendo"]
                row["prediction_function"] = "ctfidf_atnezendo"
            elif row["indoklas_fuzzy"]:
                row["predicted_function"] = row["indoklas_fuzzy"]
                row["prediction_function"] = "indoklas_fuzzy"
            else:
                row["predicted_function"] = None
                row["prediction_function"] = None
            return row

        return detailed_predictions.apply(lambda row: process_row(row))

    @staticmethod
    def create_matches_dataframe(detailed_predictions, y):
        """Create matches dataframe from predictions."""
        return pd.DataFrame(
            {
                "section_name": detailed_predictions.apply(
                    lambda x: x["oldrow"]["section_name"]
                ),
                "fid": detailed_predictions.apply(lambda x: x["oldrow"]["fid"]),
                "name": detailed_predictions.apply(lambda x: x["oldrow"]["name"]),
                "indoklas": detailed_predictions.apply(
                    lambda x: x["oldrow"]["indoklas"]
                ),
                "predicted_function": detailed_predictions.apply(
                    lambda x: x["predicted_function"]
                ),
                "prediction_function": detailed_predictions.apply(
                    lambda x: x["prediction_function"]
                ),  
                "method_sureness": detailed_predictions.apply(
                    lambda x: "helyesnek elfogadott" if str(x["prediction_function"]) in PRECISE_METHODS else "átnézendő"
                ),
                "manuális címke": detailed_predictions.apply(
                    lambda x: '',
                ),
                "ahtt_exact": detailed_predictions.apply(
                    lambda x: x["ahtt_exact"]
                ),
                "name_exact": detailed_predictions.apply(
                    lambda x: x["name_exact"]
                ),
                "fid_exact": detailed_predictions.apply(lambda x: x["fid_exact"]),
                "name_fuzzy": detailed_predictions.apply(
                    lambda x: x["name_fuzzy"]
                ),
                "zarszam_name": detailed_predictions.apply(lambda x: x["zarszam_name"]),
                "fid_fuzzy": detailed_predictions.apply(lambda x: x["fid_fuzzy"]),
                "indoklas_fuzzy": detailed_predictions.apply(
                    lambda x: x["indoklas_fuzzy"]
                ),
                "ctfidf": detailed_predictions.apply(lambda x: x["ctfidf"]),
                "ctfidf_atnezendo": detailed_predictions.apply(
                    lambda x: x["ctfidf_atnezendo"]
                ),
                "name_fuzzy_fallback": detailed_predictions.apply(
                    lambda x: x["name_fuzzy_fallback"]
                ),
                "sum": detailed_predictions.apply(lambda x: x["oldrow"]["sum"]),
                "income_sum": detailed_predictions.apply(lambda x: x["oldrow"]["income_sum"]),
                "ctfidf_similarity": detailed_predictions.apply(
                    lambda x: x["ctfidf_distance"]
                ),
                "true_function": y,
                "ÁHT-T": detailed_predictions.apply(lambda x: x["oldrow"]["ÁHT-T"]),
            }
        )

    @staticmethod
    def analyze_results(matches_df, selected_year):
        """Analyze and print classification results."""
        matches_df["is_correct"] = (
            matches_df["predicted_function"] == matches_df["true_function"]
        )

        print(f"Overall accuracy: {matches_df['is_correct'].mean():.4f}")
        print(f"Coverage: {matches_df['predicted_function'].notnull().mean():.4f}")
        print()

        # Analyze precise methods
        print("=== PRECISE METHODS ===")
        precise_stats = ResultAnalyzer._analyze_method_group(matches_df, PRECISE_METHODS)
        print()

        # Analyze imprecise methods (excluding cases covered by precise methods)
        print("=== IMPRECISE METHODS ===")
        imprecise_stats = ResultAnalyzer._analyze_method_group(
            matches_df, IMPRECISE_METHODS, exclude_methods=PRECISE_METHODS
        )
        print()

        # Individual method analysis
        print("=== INDIVIDUAL METHOD ANALYSIS ===")
        all_methods = PRECISE_METHODS + IMPRECISE_METHODS
        method_stats = {}

        for method in all_methods:
            mask = matches_df[method].notnull()
            if mask.sum() > 0:
                accuracy = (
                    matches_df[method] == matches_df["true_function"]
                ).sum() / mask.sum()
                coverage = mask.sum() / len(matches_df)
                sum_accuracy = (
                    matches_df[matches_df[method] == matches_df["true_function"]][
                        "sum"
                    ].sum()
                    / matches_df[mask]["sum"].sum()
                )
                sum_coverage = matches_df[mask]["sum"].sum() / matches_df["sum"].sum()

                method_stats[method] = {
                    'accuracy': accuracy,
                    'coverage': coverage,
                    'sum_accuracy': sum_accuracy,
                    'sum_coverage': sum_coverage
                }

                print(f"{method}: Accuracy = {accuracy:.4f}, Coverage = {coverage:.4f}")
                print(
                    f"  Sum Accuracy = {sum_accuracy:.4f}, Sum Coverage = {sum_coverage:.4f}"
                )
            else:
                method_stats[method] = {
                    'accuracy': 0,
                    'coverage': 0,
                    'sum_accuracy': 0,
                    'sum_coverage': 0
                }

        # Print formatted output for easy copying to spreadsheet
        print("\n=== FORMATTED OUTPUT ===")
        print(f"year\t{selected_year}")
        print(f"helyesként számontartott accuracy\t{precise_stats['accuracy']:.4f}")
        print(f"helyesként számontartott coverage\t{precise_stats['coverage']:.4f}")
        print(f"helyesként számontartott sum accuracy\t{precise_stats['sum_accuracy']:.4f}")
        print(f"helyesként számontartott sum coverage\t{precise_stats['sum_coverage']:.4f}")
        print(f"átnézendő accuracy\t{imprecise_stats['accuracy']:.4f}")
        print(f"átnézendő coverage\t{1-precise_stats['coverage']:.4f}")
        print(f"átnézendő sum accuracy\t{imprecise_stats['sum_accuracy']:.4f}")
        print(f"átnézendő sum coverage\t{1-precise_stats['sum_coverage']:.4f}")
        print()
        for method in all_methods:
            print(f"{method} accuracy\t{method_stats[method]['accuracy']:.4f}")
            print(f"{method} coverage\t{method_stats[method]['coverage']:.4f}")
            print(f"{method} sum accuracy\t{method_stats[method]['sum_accuracy']:.4f}")
            print(f"{method} sum coverage\t{method_stats[method]['sum_coverage']:.4f}")


    @staticmethod
    def _analyze_method_group(matches_df, methods, exclude_methods=None):
        """Analyze a group of methods."""
        # Create exclusion mask if exclude_methods provided
        exclusion_mask = pd.Series(False, index=matches_df.index)
        if exclude_methods:
            for method in exclude_methods:
                exclusion_mask |= matches_df[method].notnull()

        # Create mask for any method in the group (before exclusion)
        group_mask_raw = pd.Series(False, index=matches_df.index)
        for method in methods:
            group_mask_raw |= matches_df[method].notnull()

        # Apply exclusion - only analyze cases not covered by excluded methods
        if exclude_methods:
            # Final group mask: cases covered by this group AND not covered by excluded methods
            group_mask = group_mask_raw & ~exclusion_mask
            available_cases = len(matches_df) - exclusion_mask.sum()
        else:
            group_mask = group_mask_raw
            available_cases = len(matches_df)

        if group_mask.sum() == 0:
            print("No predictions from this method group")
            return {
                'accuracy': 0,
                'coverage': 0,
                'sum_accuracy': 0,
                'sum_coverage': 0
            }

        # Calculate group-level accuracy - only for cases in the final group mask
        group_correct = pd.Series(False, index=matches_df.index)
        for method in methods:
            # Only consider predictions that are in the final group mask
            method_mask = matches_df[method].notnull() & group_mask
            group_correct |= (
                matches_df[method] == matches_df["true_function"]
            ) & method_mask

        group_accuracy = group_correct.sum() / group_mask.sum()
        group_coverage = group_mask.sum() / available_cases

        # Sum-weighted metrics
        group_sum_correct = matches_df[group_correct]["sum"].sum()
        group_sum_total = matches_df[group_mask]["sum"].sum()
        group_sum_accuracy = (
            group_sum_correct / group_sum_total if group_sum_total > 0 else 0
        )

        if exclude_methods:
            available_sum = matches_df[~exclusion_mask]["sum"].sum()
        else:
            available_sum = matches_df["sum"].sum()
        group_sum_coverage = group_sum_total / available_sum if available_sum > 0 else 0

        print(f"Group Accuracy: {group_accuracy:.4f}")
        print(f"Group Coverage: {group_coverage:.4f}")
        print(f"Group Sum Accuracy: {group_sum_accuracy:.4f}")
        print(f"Group Sum Coverage: {group_sum_coverage:.4f}")
        
        return {
            'accuracy': group_accuracy,
            'coverage': group_coverage,
            'sum_accuracy': group_sum_accuracy,
            'sum_coverage': group_sum_coverage
        }


def main(selected_year):
    """Main execution function."""
    # Load data
    datasets = DataLoader.load_budget_data()

    df_old_list = []
    for year in range(2016, min(selected_year, 2020)):
        df_old_list.append(datasets[year])
    df_old_list.reverse()  # this helps to use the most recent data first
    df_old = pd.concat(df_old_list, ignore_index=True)

    names_list = []
    for year in [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]:
        if year < selected_year:
            names_list.append(get_zarszam_names(year))

    df_new = datasets[selected_year]

    # Filter data
    df_new = df_new[(df_new["sum"] > 0) | (df_new["income_sum"] > 0)].reset_index(drop=True)

    # Initialize classifier
    classifier = FunctionClassifier()
    classifier.setup_ctfidf_classifier(df_old)

    # Prepare TF-IDF data
    processed_indoklas_old = (
        df_old["indoklas"].fillna("").apply(classifier.text_processor.preprocess_text)
    )
    tfidf_data_old = classifier.text_processor.precompute_tfidf(
        processed_indoklas_old, df_old["function"]
    )

    # Classify
    X = df_new.drop(columns=["function"])
    y = df_new["function"]

    detailed_predictions = X.apply(
        classifier.weighted_function_classifier,
        axis=1,
        df_old=df_old,
        tfidf_data=tfidf_data_old,
        names_list=names_list,
    )

    # Process results
    detailed_predictions = ResultAnalyzer.process_predictions(detailed_predictions)
    matches_df = ResultAnalyzer.create_matches_dataframe(detailed_predictions, y)

    # Analyze results
    ResultAnalyzer.analyze_results(matches_df, selected_year)

    # Save results
    matches_df.to_excel(f"matches_df_{selected_year}.xlsx", index=False)

    return matches_df


if __name__ == "__main__":
    all_matches_dfs = {}
    for year in [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]:
        matches_df = main(year)
        all_matches_dfs[year] = matches_df
    
    # Save all dataframes to a single Excel file with separate sheets
    with pd.ExcelWriter("all_matches_combined.xlsx", engine='openpyxl') as writer:
        for year, df in all_matches_dfs.items():
            df.to_excel(writer, sheet_name=str(year), index=False)

