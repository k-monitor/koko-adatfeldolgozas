import os
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
from time import sleep
from dotenv import load_dotenv
from google import genai
from google.genai import types
import nltk

# Load environment variables
load_dotenv(dotenv_path=".env")

# Constants
POSSIBLE_FUNCTIONS = [
    "F01.a", "F01.b", "F01.c", "F01.d", "F01.e", "F01.f",
    "F02", "F03.a", "F03.b", "F03.c", "F03.d",
    "F04.a", "F04.b", "F04.c", "F04.d",
    "F05.a", "F05.b", "F05.c", "F05.d", "F05.e",
    "F06.a", "F06.b", "F06.c", "F06.d", "F06.e", "F06.f", "F06.g",
    "F07", "F08.a", "F08.b", "F08.c", "F08.d", "F08.e", "F08.f",
    "F09", "F10", "F11", "F12.a", "F12.b", "F12.c", "F12.d",
    "F13.a", "F13.b", "F14", "F15", "F16"
]

YEARS = ["2016", "2017", "2018", "2019"]

USE_N2F = False

# Method classification
PRECISE_METHODS = ["ahtt_exact_match", "name_exact_match", "fid_exact_match", "name_fuzzy_match"]
IMPRECISE_METHODS = ["fid_fuzzy_match", "indoklas_fuzzy", "name_fuzzy_fallback", "ctfidf"]

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
            idf, offsets=0, shape=(n_features, n_features),
            format="csr", dtype=np.float64
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
        for year in range(2016, 2022):
            df = pd.read_json(f"dataset/{year}.json", lines=True)
            datasets[year] = DataLoader.preprocess_df(df)
        return datasets
    
    @staticmethod
    def preprocess_df(df):
        """Preprocess a budget dataframe."""
        df["indoklas"].fillna("", inplace=True)
        df.fillna(0, inplace=True)
        df["sum"] = df["spending"] + df["accumulated_spending"]
        return df
    
    @staticmethod
    def load_excel_data():
        """Load and process Excel data for all years."""
        excel_file = "adatok/koltsegvetesek.xlsx"
        excel_dfs = {}
        
        for year in YEARS:
            df = pd.read_excel(excel_file, sheet_name=year)
            df = DataLoader.process_excel_df(df, "MEGNEVEZÉS")
            excel_dfs[year] = df
        
        return excel_dfs
    
    @staticmethod
    def process_excel_df(df, name_column):
        """Process individual Excel dataframe."""
        df = df[df["FEJEZET"].notna()]
        df["CIM"].fillna(0, inplace=True)
        df["ALCIM"].fillna(0, inplace=True)
        df["JOGCIM1"].fillna(0, inplace=True)
        df["JOGCIM2"].fillna(0, inplace=True)
        
        # Generate FIDs
        numbered_rows = [
            DataLoader.to_numbers(row) for row in 
            df[["FEJEZET", "CIM", "ALCIM", "JOGCIM1", "JOGCIM2"]].itertuples(index=False, name=None)
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
            return " ".join([
                self.stemmer.stem(word) for word in words 
                if word not in self.hungarian_stopwords
            ])
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
        
        docs_per_class = df_old.groupby(["f"], as_index=False).agg({"indoklas": " ".join})
        
        self.count_vectorizer = CountVectorizer(
            ngram_range=(1, 2), 
            stop_words=self.text_processor.hungarian_stopwords
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
        return self.c2f(prediction[0])
    
    def get_sub_names(self, fid, year):
        """Get sub-names for a given FID and year."""
        edf = self.excel_dfs[year]
        sub_names = edf[edf.fid.apply(lambda x: x.startswith(fid))]
        return "\n".join(sub_names["name"].tolist())
    
    def weighted_function_classifier(self, row, df_old, tfidf_data,
                                   name_threshold=0.84, indoklas_threshold=0.45):
        """Classify function using weighted approach."""
        name = row["name"]
        ahtt = row["ÁHT-T"]
        fid = row["fid"]
        indoklas = row["indoklas"]
        
        method_matches = {
            "ahtt_exact": None, "name_exact": None, "fid_exact": None,
            "name_fuzzy": None, "fid_fuzzy": None, "indoklas_fuzzy": None,
            "name_fuzzy_fallback": None, "ctfidf": None
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
        if indoklas:
            method_matches["ctfidf"] = self.classify_ctfidf(indoklas)
        
        # 6. Fuzzy FID matching
        method_matches["fid_fuzzy"] = self._fuzzy_fid_match(fid, df_old)
        
        # 7. Fuzzy indoklas matching
        method_matches["indoklas_fuzzy"] = self._fuzzy_indoklas_match(
            indoklas, df_old, tfidf_data, indoklas_threshold
        )
        
        # 8. Fallback name matching
        method_matches["name_fuzzy_fallback"] = self._fallback_name_match(name, df_old)
        
        return {
            **method_matches,
            "oldrow": row.to_dict(),
            "predicted_function": None
        }
    
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
                if textdistance.algorithms.levenshtein.normalized_similarity(
                    self.text_processor.stem(name), name_key
                ) > 0.95:
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
            (df_old.index[i], sim) for i, sim in enumerate(cosine_similarities)
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
            elif row["fid_fuzzy"]:
                row["predicted_function"] = row["fid_fuzzy"]
                row["prediction_function"] = "fid_fuzzy"
            elif row["ctfidf"]:
                row["predicted_function"] = row["ctfidf"]
                row["prediction_function"] = "ctfidf"
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
        return pd.DataFrame({
            "section_name": detailed_predictions.apply(lambda x: x["oldrow"]["section_name"]),
            "fid": detailed_predictions.apply(lambda x: x["oldrow"]["fid"]),
            "ÁHT-T": detailed_predictions.apply(lambda x: x["oldrow"]["ÁHT-T"]),
            "name": detailed_predictions.apply(lambda x: x["oldrow"]["name"]),
            "indoklas": detailed_predictions.apply(lambda x: x["oldrow"]["indoklas"]),
            "predicted_function": detailed_predictions.apply(lambda x: x["predicted_function"]),
            "prediction_function": detailed_predictions.apply(lambda x: x["prediction_function"]),
            "ahtt_exact_match": detailed_predictions.apply(lambda x: x["ahtt_exact"]),
            "name_exact_match": detailed_predictions.apply(lambda x: x["name_exact"]),
            "fid_exact_match": detailed_predictions.apply(lambda x: x["fid_exact"]),
            "name_fuzzy_match": detailed_predictions.apply(lambda x: x["name_fuzzy"]),
            "fid_fuzzy_match": detailed_predictions.apply(lambda x: x["fid_fuzzy"]),
            "indoklas_fuzzy": detailed_predictions.apply(lambda x: x["indoklas_fuzzy"]),
            "ctfidf": detailed_predictions.apply(lambda x: x["ctfidf"]),
            "name_fuzzy_fallback": detailed_predictions.apply(lambda x: x["name_fuzzy_fallback"]),
            "sum": detailed_predictions.apply(lambda x: x["oldrow"]["sum"]),
            "true_function": y,
        })
    
    @staticmethod
    def analyze_results(matches_df):
        """Analyze and print classification results."""
        matches_df["is_correct"] = (
            matches_df["predicted_function"] == matches_df["true_function"]
        )
        
        print(f"Overall accuracy: {matches_df['is_correct'].mean():.4f}")
        print(f"Coverage: {matches_df['predicted_function'].notnull().mean():.4f}")
        print()
        
        # Analyze precise methods
        print("=== PRECISE METHODS ===")
        ResultAnalyzer._analyze_method_group(matches_df, PRECISE_METHODS)
        print()
        
        # Analyze imprecise methods
        print("=== IMPRECISE METHODS ===")
        ResultAnalyzer._analyze_method_group(matches_df, IMPRECISE_METHODS)
        print()
        
        # Individual method analysis
        print("=== INDIVIDUAL METHOD ANALYSIS ===")
        all_methods = PRECISE_METHODS + IMPRECISE_METHODS
        
        for method in all_methods:
            mask = matches_df[method].notnull()
            if mask.sum() > 0:
                accuracy = (matches_df[method] == matches_df["true_function"]).sum() / mask.sum()
                coverage = mask.sum() / len(matches_df)
                sum_accuracy = (
                    matches_df[matches_df[method] == matches_df["true_function"]]["sum"].sum()
                    / matches_df[mask]["sum"].sum()
                )
                sum_coverage = matches_df[mask]["sum"].sum() / matches_df["sum"].sum()
                
                print(f"{method}: Accuracy = {accuracy:.4f}, Coverage = {coverage:.4f}")
                print(f"  Sum Accuracy = {sum_accuracy:.4f}, Sum Coverage = {sum_coverage:.4f}")
    
    @staticmethod
    def _analyze_method_group(matches_df, methods):
        """Analyze a group of methods."""
        # Create mask for any method in the group
        group_mask = pd.Series(False, index=matches_df.index)
        for method in methods:
            group_mask |= matches_df[method].notnull()
        
        if group_mask.sum() == 0:
            print("No predictions from this method group")
            return
        
        # Calculate group-level accuracy
        group_correct = pd.Series(False, index=matches_df.index)
        for method in methods:
            method_mask = matches_df[method].notnull()
            group_correct |= (matches_df[method] == matches_df["true_function"]) & method_mask
        
        group_accuracy = group_correct.sum() / group_mask.sum()
        group_coverage = group_mask.sum() / len(matches_df)
        
        # Sum-weighted metrics
        group_sum_correct = matches_df[group_correct]["sum"].sum()
        group_sum_total = matches_df[group_mask]["sum"].sum()
        group_sum_accuracy = group_sum_correct / group_sum_total if group_sum_total > 0 else 0
        group_sum_coverage = group_sum_total / matches_df["sum"].sum()
        
        print(f"Group Accuracy: {group_accuracy:.4f}")
        print(f"Group Coverage: {group_coverage:.4f}")
        print(f"Group Sum Accuracy: {group_sum_accuracy:.4f}")
        print(f"Group Sum Coverage: {group_sum_coverage:.4f}")

def main():
    """Main execution function."""
    # Load data
    datasets = DataLoader.load_budget_data()
    df_old = datasets[2016]
    df_new = datasets[2017]
    
    # Filter data
    df_new = df_new[df_new["sum"] > 0]
    
    # Initialize classifier
    classifier = FunctionClassifier()
    classifier.setup_ctfidf_classifier(df_old)
    
    # Prepare TF-IDF data
    processed_indoklas_old = df_old["indoklas"].fillna("").apply(
        classifier.text_processor.preprocess_text
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
        tfidf_data=tfidf_data_old
    )
    
    # Process results
    detailed_predictions = ResultAnalyzer.process_predictions(detailed_predictions)
    matches_df = ResultAnalyzer.create_matches_dataframe(detailed_predictions, y)
    
    # Analyze results
    ResultAnalyzer.analyze_results(matches_df)
    
    # Save results
    matches_df.to_excel("matches_df_2017.xlsx", index=False)
    
    return matches_df

if __name__ == "__main__":
    matches_df = main()
