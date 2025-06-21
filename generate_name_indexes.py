import pandas as pd
import json
from nltk.stem import SnowballStemmer

excel_file = "adatok/koltsegvetesek.xlsx"

years = [
    {
        "excel_sheet": "2016",
        "pdf_file": "javaslatok/2016 összefűzött javaslat.pdf",
        "columns": {
            "name": "MEGNEVEZÉS",
            "spending": "Kiadás",
            "income": "Bevétel",
            "support": "Támogatás",
        },
    },
    {
        "excel_sheet": "2017",
        "pdf_file": "javaslatok/2017 összefűzött javaslat.pdf",
        "columns": {
            "name": "MEGNEVEZÉS",
            "spending": "Működési kiadás",
            "income": "Működési bevétel",
            "accumulated_spending": "Felhalmozási kiadás",
            "accumulated_income": "Felhalmozási bevétel",
        },
    },
    {
        "excel_sheet": "2018",
        "pdf_file": "javaslatok/2018 összefűzött javaslat.pdf",
        "columns": {
            "name": "MEGNEVEZÉS",
            "spending": "Működési kiadás",
            "income": "Működési bevétel",
            "accumulated_spending": "Felhalmozási kiadás",
            "accumulated_income": "Felhalmozási bevétel",
        },
    },
    # {
    #     "excel_sheet": "2019",
    #     "pdf_file": "javaslatok/2019 összefűzött javaslat.pdf",
    #     "columns": {
    #         "name": "MEGNEVEZÉS",
    #         "spending": "Működési kiadás",
    #         "income": "Működési bevétel",
    #         "accumulated_spending": "Felhalmozási kiadás",
    #         "accumulated_income": "Felhalmozási bevétel",
    #     },
    # },
]

stemmer = SnowballStemmer("hungarian")


def stem(text):
    words = text.lower().split()
    return " ".join([stemmer.stem(word) for word in words])


n2f = {}


def to_numbers(row):
    """
    Convert a row of strings to numbers.
    """
    return [int(x) if str(x).isdigit() else x for x in row]


for year in years:
    excel_sheet = year["excel_sheet"]
    name_column = year["columns"]["name"]

    df = pd.read_excel(excel_file, sheet_name=excel_sheet, dtype={"ÁHT-T": str})
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

    print(df[[name_column, "Funkció"]].head(10))
    for name, function, fid in zip(df[name_column], df["Funkció"], df["fid"]):
        print(f"Name: {name}, Function: {function}, FID: {fid}")
        stemmed = stem(name)
        print(f"Stemmed: {stemmed}")
        stemmed = stemmed.strip()

        fs = None
        if not function.strip():
            for oname, ofunction, ofid in zip(df[name_column], df["Funkció"], df["fid"]):
                if fid in ofid:
                    if not ofunction.strip():
                        continue
                    if fs and fs != ofunction:
                        print(f"Conflict: {fs} != {ofunction}")
                        fs = None
                        break
                    fs = ofunction.strip()
            if fs:
                print(f"Function found: {fs}")
                function = fs

        if stemmed and function.strip():
            n2f[stemmed] = function

with open("n2f.json", "w", encoding="utf-8") as f:
    json.dump(n2f, f, ensure_ascii=False, indent=4)
