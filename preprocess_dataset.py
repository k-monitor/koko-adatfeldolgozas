from pandas.core.frame import DataFrame
import pandas as pd
import os

excel_file = "adatok/koltsegvetesek.xlsx"

years = [
    {
        "excel_sheet": "2016",
        "columns": {
            "name": "MEGNEVEZÉS",
            "spending": "Kiadás",
            "income": "Bevétel",
            "support": "Támogatás",
        },
    },
    {
        "excel_sheet": "2017",
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
        "columns": {
            "name": "MEGNEVEZÉS",
            "spending": "Működési kiadás",
            "income": "Működési bevétel",
            "accumulated_spending": "Felhalmozási kiadás",
            "accumulated_income": "Felhalmozási bevétel",
        },
    },
    {
        "excel_sheet": "2019",
        "columns": {
            "name": "MEGNEVEZÉS",
            "spending": "Működési kiadás",
            "income": "Működési bevétel",
            "accumulated_spending": "Felhalmozási kiadás",
            "accumulated_income": "Felhalmozási bevétel",
        },
    },
    {
        "excel_sheet": "2020",
        "columns": {
            "name": "MEGNEVEZÉS",
            "spending": "Működési kiadás",
            "income": "Működési bevétel",
            "accumulated_spending": "Felhalmozási kiadás",
            "accumulated_income": "Felhalmozási bevétel",
        },
    },
    {
        "excel_sheet": "2021",
        "columns": {
            "name": "MEGNEVEZÉS",
            "spending": "Működési kiadás",
            "income": "Működési bevétel",
            "accumulated_spending": "Felhalmozási kiadás",
            "accumulated_income": "Felhalmozási bevétel",
        },
    },
    {
        "excel_sheet": "2022",
        "columns": {
            "name": "MEGNEVEZÉS",
            "spending": "Működési kiadás",
            "income": "Működési bevétel",
            "accumulated_spending": "Felhalmozási kiadás",
            "accumulated_income": "Felhalmozási bevétel",
        },
    },
    {
        "excel_sheet": "2023",
        "columns": {
            "name": "MEGNEVEZÉS",
            "spending": "Működési kiadás",
            "income": "Működési bevétel",
            "accumulated_spending": "Felhalmozási kiadás",
            "accumulated_income": "Felhalmozási bevétel",
        },
    },
    {
        "excel_sheet": "2024",
        "columns": {
            "name": "MEGNEVEZÉS",
            "spending": "Működési kiadás",
            "income": "Működési bevétel",
            "accumulated_spending": "Felhalmozási kiadás",
            "accumulated_income": "Felhalmozási bevétel",
        },
    },
    {
        "excel_sheet": "2025",
        "columns": {
            "name": "MEGNEVEZÉS",
            "spending": "Működési kiadás",
            "income": "Működési bevétel",
            "accumulated_spending": "Felhalmozási kiadás",
            "accumulated_income": "Felhalmozási bevétel",
        },
    },
    {
        "excel_sheet": "2026",
        "columns": {
            "name": "MEGNEVEZÉS",
            "spending": "Működési kiadás",
            "income": "Működési bevétel",
            "accumulated_spending": "Felhalmozási kiadás",
            "accumulated_income": "Felhalmozási bevétel",
        },
    },
]


def to_numbers(row):
    """
    Convert a row of strings to numbers.
    """
    return [int(x) if str(x).isdigit() else x for x in row]


def left_overlap(a, b):
    for i in range(len(a)):
        if a[i] != b[i]:
            return i
        return len(a)


def positive_length(a):
    return len([x for x in a if isinstance(x, int) and x > 0])


def get_functions(df, column="function"):
    """
    Get the functions from the dataframe.
    """
    functions = {}
    for i, row in df.iterrows():
        if row["fid"] not in functions and row[column] and row[column] != 0:
            functions[row["fid"]] = row[column]
    return functions


def find_closest_function(functions, fid):
    search_fid = fid
    while search_fid:
        if search_fid in functions:
            return functions[search_fid]
        search_fid = ".".join(search_fid.split(".")[:-1])
    return None


def get_deduplicated_rows(df):
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

    deduplicated_rows = []
    for i, row in enumerate(numbered_rows):
        prev_row = numbered_rows[i - 1] if i > 0 else None
        next_row = numbered_rows[i + 1] if i < len(numbered_rows) - 1 else None
        if (
            next_row
            and positive_length(next_row) == positive_length(row)
            and prev_row
            and prev_row != row
        ):
            deduplicated_rows.append(row)
        if (
            next_row
            and positive_length(next_row) < positive_length(row)
            and prev_row
            and prev_row != row
        ):
            deduplicated_rows.append(row)

    print("Deduplicated rows:")
    print(deduplicated_rows[:10])  # Print first 10 for debugging

    str_rows = [
        ".".join([str(i) for i in l])
        .replace(".0", "")
        .replace(".0", "")
        .replace(".0", "")
        .replace(".0", "")
        for l in deduplicated_rows
    ]

    return str_rows


def convert_float(value):
    value = str(value).replace(",", ".").replace(" ", "")
    return float(value) if value else 0.0


for year in years:
    excel_sheet = year["excel_sheet"]
    name_column = year["columns"]["name"]

    df = pd.read_excel(excel_file, sheet_name=excel_sheet, dtype={"ÁHT-T": str})
    df = df[df["FEJEZET"].notna()]

    df["spending"] = df[year["columns"]["spending"]].apply(convert_float).astype(float)
    df["income"] = df[year["columns"]["income"]].apply(convert_float).astype(float)
    if "support" in year["columns"]:
        df["support"] = (
            df[year["columns"].get("support")].apply(convert_float).astype(float)
        )
    else:
        df["support"] = 0.0
    if "accumulated_spending" in year["columns"]:
        df["accumulated_spending"] = (
            df[year["columns"].get("accumulated_spending")]
            .apply(convert_float)
            .astype(float)
        )
    else:
        df["accumulated_spending"] = 0.0
    if "accumulated_income" in year["columns"]:
        df["accumulated_income"] = (
            df[year["columns"].get("accumulated_income")]
            .apply(convert_float)
            .astype(float)
        )
    else:
        df["accumulated_income"] = 0.0
    if "Funkció" in df.columns:
        df["function"] = df["Funkció"]
    else:
        df["function"] = None
    if "ÁHT-T" not in df.columns:
        df["ÁHT-T"] = None

    print(df.head(10))
    df = df.fillna({"CIM": 0, "ALCIM": 0, "JOGCIM1": 0, "JOGCIM2": 0})

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

    print(fids[:10])  # Print first 10 for debugging

    functions = get_functions(df)
    functions_null_mask = df["function"].apply(
        lambda x: isinstance(x, str) and x.strip() == ""
    )
    print(f"Nulls before: {functions_null_mask.sum()}")
    df.loc[functions_null_mask, "function"] = df.loc[functions_null_mask, "fid"].apply(
        lambda x: find_closest_function(functions, x)
    )
    functions_null_mask = df["function"].apply(
        lambda x: isinstance(x, str) and x.strip() == ""
    )
    print(f"Nulls after: {functions_null_mask.sum()}")

    section_names = (
        df[[name_column, "FEJEZET"]]
        .groupby("FEJEZET")
        .apply(
            lambda x: x.iloc[0, 0].split(" ")[0]
            + " "
            + " ".join(x.iloc[0, 0].split(" ")[1:]).title(),
            include_groups=False
        )
    )
    section_names.index = section_names.index.astype(int).astype(str)

    deduplicated_rows = get_deduplicated_rows(df)
    df = df[df["fid"].isin(deduplicated_rows)]

    def not_top_fid(row_fid):
        for irow in df.to_dict(orient="records"):
            if irow["fid"].startswith(row_fid + "."):
                return False
        return True

    df = df[df["fid"].apply(not_top_fid)]

    # sum the values in the columns 'spending', 'income', 'support', 'accumulated_spending', 'accumulated_income'
    # for other columns take the first value
    df = (
        df.groupby("fid")
        .agg(
            {
                "spending": "sum",
                "income": "sum",
                "support": "sum",
                "accumulated_spending": "sum",
                "accumulated_income": "sum",
                "function": "first",
                "ÁHT-T": "first",
                "FEJEZET": "first",
                "CIM": "first",
                "ALCIM": "first",
                "JOGCIM1": "first",
                "JOGCIM2": "first",
                name_column: "first",
            }
        )
        .reset_index()
    )

    # sort fid column to handle 11.10.1.2 and 11.10.1 like values

    df = df.sort_values(
        by=["fid"],
        key=lambda x: x.str.split(".").apply(
            lambda y: [int(i) for i in y] if isinstance(y, list) else y
        ),
    ).reset_index(drop=True)

    df["fname"] = df["fid"].apply(lambda x: section_names.loc[x.split(".")[0]])

    if os.path.isfile(f"indoklasok/szovegek/{excel_sheet}.csv"):
        df_indoklas = pd.read_csv(
            f"indoklasok/szovegek/{excel_sheet}.csv",
            index_col=0,
        )
        df_indoklas["fid"] = df_indoklas.index.astype(str)
        df_indoklas["indoklas"] = df_indoklas["text"].astype(str)
        bad_explanations = ['Nincs indoklás', 'The justification']
        df_indoklas = df_indoklas[~df_indoklas['indoklas'].str.contains('|'.join(bad_explanations), na=False)]
        df_merged: DataFrame = pd.merge(df, df_indoklas, on="fid", how="left")
    else:
        df_merged = df.copy()
        df_merged["indoklas"] = None

    dataset = df_merged[
        [
            "fid",
            "ÁHT-T",
            "fname",
            name_column,
            "indoklas",
            "spending",
            "income",
            "support",
            "accumulated_spending",
            "accumulated_income",
            "function",
        ]
    ]
    dataset["section"] = dataset["fid"].apply(lambda x: x.split(".")[0])
    dataset["name"] = dataset[name_column].apply(lambda x: x.strip())
    dataset["section_name"] = dataset["fname"]
    dataset = dataset[
        [
            "section",
            "section_name",
            "ÁHT-T",
            "fid",
            "function",
            "name",
            "indoklas",
            "spending",
            "income",
            "support",
            "accumulated_spending",
            "accumulated_income",
        ]
    ]

    dataset_deduplicated = dataset.drop_duplicates(subset=['fid'], keep="first")

    os.makedirs("dataset", exist_ok=True)

    dataset_deduplicated.to_csv(
        f"dataset/{excel_sheet}.csv",
        index=False,
        sep=";",
        encoding="utf-8-sig",
    )

    dataset_deduplicated.to_json(
        f"dataset/{excel_sheet}.json",
        orient="records",
        lines=True,
        force_ascii=False,
    )
