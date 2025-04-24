import pandas as pd
import math


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

    str_rows = [
        ".".join([str(i) for i in l])
        .replace(".0", "")
        .replace(".0", "")
        .replace(".0", "")
        .replace(".0", "")
        for l in deduplicated_rows
    ]

    return str_rows


excel_file = "xlsx/Elfogadott költségvetések.xlsx"

years = [
    {
        "excel_sheet": "2016",
        "pdf_file": "javaslatok/2016 összefűzött javaslat.pdf",
        "columns": {
            "name": "NEV",
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
]

for year in years:
    excel_sheet = year["excel_sheet"]
    name_column = year["columns"]["name"]

    df = pd.read_excel(excel_file, sheet_name=excel_sheet, dtype={"ÁHT-T": str})
    df.columns = df.iloc[0]
    df = df[1:]
    df = df[df["FEJEZET"].notna()]

    df["spending"] = df[year["columns"]["spending"]].astype(float)
    df["income"] = df[year["columns"]["income"]].astype(float)
    if "support" in year["columns"]:
        df["support"] = df[year["columns"].get("support")].astype(float)
    else:
        df["support"] = 0.0
    if "accumulated_spending" in year["columns"]:
        df["accumulated_spending"] = df[
            year["columns"].get("accumulated_spending")
        ].astype(float)
    else:
        df["accumulated_spending"] = 0.0
    if "accumulated_income" in year["columns"]:
        df["accumulated_income"] = df[year["columns"].get("accumulated_income")].astype(
            float
        )
    else:
        df["accumulated_income"] = 0.0
    df["function"] = df["Funkció"]

    # concat 'FEJEZET', 'CIM', 'ALCIM', 'JOGCIM1'
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

    functions = get_functions(df)
    functions_null_mask = df["function"].apply(
        lambda x: isinstance(x, str) and x.strip() == ""
    )
    print(f"Nulls before: {functions_null_mask.sum()}")
    df["function"][functions_null_mask] = df["fid"][functions_null_mask].apply(
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
            + " ".join(x.iloc[0, 0].split(" ")[1:]).title()
        )
    )
    section_names.index = section_names.index.astype(int).astype(str)


    deduplicated_rows = get_deduplicated_rows(df)
    df = df[df["fid"].isin(deduplicated_rows)]


    # sum the values in the columns 'spending', 'income', 'support', 'accumulated_spending', 'accumulated_income'
    # for other columns take the first value
    df = df.groupby("fid").agg(
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
    ).reset_index()

    # sort fid column to handle 11.10.1.2 and 11.10.1 like values

    df = df.sort_values(
        by=["fid"],
        key=lambda x: x.str.split(".").apply(
            lambda y: [int(i) for i in y] if isinstance(y, list) else y
        ),
    ).reset_index(drop=True)
    

    # ahtts = get_functions(df, column="ÁHT-T")
    # print(f"Functions: {len(ahtts)}")
    # print(f"ÁHT-T: {df['ÁHT-T'].head(10)}")
    # ahtts_null_mask = df["ÁHT-T"].apply(
    #     lambda x: not x
    #     or (isinstance(x, str) and x.strip() == "NaN")
    #     or (type(x) == float and math.isnan(x))
    # )
    # print(f"ÁHT-T before: {ahtts_null_mask.sum()}")

    # df["ÁHT-T"][ahtts_null_mask] = df["fid"][ahtts_null_mask].apply(
    #     lambda x: find_closest_function(ahtts, x)
    # )
    # ahtts_null_mask = df["ÁHT-T"].apply(
    #     lambda x: not x
    #     or (isinstance(x, str) and x.strip() == "NaN")
    #     or (type(x) == float and math.isnan(x))
    # )
    # print(f"ÁHT-T after: {ahtts_null_mask.sum()}")
    # df["ÁHT-T"] = df["ÁHT-T"].astype(str).str.replace(r"\.0$", "", regex=True)

    df["fname"] = df["fid"].apply(lambda x: section_names.loc[x.split(".")[0]])

    df_indoklas = pd.read_csv(f"descriptions_{excel_sheet}.csv", index_col=0)
    df_merged = pd.merge(df, df_indoklas, on="fid", how="left")

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

    dataset.to_csv(
        f"dataset_{excel_sheet}.csv",
        index=False,
        sep=";",
        encoding="utf-8-sig",
    )

    dataset.to_json(
        f"dataset_{excel_sheet}.json",
        orient="records",
        lines=True,
        force_ascii=False,
    )
    print(f"Dataset for {year['pdf_file']} created.")
