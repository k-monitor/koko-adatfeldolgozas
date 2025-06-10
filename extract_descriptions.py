# imports
from utils.pdf_extractor import extract_text_by_page
from collections import defaultdict
from pprint import pprint
import pandas as pd
import os
import PyPDF2  # Add this import for PDF splitting
import re
import base64
from google import genai
from google.genai import types
from dotenv import load_dotenv
import json
from tqdm import tqdm
import traceback
from utils.pdf_extractor import get_page_lenth

load_dotenv(dotenv_path=".env")

excel_file = "xlsx/Elfogadott költségvetések.xlsx"

years = [
    # {
    #     "excel_sheet": "2016",
    #     "pdf_file": "javaslatok/2016 összefűzött javaslat.pdf",
    #     "name_column": "NEV",
    # },
    # {
    #     "excel_sheet": "2017",
    #     "pdf_file": "javaslatok/2017 összefűzött javaslat.pdf",
    #     "name_column": "MEGNEVEZÉS",
    # },
    # {
    #     "excel_sheet": "2018",
    #     "pdf_file": "javaslatok/2018 összefűzött javaslat.pdf",
    #     "name_column": "MEGNEVEZÉS",
    # },
    # {
    #     "excel_sheet": "2019",
    #     "pdf_file": "javaslatok/2019 összefűzött javaslat.pdf",
    #     "name_column": "MEGNEVEZÉS",
    # },
    # {
    #     "excel_sheet": "2020",
    #     "pdf_file": "javaslatok/2020 összefűzött javaslat.pdf",
    #     "name_column": "MEGNEVEZÉS",
    # },
    {
        "excel_sheet": "2021",
        "pdf_file": "javaslatok/2021 összefűzött javaslat.pdf",
        "name_column": "MEGNEVEZÉS",
    },
]


def generate(prompt, file_path, temp):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    files = [
        client.files.upload(file=file_path),
    ]
    model = "gemini-2.5-flash-preview-05-20"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_uri(
                    file_uri=files[0].uri,
                    mime_type=files[0].mime_type,
                ),
                types.Part.from_text(text=prompt),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=temp,
        thinking_config=types.ThinkingConfig(
            thinking_budget=1024,
        ),
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type=genai.types.Type.OBJECT,
            required=["texts"],
            properties={
                "texts": genai.types.Schema(
                    type=genai.types.Type.ARRAY,
                    items=genai.types.Schema(
                        type=genai.types.Type.OBJECT,
                        required=["id", "text"],
                        properties={
                            "id": genai.types.Schema(
                                type=genai.types.Type.STRING,
                            ),
                            "text": genai.types.Schema(
                                type=genai.types.Type.STRING,
                            ),
                        },
                    ),
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


def to_roman_numeral(num):
    lookup = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ]
    res = ""
    for n, roman in lookup:
        (d, num) = divmod(num, n)
        res += roman * d
    return res


def from_roman_numeral(roman):
    """Convert a Roman numeral to an integer."""
    roman_dict = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}

    result = 0
    for i in range(len(roman)):
        # If current value is less than next value, subtract it
        if i < len(roman) - 1 and roman_dict[roman[i]] < roman_dict[roman[i + 1]]:
            result -= roman_dict[roman[i]]
        else:
            result += roman_dict[roman[i]]

    return result


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


def init_row(row):
    r = [0] * 5
    for i, v in enumerate(row.split(".")):
        if v.isdigit():
            r[i] = int(v)
        else:
            r[i] = v
    return r

def get_deduplicated_rows(df):
    numbered_rows = [
        init_row(row)
        for row in df["fid"]
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


def get_prompt_v1(section, filtered_rows, is_part):
    return f"""
Hierarchikusan strukturált költségvetéssel kapcsolatos indoklás szövegeket kell kinyerned.

Most minden szükséges információt a csatolt dokumentum "III."-mal jelzett részében találsz.

A fejezet a legmagasabb hierarchikus szint (római számmal jelölve), alatta található a cím (arab számmal), az alatt az alcím és végül a jogcímek.

Formátum:
Ezeket a részeket néha "/" jellel választják el (pl.: "(4/2/1)"), máskor szövegesen van jelölve, (pl.: "1. cím Bíróságok" vagy "3. cím 1. alcím").

Amikor ilyen egyértelmű utalás van alcímekre, bontsd fel a szöveget, de az egyéb magyarázó szövegeket, amik nem címhez tartoznak ne vedd bele.

Csak a konkrét címekkel foglalkozz, a többi bevezető szöveget hagyd figyelmen kívül. Például egy olyan részt, hogy "III.1" még önmagában nem feltétlen kell bevenni.

Az összes leírás egyben legyen meg egy adott fejezet/cím(/alcím/jogcímek) tételhez. Az indoklás szövege egy hosszabb, legalább 1-2 bekezdéses leírás.

Az id formátuma: fejezet.cím(.alcím.jocímek)
Pl.: VI.1.2.3

Nyerd ki strukturált módon a csatolt dokumentumból a benne szereplő teljes indoklás szövegeket tiszta markdown formátumban, szó szerint, a táblázatok nélkül!

Lehet, hogy a dokumentum nem tartalmazza a kért részeket{', hanem csak az elejét' if is_part else ''}. Ha egy adott részre nincs egyértelmű utalás a dokumentumban, akkor azt hagyd ki a kimeneti listából. Üres lista is egy valid válasz.

Az indoklás szövegeket szó szerint csak úgy vedd át, ahogy a csatolt dokumentumban szerepelnek.

Most csak a {section}. fejezet dokumentumát csatoltam. Ebből nyerd ki a következő részeket: {", ".join(filtered_rows)}
"""


def get_prompt(section, filtered_rows, names, is_part):
    return f"""
Hierarchikusan strukturált, költségvetéssel kapcsolatos indoklás szövegeket kell kinyerned azonosítók és nevek alapján.

Általában minden szükséges információt a csatolt dokumentum "III."-mal jelzett részében találsz.

A fejezet a legmagasabb hierarchikus szint (gyakran római számmal jelölve), alatta található a cím (arab számmal), az alatt az alcím és végül a jogcímek.

Ezeket a részeket néha "/" jellel választják el (pl.: "(4/2/1)"), máskor szövegesen van jelölve, (pl.: "1. cím Bíróságok" vagy "3. cím 1. alcím"). De az is előfordul, hogy az azonosító számok nem szerepelnek, csak a nevek. Ezeket a neveket is figyelembe kell venni.

Amikor ilyen egyértelmű utalás van alcímekre, bontsd fel a szöveget, de az egyéb magyarázó szövegeket, amik nem címhez tartoznak ne vedd bele.

Csak a konkrét címekkel foglalkozz, a többi bevezető szöveget hagyd figyelmen kívül. Például egy olyan részt, hogy "III.1" még önmagában nem feltétlen kell bevenni.

{'Lehet, hogy a dokumentum nem tartalmazza a kért részeket, hanem csak az elejét.' if is_part else 'Az összes leírás egyben legyen meg egy adott fejezet/cím(/alcím/jogcímek) tételhez.'} Az indoklás szövege egy hosszabb, legalább 1-2 bekezdéses leírás.

Az id formátuma: fejezet.cím(.alcím.jogcím1.jogcím2)
Pl.: VI.1.2.3

Nyerd ki strukturált módon a csatolt dokumentumból a benne szereplő teljes indoklás szövegeket tiszta markdown formátumban, szó szerint, a táblázatok nélkül!

Minden egyes azonosítóhoz kell, hogy legyen egy indoklás szöveg, ami a csatolt dokumentumban szerepel.

Az indoklás szövegeket szó szerint csak úgy vedd át, ahogy a csatolt dokumentumban szerepelnek.

Most csak a {section}. fejezet dokumentumát csatoltam. Ebből nyerd ki a következő részeket (azonosítók és nevek listája):
{"\n".join([f" - {r} ({n.strip()})" for r, n in zip(filtered_rows, names)])}
"""


def get_prompt_textgen(section, filtered_rows, names, is_part):
    return f"""
Hierarchikusan strukturált, költségvetéssel kapcsolatos indoklás szövegeket kell kinyerned azonosítók és nevek alapján.

Általában minden szükséges információt a csatolt dokumentum "III."-mal jelzett részében találsz.

A fejezet a legmagasabb hierarchikus szint (gyakran római számmal jelölve), alatta található a cím (arab számmal), az alatt az alcím és végül a jogcímek.

Ezeket a részeket néha "/" jellel választják el (pl.: "(4/2/1)"), máskor szövegesen van jelölve, (pl.: "1. cím Bíróságok" vagy "3. cím 1. alcím"). De az is előfordul, hogy az azonosító számok nem szerepelnek, csak a nevek. Ezeket a neveket is figyelembe kell venni.

Amikor ilyen egyértelmű utalás van alcímekre, bontsd fel a szöveget, de az egyéb magyarázó szövegeket, amik nem címhez tartoznak ne vedd bele.

Csak a konkrét címekkel foglalkozz, a többi bevezető szöveget hagyd figyelmen kívül. Például egy olyan részt, hogy "III.1" még önmagában nem feltétlen kell bevenni.

{'Lehet, hogy a dokumentum nem tartalmazza a kért részeket, hanem csak az elejét.' if is_part else 'Az összes leírás egyben legyen meg egy adott fejezet/cím(/alcím/jogcímek) tételhez.'} Az indoklás szövege egy hosszabb, legalább 1-2 bekezdéses leírás.

Az id formátuma: fejezet.cím(.alcím.jogcím1.jogcím2)
Pl.: VI.1.2.3

Nyerd ki strukturált módon a csatolt dokumentumból a benne szereplő teljes indoklás szövegeket tiszta markdown formátumban, szó szerint, a táblázatok nélkül!

Minden egyes azonosítóhoz kell, hogy legyen egy indoklás szöveg, ami a csatolt dokumentumban szerepel. Ha egy adott részre nincs egyértelmű utalás a dokumentumban, akkor azt hagyd ki a kimeneti listából. Üres lista is egy valid válasz.

Az indoklás szövegeket szó szerint csak úgy vedd át, ahogy a csatolt dokumentumban szerepelnek.

Most csak a {section}. fejezet dokumentumát csatoltam. Ebből nyerd ki a következő részeket (azonosítók és nevek listája):
{"\n".join([f" - {r} ({n.strip()})" for r, n in zip(filtered_rows, names)])}
"""


def to_numbers(row):
    """
    Convert a row of strings to numbers.
    """
    return [int(x) if str(x).isdigit() else x for x in row]

def extract_text_from_section(
    pdf_file, section, str_rows, names_by_fid, start_from=0, part=None
):
    is_part = part is not None
    roman_section = to_roman_numeral(section)
    filtered_rows = [
        f"{roman_section}.{'.'.join(row.split('.')[1:])}"
        for row in str_rows
        if row.startswith(f"{section}.")
    ]
    fids = [row for row in str_rows if row.startswith(f"{section}.")]
    names = [names_by_fid[r] for r in fids]

    filtered_rows = filtered_rows[start_from:]
    print(f"Filtered rows: {filtered_rows}, start_from: {start_from}")
    names = names[start_from:]

    success = []
    for i in range(4):
        if i < 2:
            prompt = get_prompt(roman_section, filtered_rows, names, is_part)
        elif i == 2:
            prompt = get_prompt_v1(roman_section, filtered_rows, is_part)
        elif i == 3:
            prompt = get_prompt_textgen(roman_section, filtered_rows, names, is_part)
        if roman_section == "XI":
            prompt = get_prompt_v1(roman_section, filtered_rows, is_part)
        temp = 0
        if i == 1:
            temp = 0.5
        # print("prompt")
        # print("---")
        # print(prompt)
        # print("---")

        generated = generate(prompt, pdf_file, temp)
        data = generated.parsed

        if data and "texts" in data and len(data["texts"]) > 0:
            try:
                success = [
                    1 if len(t["text"].strip()) > 80 else 0 for t in data["texts"]
                ]
                print(f"Generated success: {success}")
                if sum(success) / len(success) > 0.8 or is_part:
                    break
            except Exception as e:
                print(f"Error processing section {section}: {e}")
                continue
        else:
            print(roman_section)
            print("No texts found in the response. Retrying...")
            continue

    if success:
        last_success = -1
        for i, s in enumerate(success):
            if s == 1:
                last_success = i
    print(f"Last success: {last_success}")

    # print(data)
    # print(data["texts"])

    for text in data["texts"]:
        text["id"] = text["id"].replace(roman_section, str(section))
        # print(text["id"])
        # print(text["indoklás szöveg"])
        # print("\n\n")

    # Create directory for extracted descriptions if it doesn't exist
    extracted_dir = "extracted_descriptions"
    os.makedirs(extracted_dir, exist_ok=True)

    # Create a filename based on year and section
    if part is not None:
        part = f"_{part+1}"
    else:
        part = ""
    csv_filename = f"{extracted_dir}/{excel_sheet}_section_{section}{part}.csv"

    # Convert data to DataFrame and save as CSV
    descriptions_df = pd.DataFrame(data["texts"])
    descriptions_df.to_csv(csv_filename, index=False, encoding="utf-8")

    print(f"Saved extracted descriptions to {csv_filename}")

    return last_success


def split_pdf_by_pages(pdf_path, output_dir, name_prefix, splits):
    """Split the PDF file into sections based on page ranges."""
    os.makedirs(output_dir, exist_ok=True)

    with open(pdf_path, "rb") as file:
        pdf = PyPDF2.PdfReader(file)
        total_pages = len(pdf.pages)

        # Calculate base pages per split
        pages_per_split = total_pages // splits

        # Create list to store created file paths
        split_files = []

        for i in range(splits):
            output_pdf = PyPDF2.PdfWriter()

            # Calculate start and end page for this split
            start_page = i * pages_per_split
            end_page = (i + 1) * pages_per_split if i < splits - 1 else total_pages

            # Add pages to the new PDF
            for page_num in range(start_page, end_page):
                output_pdf.add_page(pdf.pages[page_num])

            # Save the split PDF
            output_file = os.path.join(output_dir, f"{name_prefix}_{i+1}.pdf")
            with open(output_file, "wb") as out_file:
                output_pdf.write(out_file)

            split_files.append(output_file)
            print(f"Created split PDF: {output_file} (pages {start_page+1}-{end_page})")

        return split_files


for year in years:
    name_column = year["name_column"]
    excel_sheet = year["excel_sheet"]
    pdf_file = year["pdf_file"]

    df = pd.read_excel(excel_file, sheet_name=excel_sheet)
    df.columns = df.iloc[0]
    df = df[1:]
    df["ALCIM"].fillna(0, inplace=True)
    df["JOGCIM1"].fillna(0, inplace=True)
    df["JOGCIM2"].fillna(0, inplace=True)
    df = df[df["FEJEZET"].notna()]


    print(df.head(10))
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
    print(numbered_rows[:50])  # Print first 10 for debugging

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
                    else: # current_row_sparse[i] is 0
                        # Inherit from prev_filled_row as context hasn't changed at this point.
                        new_filled_row[i] = prev_filled_row[i]
        
        filled_rows.append(new_filled_row)
        prev_filled_row = new_filled_row # Update prev_filled_row for the next iteration
    
    print("Filled rows:")
    print(filled_rows[:50])  # Print first 10 for debugging

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

    print("Fids:")
    print(df["fid"].head(100))  # Print first 10 for debugging

    df["fid"] = (
        df["fid"]
        .str.replace(r"\.0$", "", regex=True)
        .replace(r"\.0$", "", regex=True)
        .replace(r"\.0$", "", regex=True)
        .replace(r"\.0$", "", regex=True)
        .replace(r"\.0$", "", regex=True)
    )

    deduplicated_rows = get_deduplicated_rows(df)
    df = df[df["fid"].isin(deduplicated_rows)]
    names = df[name_column].tolist()
    names_by_fid = {}
    for i, row in df.iterrows():
        fid = row["fid"]
        name = row[name_column]
        if fid not in names_by_fid:
            names_by_fid[fid] = name

    # for r in str_rows:
    #     print(r)

    # print(len(str_rows), len(set(str_rows)))

    with open(f"{excel_sheet}_section_structure.json", "r") as f:
        section_structures = list(json.load(f).items())

    filtered_sections = section_structures[0:]
    # filtered_sections = [s for s in section_structures if s[0].startswith("XX.")]

    for title, section in tqdm(filtered_sections):
        section_number = from_roman_numeral(title.split(" ")[0].strip("."))
        pdf_file = section["file_path"]
        # print(f"names_by_fid: {names_by_fid}")

        try:
            page_count = get_page_lenth(pdf_file)
            if page_count > 100:
                # Split the PDF into 100 page chunks to page count
                splits = page_count // 100 + 1
                split_files = split_pdf_by_pages(
                    pdf_file,
                    "split",
                    f"{excel_sheet}_{section_number}",
                    splits,
                )
                print(f"Split PDF into {len(split_files)} parts.")
                last_success = 0
                for i, split_file in enumerate(split_files):
                    print(f"Processing split file: {split_file}")
                    last_success += extract_text_from_section(
                        split_file,
                        section_number,
                        deduplicated_rows,
                        names_by_fid,
                        start_from=last_success,
                        part=i,
                    )
            else:
                extract_text_from_section(
                    pdf_file, section_number, deduplicated_rows, names_by_fid
                )
        except Exception as e:
            print(f"Error processing section {section_number}: {e}")
            stack_trace = traceback.format_exc()
            print(stack_trace)
            continue
