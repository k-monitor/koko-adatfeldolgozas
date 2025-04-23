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

load_dotenv(dotenv_path=".env")

excel_file = "xlsx/Elfogadott költségvetések.xlsx"

years = [
    # {"excel_sheet": "2016", "pdf_file": "javaslatok/2016 összefűzött javaslat.pdf"},
    {"excel_sheet": "2017", "pdf_file": "javaslatok/2017 összefűzött javaslat.pdf"},
]


def generate(prompt, file_path):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    files = [
        client.files.upload(file=file_path),
    ]
    model = "gemini-2.5-flash-preview-04-17"
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
        temperature=0,
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type=genai.types.Type.OBJECT,
            required=["texts"],
            properties={
                "texts": genai.types.Schema(
                    type=genai.types.Type.ARRAY,
                    items=genai.types.Schema(
                        type=genai.types.Type.OBJECT,
                        required=["id", "indoklás szöveg"],
                        properties={
                            "id": genai.types.Schema(
                                type=genai.types.Type.STRING,
                            ),
                            "indoklás szöveg": genai.types.Schema(
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


def get_deduplicated_rows(df):
    numbered_rows = [
        to_numbers(row)
        for row in df[[f"Unnamed: {n}" for n in range(3, 7)]].itertuples(
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


def get_prompt(section, filtered_rows):
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

Lehet, hogy a dokumentum nem tartalmazza a kért részeket. Ha egy adott részre nincs egyértelmű utalás a dokumentumban, akkor azt hagyd ki a kimeneti listából. Üres lista is egy valid válasz.

Most csak a {section}. fejezet dokumentumát csatoltam. Ebből nyerd ki a következő részeket: {", ".join(filtered_rows)}
"""


def extract_text_from_section(pdf_file, section, str_rows, part=None):
    roman_section = to_roman_numeral(section)
    filtered_rows = [
        f"{roman_section}.{'.'.join(row.split('.')[1:])}"
        for row in str_rows
        if row.startswith(f"{section}.")
    ]

    prompt = get_prompt(roman_section, filtered_rows)
    # print(prompt)

    generated = generate(prompt, pdf_file)
    data = generated.parsed
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


for year in years:
    excel_sheet = year["excel_sheet"]
    pdf_file = year["pdf_file"]

    df = pd.read_excel(excel_file, sheet_name=excel_sheet)
    str_rows = get_deduplicated_rows(df)
    # for r in str_rows:
    #     print(r)

    # print(len(str_rows), len(set(str_rows)))

    with open(f"{excel_sheet}_section_structure.json", "r") as f:
        section_structures = list(json.load(f).items())

    filtered_sections = section_structures[0:]
    # filtered_sections = [s for s in section_structures if s[0] == "IX. Helyi Önkormányzatok Támogatásai"]

    for title, section in tqdm(filtered_sections):
        section_number = from_roman_numeral(title.split(" ")[0].strip("."))
        pdf_file = section["file_path"]
        try:
            extract_text_from_section(pdf_file, section_number, str_rows)
        except Exception as e:
            print(f"Error processing section {section_number}: {e}")
            continue
