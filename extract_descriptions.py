import pandas as pd
import os
import PyPDF2
from google import genai
from google.genai import types
from dotenv import load_dotenv
import json
from tqdm import tqdm
import traceback
import logging
import time
from typing import Tuple, Any
from utils.pdf_extractor import get_page_lenth

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Silence pdfminer warnings
logging.getLogger("pdfminer").setLevel(logging.ERROR)

load_dotenv(override=True)

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    logger.error("GEMINI_API_KEY environment variable is not set.")
    raise ValueError("GEMINI_API_KEY environment variable is required.")

logger.info(f"using api key: ***{api_key[-5:]}")

# Configuration constants
EXCEL_FILE = "adatok/koltsegvetesek.xlsx"
PROCESSED_DIR = "indoklasok/feldolgozott"
EXTRACTED_DIR = "indoklasok/szovegek"
MODEL_NAME = "gemini-2.5-flash-preview-09-2025"
YEARS = [
    "2016",
    "2017",
    # "2018",
    # "2019",
    # "2020",
    # "2021",
    # "2022",
    # "2023",
    # "2024",
    # "2025",
    # "2026",
]
# Maximum pages before splitting (to avoid Gemini token limits)
# Gemini has ~1M token context, ~100 pages ≈ 200k tokens
MAX_PAGES_PER_SPLIT = 100

# Minimum text length to consider extraction successful
# Chosen based on typical indoklás length (usually 1-2 paragraphs)
MIN_TEXT_LENGTH = 80

# Success threshold for non-split sections
# Set to 80% to allow for some sections without descriptions
SUCCESS_THRESHOLD = 0.8

# Retry attempts for API failures
MAX_RETRIES = 4


def log_error(context: str, error: Exception, include_traceback: bool = True) -> None:
    """Standardized error logging."""
    logger.error(f"{context}: {error}")
    if include_traceback:
        logger.error(traceback.format_exc())


def validate_extracted_text(text_item: dict, expected_section: int) -> bool:
    """Validate extracted text item."""
    if not text_item.get("id") or not text_item.get("text"):
        return False
    
    # Check ID format matches expected section
    if not text_item["id"].startswith(f"{expected_section}."):
        logger.warning(f"ID {text_item['id']} doesn't match section {expected_section}")
        return False
    
    # Check text quality
    text = text_item["text"].strip()
    if len(text) < MIN_TEXT_LENGTH:
        return False
    
    # Check for common extraction errors (too many encoding errors)
    if text.count("�") > 5:
        logger.warning(f"Text for {text_item['id']} has encoding issues")
        return False
    
    return True


def generate(prompt: str, file_path: str, temp: float) -> Any:
    """Generate content using Gemini API."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    uploaded_file = None
    try:
        uploaded_file = client.files.upload(file=file_path)
        files = [uploaded_file]
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
            thinking_config=types.ThinkingConfig(thinking_budget=1024),
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
                                "id": genai.types.Schema(type=genai.types.Type.STRING),
                                "text": genai.types.Schema(type=genai.types.Type.STRING),
                            },
                        ),
                    ),
                },
            ),
        )

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=generate_content_config,
        )
        return response
    finally:
        # Clean up uploaded file
        if uploaded_file:
            try:
                client.files.delete(name=uploaded_file.name)
                logger.debug(f"Deleted uploaded file: {uploaded_file.name}")
            except Exception as e:
                logger.warning(f"Failed to delete uploaded file: {e}")


def to_roman_numeral(num: int) -> str:
    """Convert an integer to Roman numeral."""
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
    result = ""
    for value, numeral in lookup:
        count, num = divmod(num, value)
        result += numeral * count
    return result


def from_roman_numeral(roman: str) -> int:
    """Convert a Roman numeral to an integer."""
    roman_dict = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    result = 0
    for i in range(len(roman)):
        if i < len(roman) - 1 and roman_dict[roman[i]] < roman_dict[roman[i + 1]]:
            result -= roman_dict[roman[i]]
        else:
            result += roman_dict[roman[i]]
    return result


def to_numbers(row: list[Any]) -> list[Any]:
    """Convert a row of strings to numbers where possible."""
    return [int(x) if str(x).isdigit() else x for x in row]


def init_row(row_str: str) -> list[Any]:
    """Initialize a row from string format."""
    result = [0] * 5
    for i, value in enumerate(row_str.split(".")):
        if value.isdigit():
            result[i] = int(value)
        else:
            result[i] = value
    return result


def positive_length(row: list[Any]) -> int:
    """Count positive integers in a row."""
    return len([x for x in row if isinstance(x, int) and x > 0])


def get_deduplicated_rows(df: pd.DataFrame) -> list[str]:
    """Get deduplicated rows from DataFrame."""
    numbered_rows = [init_row(row) for row in df["fid"]]
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

    # Clean up redundant .0 patterns
    str_rows = []
    for row in deduplicated_rows:
        fid = ".".join([str(i) for i in row])
        while ".0" in fid:
            fid = fid.replace(".0", "")
        str_rows.append(fid)
    return str_rows


def get_prompt(
    section: str, filtered_rows: list[str], names: list[str], is_part: bool
) -> str:
    """Generate prompt for text extraction."""
    part_text = (
        "Lehet, hogy a dokumentum nem tartalmazza a kért részeket, hanem csak az elejét."
        if is_part
        else "Az összes leírás egyben legyen meg egy adott fejezet/cím(/alcím/jogcímek) tételhez."
    )

    return f"""
Hierarchikusan strukturált, költségvetéssel kapcsolatos indoklás szövegeket kell kinyerned azonosítók és nevek alapján.

Általában minden szükséges információt a csatolt dokumentum "III."-mal jelzett részében találsz.

A fejezet a legmagasabb hierarchikus szint (gyakran római számmal jelölve), alatta található a cím (arab számmal), az alatt az alcím és végül a jogcímek.

Ezeket a részeket néha "/" jellel választják el (pl.: "(4/2/1)"), máskor szövegesen van jelölve, (pl.: "1. cím Bíróságok" vagy "3. cím 1. alcím"). De az is előfordul, hogy az azonosító számok nem szerepelnek, csak a nevek. Ezeket a neveket is figyelembe kell venni.

Amikor ilyen egyértelmű utalás van alcímekre, bontsd fel a szöveget, de az egyéb magyarázó szövegeket, amik nem címhez tartoznak ne vedd bele.

Csak a konkrét címekkel foglalkozz, a többi bevezető szöveget hagyd figyelmen kívül. Például egy olyan részt, hogy "III.1" még önmagában nem feltétlen kell bevenni.

{part_text} Az indoklás szövege egy hosszabb, legalább 1-2 bekezdéses leírás.

Az id formátuma: fejezet.cím(.alcím.jogcím1.jogcím2)
Pl.: VI.1.2.3

Nyerd ki strukturált módon a csatolt dokumentumból a benne szereplő teljes indoklás szövegeket tiszta markdown formátumban, szó szerint, a táblázatok nélkül!

Minden egyes azonosítóhoz kell, hogy legyen egy indoklás szöveg, ami a csatolt dokumentumban szerepel. Ha egy adott részre nincs egyértelmű utalás a dokumentumban, akkor azt hagyd ki a kimeneti listából. Üres lista is egy valid válasz.

Az indoklás szövegeket szó szerint csak úgy vedd át, ahogy a csatolt dokumentumban szerepelnek.

Most csak a {section}. fejezet dokumentumát csatoltam. Ebből nyerd ki a következő részeket (azonosítók és nevek listája):
{chr(10).join([f" - {r} ({n.strip()})" for r, n in zip(filtered_rows, names)])}
"""


def split_pdf_by_pages(
    pdf_path: str, output_dir: str, name_prefix: str, splits: int
) -> list[str]:
    """Split the PDF file into sections based on page ranges."""
    os.makedirs(output_dir, exist_ok=True)
    split_files = []

    with open(pdf_path, "rb") as file:
        pdf = PyPDF2.PdfReader(file)
        total_pages = len(pdf.pages)
        pages_per_split = total_pages // splits

        for i in range(splits):
            output_pdf = PyPDF2.PdfWriter()
            start_page = i * pages_per_split
            end_page = (i + 1) * pages_per_split if i < splits - 1 else total_pages

            for page_num in range(start_page, end_page):
                output_pdf.add_page(pdf.pages[page_num])

            output_file = os.path.join(output_dir, f"{name_prefix}_{i+1}.pdf")
            with open(output_file, "wb") as out_file:
                output_pdf.write(out_file)

            split_files.append(output_file)
            logger.info(
                f"Created split PDF: {output_file} (pages {start_page+1}-{end_page})"
            )

    return split_files


def extract_text_from_section(
    pdf_file: str,
    section: int,
    str_rows: list[str],
    names_by_fid: dict[str, str],
    excel_sheet: str,
    start_from: int = 0,
    part: int | None = None,
) -> Tuple[int, list[dict[str, str]]]:
    """Extract text from a specific section of the PDF."""
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
    names = names[start_from:]

    logger.info(f"Filtered rows: {filtered_rows}, start_from: {start_from}")

    # Try different prompts and temperatures
    extraction_quality = []
    data = None
    temperatures = [0, 0.3, 0.5, 0.7]
    
    for attempt in range(MAX_RETRIES):
        temp = temperatures[min(attempt, len(temperatures)-1)]
        prompt = get_prompt(roman_section, filtered_rows, names, is_part)

        try:
            # Add exponential backoff delay for retries
            if attempt > 0:
                delay = min(2 ** attempt, 10)  # Max 10 seconds
                logger.info(f"Waiting {delay}s before retry {attempt + 1}/{MAX_RETRIES}")
                time.sleep(delay)
            
            generated = generate(prompt, pdf_file, temp)
            data = generated.parsed

            if data and "texts" in data and len(data["texts"]) > 0:
                extraction_quality = [
                    1 if len(t["text"].strip()) > MIN_TEXT_LENGTH else 0
                    for t in data["texts"]
                ]
                logger.info(f"Extraction quality: {extraction_quality}")

                if extraction_quality and (sum(extraction_quality) / len(extraction_quality) > SUCCESS_THRESHOLD or is_part):
                    break
            else:
                logger.warning(
                    f"No texts found in response for section {roman_section}. Retrying..."
                )
                continue

        except Exception as e:
            log_error(f"Error processing section {section}, attempt {attempt}", e, include_traceback=False)
            continue

    if not extraction_quality or not data:
        logger.error(f"Failed to extract text from section {section}")
        return 0, []

    last_success_idx = max([i for i, s in enumerate(extraction_quality) if s == 1], default=-1)
    logger.info(f"Last successful extraction index: {last_success_idx}")

    # Process results and convert roman numerals back to numbers, with validation
    extracted_texts = []
    for text in data["texts"]:
        text["id"] = text["id"].replace(roman_section, str(section))
        if validate_extracted_text(text, section):
            extracted_texts.append(text)
        else:
            logger.warning(f"Skipping invalid text: {text.get('id', 'unknown')}")

    return last_success_idx, extracted_texts


def process_dataframe(df: pd.DataFrame) -> Tuple[list[str], dict[str, str]]:
    """Process DataFrame to create FIDs and name mappings."""
    # Fill NaN values
    df = df.fillna({"CIM": 0, "ALCIM": 0, "JOGCIM1": 0, "JOGCIM2": 0})

    df = df[df["FEJEZET"].notna()]

    # Convert to numbered rows and fill missing values
    numbered_rows = [
        to_numbers(row)
        for row in df[["FEJEZET", "CIM", "ALCIM", "JOGCIM1", "JOGCIM2"]].itertuples(
            index=False, name=None
        )
    ]

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

    # Create FIDs
    str_rows = []
    for row in filled_rows:
        fid = ".".join([str(i) for i in row])
        while ".0" in fid:
            fid = fid.replace(".0", "")
        str_rows.append(fid)
    df["fid"] = str_rows

    # Clean up FIDs
    df["fid"] = df["fid"].str.replace(r"\.0$", "", regex=True)

    # Get deduplicated rows and create name mapping
    deduplicated_rows = get_deduplicated_rows(df)
    df = df[df["fid"].isin(deduplicated_rows)]

    names_by_fid = {}
    for _, row in df.iterrows():
        fid = row["fid"]
        name = row["MEGNEVEZÉS"]
        if fid not in names_by_fid:
            names_by_fid[fid] = name

    return deduplicated_rows, names_by_fid


def process_year(year: str) -> None:
    """Process a single year's data."""
    logger.info(f"Processing year: {year}")

    excel_sheet = year
    pdf_file = f"{PROCESSED_DIR}/{year}.pdf"
    all_extracted_texts = []

    try:
        df = pd.read_excel(EXCEL_FILE, sheet_name=excel_sheet)
        df = df[1:]  # Skip first row

        deduplicated_rows, names_by_fid = process_dataframe(df)

        with open(f"{PROCESSED_DIR}/{excel_sheet}/summary.json", "r") as f:
            section_structures = list(json.load(f).items())

        for title, section in tqdm(section_structures):
            section_number = from_roman_numeral(title.split(" ")[0].strip("."))
            # Use file_path directly - it already contains the full path
            section_pdf_file = section["file_path"]

            try:
                page_count = get_page_lenth(section_pdf_file)

                if page_count > MAX_PAGES_PER_SPLIT:
                    splits = page_count // MAX_PAGES_PER_SPLIT + 1
                    split_files = split_pdf_by_pages(
                        section_pdf_file,
                        "split",
                        f"{excel_sheet}_{section_number}",
                        splits,
                    )
                    logger.info(f"Split PDF into {len(split_files)} parts.")

                    last_success_idx = 0
                    for i, split_file in enumerate(split_files):
                        logger.info(f"Processing split file: {split_file}")
                        last_idx, section_texts = extract_text_from_section(
                            split_file,
                            section_number,
                            deduplicated_rows,
                            names_by_fid,
                            excel_sheet,
                            start_from=last_success_idx,
                            part=i,
                        )
                        all_extracted_texts.extend(section_texts)
                        # Move forward by the number of items we successfully processed
                        if last_idx >= 0:
                            last_success_idx += (last_idx + 1)  # +1 because index is 0-based
                else:
                    _, section_texts = extract_text_from_section(
                        section_pdf_file,
                        section_number,
                        deduplicated_rows,
                        names_by_fid,
                        excel_sheet,
                    )
                    all_extracted_texts.extend(section_texts)

            except Exception as e:
                log_error(f"Error processing section {section_number}", e)
                continue

        # Save all extracted texts for the year in one CSV file
        if all_extracted_texts:
            os.makedirs(EXTRACTED_DIR, exist_ok=True)
            csv_filename = f"{EXTRACTED_DIR}/{excel_sheet}.csv"
            descriptions_df = pd.DataFrame(all_extracted_texts)
            descriptions_df.to_csv(csv_filename, index=False, encoding="utf-8")
            logger.info(
                f"Saved all extracted descriptions for {year} to {csv_filename}"
            )
        else:
            logger.warning(f"No texts extracted for year {year}")

    except Exception as e:
        log_error(f"Error processing year {year}", e)


def main() -> None:
    """Main function to process all years."""
    for year in YEARS:
        process_year(year)


if __name__ == "__main__":
    main()
