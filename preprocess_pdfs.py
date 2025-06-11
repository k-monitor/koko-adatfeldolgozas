# imports
from utils.pdf_extractor import extract_text_by_page
from collections import defaultdict
import pandas as pd
import os
import PyPDF2  # Add this import for PDF splitting
import re
import json
import logging
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

excel_file = "adatok/koltsegvetesek.xlsx"

years = [
    "2016",
    "2017",
    "2018",
    "2019",
]


def find_section_boundaries(text_by_page, sections):
    """Find section boundaries in the PDF."""
    # Find all section start pages
    section_start_pages = {}
    for n, page in enumerate(text_by_page):
        cpage = page.replace("  ", " ").lower().strip()
        for section in sections:
            if section.lower()[:30] in cpage and len(cpage) < 100:
                if section not in section_start_pages:
                    section_start_pages[section] = n
            elif section.lower()[:30] in cpage and len(cpage) < 500:
                print(
                    f"Section '{section}' found on page {n} but skipped due to length: {len(cpage)}"
                )

    # Sort sections by their start page
    sorted_sections = sorted(section_start_pages.items(), key=lambda x: x[1])

    # Create a dictionary with section boundaries (start and end pages)
    section_boundaries = {}
    for i, (section, start_page) in enumerate(sorted_sections):
        if i < len(sorted_sections) - 1:
            end_page = sorted_sections[i + 1][1] - 1
        else:
            # For the last section, use the last page
            end_page = len(text_by_page) - 1
        section_boundaries[section] = {"start_page": start_page, "end_page": end_page}

    return section_boundaries


def extract_title_from_page(page_text):
    """Extract a title from page text."""
    if len(page_text.split("\n")) >= 3:
        title = "\n".join(page_text.split("\n")[1:-1]).strip()
    elif len(page_text.split("\n")) == 2:
        title = page_text.split("\n")[0].strip()
    else:
        title = page_text.strip()
    return title


def find_subsections(text_by_page, section, boundaries):
    """Find subsections within a section."""
    start_page = boundaries["start_page"]
    end_page = boundaries["end_page"]

    # Extract the main section title from the start page
    main_page_text = text_by_page[start_page]
    main_title = extract_title_from_page(main_page_text)

    # Initialize subsections with the main section
    subsections = [
        {
            "title": main_title,
            "page": start_page,
            "is_main_section": True,
            "start_page": start_page,
            "end_page": end_page,
            "page_count": end_page - start_page + 1,
        },
        # First subsection is always the main page
        {
            "title": main_title,
            "page": start_page,
            "is_main_section": False,
            "start_page": start_page,
            "is_first_subsection": True,
        },
    ]

    # Look for additional subsections within the section's range of pages
    for n in range(start_page + 1, end_page + 1):
        page = text_by_page[n]
        cpage = page.replace("  ", " ").lower().strip()

        # Check if this page might be a subsection
        if len(cpage) < 100:
            title = extract_title_from_page(page)

            # Add as a subsection if it has content
            if title and len(title) > 1:
                subsections.append(
                    {
                        "title": title,
                        "page": n,
                        "is_main_section": False,
                        "start_page": n,
                    }
                )

    return subsections


def calculate_subsection_boundaries(subsections, section_end_page):
    """Calculate end pages for subsections."""
    # Sort subsections by page number (excluding the main section entry)
    only_subsections = sorted(
        [e for e in subsections if not e.get("is_main_section")],
        key=lambda x: x["page"],
    )

    # If there are no additional subsections beyond the first one
    if len(only_subsections) == 1:
        only_subsections[0]["end_page"] = section_end_page
        only_subsections[0]["page_count"] = (
            section_end_page - only_subsections[0]["start_page"] + 1
        )
        return subsections

    # Calculate end pages for each subsection
    for i, subsection in enumerate(only_subsections):
        if i < len(only_subsections) - 1:
            # End page is one less than the start of the next subsection
            subsection["end_page"] = only_subsections[i + 1]["start_page"] - 1
        else:
            # Last subsection ends at the section end
            subsection["end_page"] = section_end_page

        # Calculate page count
        subsection["page_count"] = subsection["end_page"] - subsection["start_page"] + 1

    return subsections


def sanitize_filename(name):
    """Sanitize a string to be used as a filename."""
    # Replace non-alphanumeric characters with underscores
    sanitized = re.sub(r"[^\w\s-]", "_", name)
    # Replace multiple spaces with a single underscore
    sanitized = re.sub(r"\s+", "_", sanitized)
    return sanitized


def split_pdf_by_sections(pdf_path, output_dir, section_summary, year_prefix):
    """Split the PDF file into sections based on page ranges."""
    os.makedirs(output_dir, exist_ok=True)

    with open(pdf_path, "rb") as file:
        pdf = PyPDF2.PdfReader(file)

        for section_name, section_info in section_summary.items():
            section_sanitized = sanitize_filename(section_name)

            # Create PDF writer instance
            pdf_writer = PyPDF2.PdfWriter()

            # Get the main section information
            main_section = section_info["main"]
            start_page = main_section["start_page"]
            end_page = main_section["end_page"]

            # Add all pages for this section
            for page_num in range(start_page, end_page + 1):
                if page_num < len(pdf.pages):
                    pdf_writer.add_page(pdf.pages[page_num])

            # Create meaningful filename
            section_title = sanitize_filename(main_section["title"])
            if len(section_title) > 30:
                section_title = section_title[:30]  # Limit length

            filename = f"{year_prefix}_{section_sanitized}_{section_title}.pdf"
            file_path = os.path.join(output_dir, filename)

            # Save the section PDF
            with open(file_path, "wb") as output_pdf:
                pdf_writer.write(output_pdf)

            # Add file path to section info
            section_info["file_path"] = file_path

    return section_summary


def prepare_export_summary(section_summary):
    """Prepare a serializable version of the section summary."""
    export_summary = {}
    for section, info in section_summary.items():
        export_summary[section] = {"file_path": info.get("file_path", "")}
    return export_summary


def process_year(
    excel_sheet: str, pdf_file: str, output_dir: str = "split_pdfs"
) -> str:
    """Process a single year's data and return the JSON file path."""
    if not os.path.exists(pdf_file):
        raise FileNotFoundError(f"PDF file not found: {pdf_file}")

    logger.info(f"Processing year {excel_sheet} with PDF {pdf_file}")

    df = pd.read_excel(excel_file, sheet_name=excel_sheet)
    df = df.dropna(subset=["MEGNEVEZÉS"])
    sections = [
        s.split(" ")[0] + " " + " ".join(s.split(" ")[1:]).title()
        for s in list(df["MEGNEVEZÉS"])[1:]
        if not s.startswith(" ")
    ]
    text_by_page = extract_text_by_page(pdf_path=pdf_file)

    # Extract year from pdf_file for use in output filenames
    year_match = re.search(r"(\d{4})", pdf_file)
    year_prefix = year_match.group(1) if year_match else "unknown_year"

    # Find section boundaries
    section_boundaries = find_section_boundaries(text_by_page, sections)

    # Initialize parsed_sections with defaultdict that will hold subsections
    parsed_sections = defaultdict(list)

    # Process all pages to identify subsections within sections
    for section, boundaries in section_boundaries.items():
        # Find subsections in this section
        subsections = find_subsections(text_by_page, section, boundaries)
        # Add all subsections to the parsed_sections dictionary
        parsed_sections[section].extend(subsections)
        # Calculate end pages and page counts for subsections
        parsed_sections[section] = calculate_subsection_boundaries(
            parsed_sections[section], boundaries["end_page"]
        )

    # Prepare the section summary
    section_summary = {}
    for section, entries in parsed_sections.items():
        main_section = next((e for e in entries if e.get("is_main_section")), None)
        # Filter subsections - excluding the main section entry
        subsections = [e for e in entries if not e.get("is_main_section")]

        # Create section summary with detailed subsection information
        section_summary[section] = {
            "main": main_section,
            "page_count": main_section["page_count"] if main_section else 0,
            "subsections": subsections,
            "subsection_count": len(subsections),
        }

    # Split the PDF and update section_summary with file paths
    section_summary = split_pdf_by_sections(
        pdf_file, output_dir, section_summary, year_prefix
    )

    export_summary = prepare_export_summary(section_summary)

    # Save to disk with proper error handling
    os.makedirs(f"indoklasok/feldolgozott/{year_prefix}", exist_ok=True)
    json_file_path = f"indoklasok/feldolgozott/{year_prefix}/summary.json"

    try:
        with open(json_file_path, "w", encoding="utf-8") as f:
            json.dump(export_summary, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved processed data to {json_file_path}")
    except Exception as e:
        logger.error(f"Error saving JSON file {json_file_path}: {e}")
        traceback.print_exc()
        raise

    return json_file_path


def main():
    """Main processing function."""
    # Process all years and collect JSON file paths
    json_files = []
    for year in years:
        try:
            json_file = process_year(
                year,
                f"indoklasok/nyers/javaslatok/{year}.pdf",
                output_dir=f"indoklasok/feldolgozott/{year}",
            )
            json_files.append(json_file)
        except Exception as e:
            logger.error(f"Failed to process year {year}: {e}")
            traceback.print_exc()
            continue

    logger.info(f"Generated JSON files: {json_files}")
    return json_files


if __name__ == "__main__":
    main()
