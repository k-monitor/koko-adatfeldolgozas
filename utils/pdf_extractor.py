from typing import List
from pdfminer.high_level import extract_pages, extract_text # Import necessary pdfminer components
from pdfminer.layout import LTTextContainer, LAParams # Import LAParams
from pdfminer.pdfparser import PDFSyntaxError # Specific pdfminer error
from pdfminer.psparser import PSSyntaxError # Another potential parsing error

def extract_text_by_page(pdf_path: str) -> List[str]:
    """
    Extracts text from each page of a PDF file using pdfminer.six, preserving line breaks.

    Args:
        pdf_path: The path to the PDF file.

    Returns:
        A list of strings, where each string is the text extracted from a page.
        Returns an empty list if the file cannot be opened or is not a PDF.
    """
    pages_text: List[str] = []
    # Set LAParams to better control layout analysis, potentially improving line break handling
    laparams = LAParams() 
    try:
        page_count = 0
        # Pass laparams to extract_pages
        for page_layout in extract_pages(pdf_path, laparams=laparams):
            page_count += 1
            page_content_parts = [] # Use a list to collect text parts
            # Iterate through elements in the page layout
            for element in page_layout:
                # Check if the element is a text container
                if isinstance(element, LTTextContainer):
                    # Append the text directly, preserving its internal newlines
                    page_content_parts.append(element.get_text()) 
            
            # Join the collected parts
            page_content = "".join(page_content_parts)

            if page_content:
                # Just strip leading/trailing whitespace from the whole page text
                cleaned_text = page_content.strip() 
                pages_text.append(cleaned_text)
            else:
                 pages_text.append("") # Append empty string if no text found
                 print(f"Warning: No text extracted from page {page_count}")

        print(f"Processed {page_count} pages in {pdf_path}")

    except FileNotFoundError:
        print(f"Error: File not found at {pdf_path}")
    except (PDFSyntaxError, PSSyntaxError) as e: # Catch specific pdfminer parsing errors
        print(f"Error reading PDF file {pdf_path}. It might be corrupted or malformed: {e}")
    except Exception as e:
        # Catch other potential errors during processing
        print(f"An unexpected error occurred processing {pdf_path} with pdfminer.six: {e}")

    return pages_text

if __name__ == '__main__':
    # Replace 'your_document.pdf' with the actual path to your PDF file
    pdf_file = 'your_document.pdf'
    extracted_data = extract_text_by_page(pdf_file)

    if extracted_data:
        for i, page_text in enumerate(extracted_data):
            print(f"--- Page {i + 1} Text ---")
            print(page_text[:200] + "..." if len(page_text) > 200 else page_text) # Print first 200 chars
            print("-" * 20)
    else:
        print("No text was extracted.")
