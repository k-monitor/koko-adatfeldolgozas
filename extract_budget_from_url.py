#!/usr/bin/env python3
"""
Extract budget data from Hungarian legal documents (jogtar.hu, net.jogtar.hu, jogkodex.hu)

Usage:
    python extract_budget_from_url.py --url <URL> --variant <VARIANT> --output <OUTPUT_FILE>

Variants:
    - jogtar_lapozos: For jogtar.hu URLs (multi-page, 6 numcols, 5 sumrows)
    - jogtar: For jogtar.hu URLs (single page, 6 numcols, 5 sumrows)
    - jogkodex: For jogkodex.hu URLs (single page, 6 numcols, 4 sumrows, different name column)
"""

import argparse
import csv
import requests
from bs4 import BeautifulSoup


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


def extract_jogtar_lapozos(base_url: str, num_pages: int = 6, verbose: bool = False):
    """
    Extract data from jogtar.hu (multi-page format)

    Args:
        base_url: Base URL (e.g., "https://mkogy.jogtar.hu/jogszabaly?docid=A2500069.TV")
        num_pages: Number of pages to scrape
        verbose: Print debug information
        
    Returns:
        tuple: (table_rows, fejezetek)
    """
    table_rows = []
    fejezetek = []
    
    for i in range(num_pages):
        url = f"{base_url}&pagenum={i+1}"
        if verbose:
            print(f"Fetching page {i+1}: {url}")
        
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for table in soup.select("table"):
            rows = table.select("tr")
            for row in rows:
                cols = [td.text for td in row.select("td")]
                numcols = cols[:6]
                
                # Check for chapter header
                if any([col.strip() and not col.strip().isdigit() for col in numcols]):
                    if verbose:
                        print("Non-num skip:", cols)
                    if len(cols) > 1 and "Kiadás" == cols[1].strip():
                        fejezet = cols[0].strip()
                        if '. ' not in fejezet:
                            if verbose:
                                print("Fejezet skip:", fejezet)
                            continue
                        fejezetszam = from_roman_numeral(fejezet.split('. ')[0])
                        if verbose:
                            print("Fejezet:", fejezet)
                        fejezetek.append((fejezet, len(table_rows), fejezetszam))
                    continue
                
                # Skip empty rows
                if all([not col.strip() for col in cols]):
                    if verbose:
                        print("Empty skip:", cols)
                    continue
                
                # Extract name and sum columns
                if len(cols) < 6:
                    if verbose:
                        print("Short skip:", cols)
                    continue
                    
                name = cols[-6]
                if "összesen" in name.lower():
                    if verbose:
                        print("Total skip:", name)
                    continue
                
                sumrows = cols[-5:]
                
                # Validate sum rows
                if any([col.strip() and not col.strip().replace(',', '').replace('.', '').replace(' ', '').isdigit() for col in sumrows]):
                    if verbose:
                        print("Non-num skip:", cols)
                    continue
                
                if len(numcols) < 6 or len(sumrows) < 5:
                    if verbose:
                        print("Short skip:", cols)
                    continue
                
                # Build table row
                table_row = []
                for col in numcols:
                    table_row.append(int(col.strip().replace(',', '').replace('.', '').replace(' ', '') or '0'))
                table_row.append(name.strip())
                for col in sumrows:
                    table_row.append(float(col.strip().replace(',', '.').replace(' ', '') or '0'))
                
                if verbose:
                    print(table_row)
                table_rows.append(table_row)
        
        if verbose:
            print(f"Page {i+1} done")
    
    return table_rows, fejezetek


def extract_jogtar(url: str, verbose: bool = False):
    """
    Extract data from jogtar.hu (single page format)

    Args:
        url: Full URL to scrape
        verbose: Print debug information
        
    Returns:
        tuple: (table_rows, fejezetek)
    """
    table_rows = []
    fejezetek = []
    
    if verbose:
        print(f"Fetching: {url}")
    
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    for table in soup.select("table"):
        rows = table.select("tr")
        for row in rows:
            cols = [td.text for td in row.select("td")]
            numcols = cols[:6]
            
            # Check for chapter header
            if any([col.strip() and not col.strip().isdigit() for col in numcols]):
                if verbose:
                    print("Non-num skip:", cols)
                if len(cols) > 1 and "Kiadás" == cols[1].strip():
                    fejezet = cols[0].strip()
                    if '. ' not in fejezet:
                        if verbose:
                            print("Fejezet skip:", fejezet)
                        continue
                    fejezetszam = from_roman_numeral(fejezet.split('. ')[0])
                    if verbose:
                        print("Fejezet:", fejezet)
                    fejezetek.append((fejezet, len(table_rows), fejezetszam))
                continue
            
            # Skip empty rows
            if all([not col.strip() for col in cols]):
                if verbose:
                    print("Empty skip:", cols)
                continue
            
            # Extract name and sum columns
            if len(cols) < 6:
                if verbose:
                    print("Short skip:", cols)
                continue
                
            name = cols[-6]
            if "összesen" in name.lower():
                if verbose:
                    print("Total skip:", name)
                continue
            
            sumrows = cols[-5:]
            
            # Validate sum rows
            if any([col.strip() and not col.strip().replace(',', '').replace('.', '').replace(' ', '').isdigit() for col in sumrows]):
                if verbose:
                    print("Non-num skip:", cols)
                continue
            
            if len(numcols) < 6 or len(sumrows) < 5:
                if verbose:
                    print("Short skip:", cols)
                continue
            
            # Build table row
            table_row = []
            for col in numcols:
                table_row.append(int(col.strip().replace(',', '').replace('.', '').replace(' ', '') or '0'))
            table_row.append(name.strip())
            for col in sumrows:
                table_row.append(float(col.strip().replace(',', '.').replace(' ', '') or '0'))
            
            if verbose:
                print(table_row)
            table_rows.append(table_row)
    
    if verbose:
        print("Page done")
    
    return table_rows, fejezetek


def extract_jogkodex(url: str, verbose: bool = False):
    """
    Extract data from jogkodex.hu (single page, different format)
    
    Args:
        url: Full URL to scrape
        verbose: Print debug information
        
    Returns:
        tuple: (table_rows, fejezetek)
    """
    table_rows = []
    fejezetek = []
    
    if verbose:
        print(f"Fetching: {url}")
    
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    for table in soup.select("table"):
        rows = table.select("tr")
        for row in rows:
            cols = [td.text for td in row.select("td")]
            numcols = cols[:6]
            
            # Check for chapter header
            if any([col.strip() and not col.strip().isdigit() for col in numcols]):
                if verbose:
                    print("Non-num skip:", cols)
                if len(cols) > 1 and "Kiadás" == cols[1].strip():
                    fejezet = cols[0].strip()
                    if '. ' not in fejezet:
                        if verbose:
                            print("Fejezet skip:", fejezet)
                        continue
                    fejezetszam = from_roman_numeral(fejezet.split('. ')[0])
                    if verbose:
                        print("Fejezet:", fejezet)
                    fejezetek.append((fejezet, len(table_rows), fejezetszam))
                continue
            
            # Skip empty rows
            if all([not col.strip() for col in cols]):
                if verbose:
                    print("Empty skip:", cols)
                continue
            
            # Extract name and sum columns (different positions for jogkodex)
            if len(cols) < 6:
                if verbose:
                    print("Short skip:", cols)
                continue
            
            # Try different name column positions
            name = cols[-5]
            if not name.strip():
                name = cols[-6]
            if not name.strip():
                name = cols[-7]
            
            if "összesen" in name.lower():
                if verbose:
                    print("Total skip:", name)
                continue
            
            sumrows = cols[-4:]  # Only 4 sum columns for jogkodex
            
            # Validate sum rows
            if any([col.strip() and not col.strip().replace(',', '').replace('.', '').replace(' ', '').isdigit() for col in sumrows]):
                if verbose:
                    print("Non-num skip:", cols)
                continue
            
            if len(numcols) < 6 or len(sumrows) < 4:
                if verbose:
                    print("Short skip:", cols)
                continue
            
            # Build table row
            table_row = []
            for col in numcols:
                table_row.append(int(col.strip().replace(',', '').replace('.', '').replace(' ', '') or '0'))
            table_row.append(name.strip())
            for col in sumrows:
                table_row.append(float(col.strip().replace(',', '.').replace(' ', '') or '0'))
            
            if verbose:
                print(table_row)
            table_rows.append(table_row)
    
    if verbose:
        print("Page done")
    
    return table_rows, fejezetek


def write_table_rows_to_csv(table_rows, fejezetek, filename):
    """
    Write table_rows data to a CSV file with chapter information
    
    Args:
        table_rows: List of lists/tuples containing the data rows
        fejezetek: List of tuples (name, start_index, number) for chapters
        filename: Output CSV filename
    """
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        currentfejezet_name = None
        
        # Write data rows
        for n, row in enumerate(table_rows):
            currentfejezet = None
            for f in fejezetek:
                if n < f[1]:
                    currentfejezet = f[2]
                    if currentfejezet_name != f[0]:
                        currentfejezet_name = f[0]
                    break
            writer.writerow([currentfejezet] + row)
    
    print(f"Data successfully written to {filename}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract budget data from Hungarian legal documents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Variants:
  jogtar_lapozos - For jogtar.hu URLs (multi-page, 6 numcols, 5 sumrows)
  jogtar         - For jogtar.hu URLs (single page, 6 numcols, 5 sumrows)
  jogkodex       - For jogkodex.hu URLs (single page, 6 numcols, 4 sumrows)

Examples:
  python extract_budget_from_url.py --url "https://mkogy.jogtar.hu/jogszabaly?docid=A2500069.TV" --variant jogtar_lapozos --output 2025_budget.csv
  python extract_budget_from_url.py --url "https://net.jogtar.hu/jogszabaly?docid=a2300055.tv" --variant jogtar --output 2023_budget.csv
  python extract_budget_from_url.py --url "https://jogkodex.hu/jsz/kvtv_2021_2020_90_torveny_9676945" --variant jogkodex --output 2021_budget.csv
        """
    )
    
    parser.add_argument('--url', required=True, help='URL to scrape')
    parser.add_argument('--variant', required=True, choices=['jogtar_lapozos', 'jogtar', 'jogkodex'],
                        help='Extraction algorithm variant')
    parser.add_argument('--output', required=True, help='Output CSV filename')
    parser.add_argument('--pages', type=int, default=6,
                        help='Number of pages to scrape (only for jogtar_lapozos variant, default: 6)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print debug information')
    
    args = parser.parse_args()
    
    # Extract data based on variant
    if args.variant == 'jogtar_lapozos':
        table_rows, fejezetek = extract_jogtar_lapozos(args.url, args.pages, args.verbose)
    elif args.variant == 'jogtar':
        table_rows, fejezetek = extract_jogtar(args.url, args.verbose)
    elif args.variant == 'jogkodex':
        table_rows, fejezetek = extract_jogkodex(args.url, args.verbose)
    
    # Write to CSV
    write_table_rows_to_csv(table_rows, fejezetek, args.output)
    
    print(f"Extracted {len(table_rows)} rows with {len(fejezetek)} chapters")


if __name__ == '__main__':
    main()
