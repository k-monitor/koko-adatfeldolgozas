# PDF Name Matcher

A Python script to extract names from PDF documents by function codes and match them with a reference dataset using text similarity algorithms.

Based on experiments from `zarszam_v1.ipynb`.

## Features

- **PDF Text Extraction**: Extract text from PDF files using PyPDF2
- **Function-based Parsing**: Split text by function patterns (e.g., "F01", "F02")
- **Name Extraction**: Extract names using regex patterns
- **Smart Filtering**: Remove short names and cross-function duplicates
- **Similarity Matching**: Match extracted names with dataset using Jaro-Winkler similarity
- **Comprehensive Output**: Generate detailed reports and save results

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements_pdf_matcher.txt
```

Or install manually:
```bash
pip install PyPDF2>=3.0.0 pandas>=1.5.0 textdistance>=4.6.0
```

## Usage

### Command Line Usage

Basic usage:
```bash
python pdf_name_matcher.py input.pdf dataset.json
```

With custom parameters:
```bash
python pdf_name_matcher.py input.pdf dataset.json \
    --min-length 8 \
    --similarity 0.85 \
    --output-dir results \
    --function-pattern "F\d\d(\.\w)? " \
    --name-pattern "\d (\D{5,}) \d"
```

### Parameters

- `pdf_path`: Path to the input PDF file
- `dataset_path`: Path to the reference dataset (JSON lines format)
- `--min-length`: Minimum name length to keep (default: 10)
- `--similarity`: Similarity threshold for matching (default: 0.9)
- `--output-dir`: Output directory for results (default: output)
- `--function-pattern`: Regex pattern to identify function codes
- `--name-pattern`: Regex pattern to extract names

### Programmatic Usage

```python
from pdf_name_matcher import (
    extract_text_from_pdf,
    extract_names_by_function,
    filter_names,
    load_dataset,
    match_names_with_dataset
)

# Extract text from PDF
text = extract_text_from_pdf("input.pdf")

# Extract names by function
names_by_function = extract_names_by_function(text)

# Filter names
filtered_names, _ = filter_names(names_by_function, min_length=10)

# Load dataset
dataset_df = load_dataset("dataset.json")

# Match names
matches, function_matches, total_extracted, total_matches = match_names_with_dataset(
    filtered_names, dataset_df, similarity_threshold=0.9
)
```

## Input Formats

### PDF Format
The PDF should contain structured text with:
- Function codes (e.g., "F01", "F02.A")
- Names following a pattern that can be captured by regex

### Dataset Format
JSON lines format with `name` and `function` columns:
```json
{"name": "Kovács Péter", "function": "F01"}
{"name": "Nagy Anna", "function": "F02"}
{"name": "Szabó József", "function": "F01"}
```

## Output Files

The script generates several output files:

1. **names_by_function.json**: Original extracted names grouped by function
2. **filtered_names_by_function.json**: Names after filtering
3. **name_matches.csv**: Detailed matching results with similarity scores

### Sample Output Structure

```json
{
  "F01": ["KOVÁCS PÉTER", "NAGY ANNA"],
  "F02": ["SZABÓ JÓZSEF", "TÓTH MÁRIA"]
}
```

## Algorithm Overview

1. **Text Extraction**: Extract all text from PDF using PyPDF2
2. **Function Splitting**: Split text into blocks using function pattern regex
3. **Name Extraction**: Extract names from each block using name pattern regex
4. **Filtering**: 
   - Remove names shorter than minimum length
   - Remove names that appear in multiple functions (likely parsing errors)
5. **Matching**: Use Jaro-Winkler similarity to match with dataset names
6. **Results**: Generate comprehensive reports and statistics

## Examples

See `example_usage.py` for detailed examples:

```bash
# Run the example with sample data
python example_usage.py
```

## Customization

### Custom Patterns

You can customize the regex patterns for your specific document format:

```bash
# Example for different function pattern
python pdf_name_matcher.py input.pdf dataset.json \
    --function-pattern "Section \d+:" \
    --name-pattern "Name: ([A-Z\s]+)"
```

### Custom Similarity Threshold

Adjust the similarity threshold based on your data quality:
- Higher values (0.9-1.0): More strict matching, fewer false positives
- Lower values (0.7-0.9): More lenient matching, may include more matches

## Troubleshooting

### Common Issues

1. **No names extracted**: Check if your function and name patterns match the PDF structure
2. **Low matching rate**: Try lowering the similarity threshold
3. **Too many false matches**: Increase the minimum name length or similarity threshold

### Debug Mode

Add print statements to see intermediate results:

```python
# Check extracted text
text = extract_text_from_pdf("input.pdf")
print("First 500 characters:", text[:500])

# Check function matches
import re
function_pattern = r"F\d\d(\.\w)? "
matches = re.findall(function_pattern, text)
print("Function codes found:", matches)
```

## Performance

- **PDF Size**: Large PDFs (>100 pages) may take several minutes to process
- **Dataset Size**: Matching complexity is O(n*m) where n=extracted names, m=dataset names
- **Memory Usage**: Keep datasets under 100k records for optimal performance

## License

This script is based on research experiments and is provided as-is for educational and research purposes.
