from collections import defaultdict
import PyPDF2
import re
import pandas as pd

excel_file = "adatok/koltsegvetesek.xlsx"
year = "2020"

df = pd.read_json(f"dataset/{year}.json", lines=True)


pdfFileObj = open("kvszervek_funkc-2021.pdf", "rb")

pdfReader = PyPDF2.PdfReader(pdfFileObj)

text = ""

for pageObj in pdfReader.pages:
    text += pageObj.extract_text() + "\n"

current_function = ""
names_by_function = defaultdict(list)

for line in text.split("\n"):
    if "Funkciók" in line:
        continue
    elif re.match(r"^F\d", line):
        current_function = line.split()[0]
        continue
    elif re.match(r"^\d+ ", line) and re.findall(r"[\d ,]+$", line):
        ending = re.findall(r"[\d ,]+$", line)[0]
        name = line[: -len(ending)].strip()
        name = re.sub(r"^\d+ ", "", name)
        names_by_function[current_function].append(name)
        continue

pdfFileObj.close()

names_to_function = dict()
for function, names in names_by_function.items():
    for name in names:
        names_to_function[name.strip()] = function

print("len functions", len(names_to_function.values()))

# df = pd.read_excel(excel_file, sheet_name=year)


def normalize_name(name):
    """Remove specified characters for better matching"""
    chars_to_remove = ',.-()"„"/'
    normalized = name.lower()
    for char in chars_to_remove:
        normalized = normalized.replace(char, "")
    return normalized.strip()


# Create normalized lookup dictionary
normalized_to_function = {}
for name, function in names_to_function.items():
    normalized_name = normalize_name(name)
    normalized_to_function[normalized_name] = function

# Track which names are used
used_names = set()


def map_function(name):
    normalized_name = normalize_name(name.strip())
    result = normalized_to_function.get(normalized_name, None)
    if result is not None:
        used_names.add(name.strip())
    return result


df["function"] = df["name"].map(map_function)
print("len df:", len(df))
print("len assigned functions:", df["function"].notna().sum())
print("isna len:", df["function"].isna().sum())
print(df.head())
print("isna %:", df["function"].isna().sum() / len(df) * 100)
# df.to_excel(f"adatok/koltsegvetesek_with_functions_{year}.xlsx", index=False)

print(df[df["function"].isna()]["name"].head())

# Find unused names
unused_normalized_names = set(normalized_to_function.keys()) - {
    normalize_name(name) for name in used_names
}
unused_original_names = {
    name
    for name in names_to_function.keys()
    if normalize_name(name) in unused_normalized_names
}

print(
    f"\nUnused names from names_to_function ({len(unused_original_names)} out of {len(names_to_function)}):"
)
for name in sorted(unused_original_names):
    print(f"- {name} -> {names_to_function[name]}")

print(f"\nUsage statistics:")
print(f"Total names in names_to_function: {len(names_to_function)}")
print(f"Names used in mapping: {len(used_names)}")
print(f"Names not used: {len(unused_original_names)}")
print(f"Usage percentage: {len(used_names)/len(names_to_function)*100:.1f}%")
