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

def tfid_to_fid(tfid):
    fejezet = int(tfid[:2])
    cim = int(tfid[2:4])
    alcim = int(tfid[4:6])
    jogcim1 = int(tfid[6:8])
    jogcim2 = int(tfid[8:10])
    return f"{fejezet}.{cim}.{alcim}.{jogcim1}.{jogcim2}".replace(".0", "").replace("0.", "").replace("0", "").replace(".0", "").replace(".0", "")


print(tfid_to_fid("0101010000"))

for pageObj in pdfReader.pages:
    text += pageObj.extract_text() + "\n"

current_function = ""
names_by_function = defaultdict(list)
fid_by_function = defaultdict(list)

for line in text.split("\n"):
    if "Funkciók" in line:
        continue
    elif re.match(r"^F\d", line):
        current_function = line.split()[0]
        continue
    elif re.match(r"^\d+ ", line) and re.findall(r"[\d ,]+$", line):
        ending = re.findall(r"[\d ,]+$", line)[0]
        name = line[: -len(ending)].strip()
        tfid = re.findall(r"^\d+ ", line)[0].strip()
        name = re.sub(r"^\d+ ", "", name)
        names_by_function[current_function].append(name)
        fid_by_function[current_function].append(tfid_to_fid(tfid))
        continue

pdfFileObj.close()

names_to_function = dict()
for function, names in names_by_function.items():
    for name in names:
        names_to_function[name.strip()] = function

fid_to_function = dict()
for function, fids in fid_by_function.items():
    for fid in fids:
        fid_to_function[fid] = function

print("len functions", len(fid_to_function.values()))

# Track which names are used
used_names = set()

def map_function(fid):
    result = fid_to_function.get(fid, None)
    if result is not None:
        used_names.add(fid)
    return result


df["function"] = df["fid"].map(map_function)
print("len df:", len(df))
print("len assigned functions:", df["function"].notna().sum())
print("isna len:", df["function"].isna().sum())
print(df.head())
print("isna %:", df["function"].isna().sum() / len(df) * 100)
# df.to_excel(f"adatok/koltsegvetesek_with_functions_{year}.xlsx", index=False)

print(df[df["function"].isna()]["fid"].head())
