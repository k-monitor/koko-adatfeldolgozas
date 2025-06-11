# költségvetés adatfeldolgozás

## előkövetelmények

- python virtuális környezet

## telepítés

```bash
pip install -r requirements.txt
```

## indoklás szövegek kinyerése

### pdf-ek előkészítése

#### régi formátum (2016-2019)

```
python preprocess_pdfs.py
```

**bemeneti adatok**

`adatok/koltsegvetesek.xlsx`

`indoklasok/nyers/2016.pdf`

`indoklasok/nyers/2017.pdf`

`indoklasok/nyers/2018.pdf`

`indoklasok/nyers/2019.pdf`


**kimeneti adatok**

feldarabolt pdf fájlok és azokat fejezetekkel összekötő `summary.json` fájlok

`indoklasok/feldolgozott/2016`

`indoklasok/feldolgozott/2017`

`indoklasok/feldolgozott/2018`

`indoklasok/feldolgozott/2019`

#### régi formátum külön pdf-ekből (2020-2023)

Ezeknél fel vannak darabolva külön pdf fájlokra az indoklásszövegek, de nem pont fejezetenként, például I. fejezetet még több kis részre bontják.

Manuálisan kell a több részre bontott fejezeteket egyesíteni és a fájlokhoz json fájlt létrehozni a megfelelő struktúrában.

Az automatizált feldolgozást az rontja el, hogy minden rész első oldala scannelt, amin a fejezet címe lenne.

#### új formátum két kötetre bontva (2024-2026)

Itt is nden rész első oldala scannelt, de a fejezetek 2 pdf-ben vannak ömlesztve.

Ehhez a `split_kotet.ipynb` Jupyter notebookot kell futtatni és az ott leírtakat követni.

Ez azon alapszik, hogy fejezetváltásoknál vannak üres oldalak, de sajnos nem csak akkor vannak, mert vannak fejezeteknél kisebb egységek is, amikor ez megesik. Az 1. fejezetet még magátol kikövetkezteti, de a többit manuálisan kell megadni.

### Indokláskinyerés futtatása

TODO

leírások összefűzése:
```bash
cat description_header.txt $(ls extracted_descriptions/2016*.csv | sort) | grep -v "id,indoklás szöveg" > descriptions_2016.csv
```

```bash
cat description_header.txt $(ls extracted_descriptions/2017*.csv | sort) | grep -v "id,indoklás szöveg" > descriptions_2017.csv
```

```bash
cat description_header.txt $(ls extracted_descriptions/2018*.csv | sort) | grep -v "id,indoklás szöveg" > descriptions_2018.csv
```
