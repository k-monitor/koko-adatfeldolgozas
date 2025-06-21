# költségvetés adatfeldolgozás

## előkövetelmények

- Python 3.13
- Költségvetési adatok táblázat

### Költségvetési adatok táblázat

Ez egy `.xlsx` kiterjesztésű excel fájl kell legyen (`adatok/koltsegvetesek.xlsx`). A munkalapok nevét az évszámok szerint kell elnevezni (pl.: `2016`, `2020`). A lapok tartalmazzák az előirányzatok listáját a szükséges adatokkal. A táblázat első sora a egy fejléc kell legyen, ami tartalmazza a szükséges oszlopneveket. A következő sortól pedig az adatoknak kell következni.

Oszlopok (ezen kívül más oszlopok is megengedettek):
- `FEJEZET` *
- `CIM` *
- `ALCIM` *
- `JOGCIM1` *
- `JOGCIM2` *
- `ÁHT-T`
- `Funkció`
- `MEGNEVEZÉS` *
- Összegek v1
  - `Kiadás` **
  - `Bevétel` **
  - `Támogatás` **
- Összegek v2
  - `Működési kiadás` ***
  - `Működési bevétel` ***
  - `Felhalmozási kiadás` ***
  - `Felhalmozási bevétel` ***

* kötelező oszlop
** kötelező 2016 és korábbi költségvetések esetén
*** kötelező 2016 utáni költségvetések esetén

## python környezet felállítása (opcionális)

Elsőre:
```bash
python -m venv .venv
```

Aktiválás (minden futtatás előtt):
```bash
source .venv/bin/activate
```

## telepítés

```bash
pip install -r requirements.txt
```

## indoklás szövegek kinyerése

### pdf-ek előkészítése

Az indoklásszövegek évek során eltérő módon lettek publikálva, ennek a résznek a célja, hogy ezeket azonos formátumra hozzuk és fejezetekre bontsuk.

Az kész adatok a `indoklasok/feldolgozott/{év}` mappába fognak kerülni.

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

*Az automatizált feldolgozást az rontja el, hogy minden rész első oldala scannelt, amin a fejezet címe lenne.*

#### új formátum két kötetre bontva (2024-2026)

Előfeltételek:
- Python környezet a megfelelő függőségekkel
- excel fájl a költségvetésekkel
- **Jupyter lab vagy egyéb jupyter notebook futtató szoftver**
- **indoklasok/nyers/kotetek mappában {év}.pdf fájl, ami a két kötet egyesítése**

Itt is minden rész első oldala scannelt, de a fejezetek 2 pdf-ben vannak ömlesztve.

Ehhez a `split_kotet.ipynb` Jupyter notebookot kell futtatni és az ott leírtakat követni.

Ez azon alapszik, hogy fejezetváltásoknál vannak üres oldalak, de sajnos nem csak akkor vannak, mert vannak fejezeteknél kisebb egységek is, amikor ez megesik. Az 1. fejezetet még magátol kikövetkezteti, de a többit manuálisan kell megadni.

### Indokláskinyerés futtatása

Előfeltételek:
- Python környezet a megfelelő függőségekkel
- excel fájl a költségvetésekkel
- **`.env` fájl vagy egyéb úton átadni a `GEMINI_API_KEY` környezeti változót**
- **feldarabolt pdf fájlok és hozzájuk tartozó summary.json fájl az indoklasok/feldolgozott/{év} mappában**

Hogy a feldarabolt pdf-ekből kinyerd a leírásokat, futtatnod kell az `extract_description.py` scriptet. Ennek az elején tudod megadni a feldolgozandó éveket a `YEARS` tömb szerkesztésével. Javasolt kikommentezni azokat az éveket, amik nem kellenek.

```bash
python extract_description.py
```

## Adathalmaz előfeldolgozása

### Új év hozzáadása

A forráskód elején található egy `years` tömb, amit ki kell egészíteni egy új elemmel, pl.:
```python
    {
        "excel_sheet": "2021",
        "columns": {
            "name": "MEGNEVEZÉS",
            "spending": "Működési kiadás",
            "income": "Működési bevétel",
            "accumulated_spending": "Felhalmozási kiadás",
            "accumulated_income": "Felhalmozási bevétel",
        },
    }
```
Ha 2016 utáni adatról van szó, akkor a fenti példában csak az `excel_sheet`-et kell átírni, ha korábbi év, akkor valószínűleg ilyesmi kell legyen:
```python
    {
        "excel_sheet": "2016",
        "columns": {
            "name": "MEGNEVEZÉS",
            "spending": "Kiadás",
            "income": "Bevétel",
            "support": "Támogatás",
        },
    }
```

### Előfeltételek

- `adatok/koltsegvetesek.xlsx`
- `indoklasok/szovegek/{évszám}.csv`

### Futtatás

A `preprocess_dataset.py` fájlt kell futtatni:

```bash
python preprocess_dataset.py
```

Ez létrehozza a `dataset/` mappában az `{évszám}.json` (jsonlines file) és `{évszám}.csv` fájlokat. Ezek a fájlok egyben tartalmazzák az indoklásokat és a költségvetési adatok egységes formátumban.

## Funkció besoroló modell

Ez maga a lényeg, ez az, ami az eddig előfeldolgozott adatok alapján ellátja az előirányzatokat funkciókódokkal. 

### Előfeltételek

- `adatok/koltsegvetesek.xlsx`
- `dataset/{évszám}.json`

### Futtatás

A `model_iterative.py` fájlt kell futtatni:

```bash
python model_iterative.py
```

Ez létrehozza a `matches_df_{évszám}.xlsx` fájlt, ebben lesznek az oszlopok: 
- `section_name`: a fejezet neve
- `fid`: a fejezet beli hely `fejezet.cím.alcím...` formátumban
- `ÁHT-T`: ÁHT-T kód (ha van)
- `name`: előirányzat neve
- `indoklas`: indoklás szövege
- `predicted_function`: a modell által tippelt funkciókód
- `prediction_function`: a tippeléshez használt módszer
- `needs_review`: át kell nézni embernek a tippet
- `{módszerek}`: módszerek tippjei (ez több oszlop)
- `sum`: az előirányzat kiadási összegei
- `true_function`: helyes funkciókód (ha van)
- `is_correct`: a true_function egyezik-e a predicted_function-nel, true/false érték

A futás közben létrejön egy `n2f.json` fájl is, amit korábbi előrányzat nevek és funkciókódok összekötésére használ a modell.
