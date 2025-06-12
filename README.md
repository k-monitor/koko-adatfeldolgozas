# költségvetés adatfeldolgozás

## előkövetelmények

- Python 3.13

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
