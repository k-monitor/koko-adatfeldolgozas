# költségvetés adatfeldolgozás

## Bevezetés

Probléma, amit meg szeretnénk oldani: számos költségvetési előirányzathoz nem tartozik funkciókód.

### Funkciók

**ÁLLAMI MŰKÖDÉSI FUNKCIÓK**

*   **F01: Általános közösségi szolgáltatások**
    *   F01.a: Törvényhozó és végrehajtó szervek
    *   F01.b: Pénzügyi és költségvetési tevékenységek és szolgáltatások
    *   F01.c: Külügyek
    *   F01.d: Alapkutatás
    *   F01.e: Műszaki fejlesztés
    *   F01.f: Egyéb általános közösségi szolgáltatások
*   **F02: Védelem**
*   **F03: Rendvédelem és közbiztonság**
    *   F03.a: Igazságszolgáltatás
    *   F03.b: Rend- és közbiztonság
    *   F03.c: Tűzvédelem
    *   F03.d: Büntetésvégrehajtási igazgatás és működtetés

**JÓLÉTI FUNKCIÓK**

*   **F04: Oktatási tevékenységek és szolgáltatások**
    *   F04.a: Iskolai előkészítés és alapfokú oktatás
    *   F04.b: Középfokú oktatás
    *   F04.c: Felsőfokú oktatás
    *   F04.d: Egyéb oktatás
*   **F05: Egészségügy**
    *   F05.a: Kórházi tevékenységek és szolgáltatások
    *   F05.b: Háziorvosi és gyermekorvosi szolgálat
    *   F05.c: Rendelői, orvosi, fogorvosi ellátás
    *   F05.d: Közegészségügyi tevékenységek és szolgáltatások
    *   F05.e: Egyéb egészségügy
*   **F06: Társadalombiztosítási és jóléti szolgáltatások**
    *   F06.a: Táppénz, anyasági vagy ideiglenes rokkantsági juttatások
    *   F06.b: Nyugellátások
    *   F06.c: Egyéb társadalombiztosítási ellátások
    *   F06.d: Munkanélküli ellátások
    *   F06.e: Családi pótlékok és gyermekeknek járó juttatások
    *   F06.f: Egyéb szociális támogatások
    *   F06.g: Szociális és jóléti intézményi szolgáltatások
*   **F07: Lakásügyek, települési és közösségi tevékenységek és szolgáltatások**
*   **F08: Szórakoztató, kulturális, vallási tevékenységek és szolgáltatások**
    *   F08.a: Sport és szabadidős tevékenységek és szolgáltatások
    *   F08.b: Kulturális tevékenységek és szolgáltatások
    *   F08.c: Műsorszórási és kiadói tevékenységek és szolgáltatások
    *   F08.d: Hitéleti tevékenységek
    *   F08.e: Párttevékenységek
    *   F08.f: Egyéb közösségi és kulturális tevékenységek

**GAZDASÁGI FUNKCIÓK**

*   **F09: Tüzelő- és üzemanyag, valamint energiaellátási feladatok**
*   **F10: Mező-, erdő-, hal- és vadgazdálkodás**
*   **F11: Bányászat és ipar**
*   **F12: Közlekedési és távközlési tevékenységek és szolgáltatások**
    *   F12.a: Közúti közlekedési tevékenységek
    *   F12.b: Vasúti közlekedésügyek és szolgáltatások
    *   F12.c: Távközlés
    *   F12.d: Egyéb közlekedés és szállítás
*   **F13: Egyéb gazdasági tevékenységek és szolgáltatások**
    *   F13.a: Többcélú fejlesztési témák tevékenységei és szolgáltatásai
    *   F13.b: Egyéb gazdasági tevékenységek és szolgáltatások
*   **F14: Környezetvédelem**

**ÁLLAMADÓSSÁG-KEZELÉS**

*   **F15: Államadósság-kezelés, államháztartás**

**FUNKCIÓBA NEM SOROLHATÓ TÉTELEK**

*   **F16: A főcsoportokba nem sorolható tételek**


## előkövetelmények

- Python 3 (3.13 verzión tesztelve)
- Költségvetési adatok táblázat

### Költségvetési adatok táblázat

Ez egy `.xlsx` kiterjesztésű excel fájl kell legyen (`adatok/koltsegvetesek.xlsx`). A munkalapok nevét az évszámok szerint kell elnevezni (pl.: `2016`, `2020`). A lapok tartalmazzák az előirányzatok listáját a szükséges adatokkal. A táblázat első sora a fejléc kell, hogy legyen, ami tartalmazza a szükséges oszlopneveket. A következő sortól pedig az adatoknak kell következni.

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

## függőségek telepítése

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
- `method_sureness`: a használt módszer melyik csoportba tartozik (`helyesnek elfogadott` vagy `átnézendő`)
- `needs_review`: át kell nézni embernek a tippet
- `{módszerek}`: módszerek tippjei (ez több oszlop)
- `sum`: az előirányzat kiadási összegei
- `true_function`: helyes funkciókód (ha van)
- `is_correct`: a true_function egyezik-e a predicted_function-nel, true/false érték

A futás közben létrejön egy `n2f.json` fájl is, amit korábbi előrányzat nevek és funkciókódok összekötésére használ a modell.

Ahhoz, hogy módosítsuk a kiválasztott évet, a TEST_YEAR változót kell átírni.

A tanításhoz/kereséshez használt évek listáját pedig a következő kódrész átírásával lehet elérni:
```python
    df_old_list = []
    for year in range(2016, 2020):
        df_old_list.append(datasets[year])
```

### Módszerek

#### Keresésen alapuló módszerek

`ahtt_exact_match`: Egy az egyben próbál ÁHT-T kód alapján előirányzatot keresni valamelyik korábbi évből és annak a funcióját adja meg.

`name_exact_match`: Egy az egyben (kis-nagy betűket figyelmen kívül hagyva) próbál név alapján előirányzatot keresni valamelyik korábbi évből és annak a funcióját adja meg.

`fid_exact_match`: A fejezetbeli hely (amit én fid-nek hívok, pl.: 14.20.5.7) szerint szó szerint próbál előirányzatot keresni valamelyik korábbi évből és annak a funcióját adja meg.

`name_fuzzy_match`: Megpróbál hasonló nevű előirányzatokat keresni korábbi évekből jaro_winkler algoritmussal, ha a score > 0.84 (threshold). Ezek közül a legvalószínűbb funkcióját adja meg.

`fid_fuzzy_match`: A fejezet/cím/alcím (ahol az előirányzat van, pl.: 14.20.5.7 esetén ez 14.20.5 alatti előirányzatok) leggyakoribb funkcióját adja meg.

`indoklas_fuzzy`: Ez sima tf-idf vektorizált indoklásszövegek közt néz coszinusz távolságot és így keresi meg korábbi évekből a hasonló előirányzatot (egy threshold mellett), annak a funkcióját adja meg.

`name_fuzzy_fallback`: Ugyanaz, mint name_fuzzy_match, csak sokkal kisebb (0.5) threshold-dal.

#### Gépi tanuláson alapuló szöveg klasszifikációs módszer

`ctfidf`: Egy c-TF-IDF modell, amit korábbi évek indoklás szöveg-funkció párosain tanítottam, és szöveg alapján mond egy funkciót.

### Módszerek egyesítése

A módszerek jelenleg egymás után sorban következnek, ha egy módszer nem tudja besorolni az előirányzatot, akkor továbbadja a következőnek.

A prioritási sorrend: `ahtt_exact_match > name_exact_match > fid_exact_match > name_fuzzy_match > indoklas_fuzzy > fid_fuzzy_match > name_fuzzy_fallback > ctfidf`

Nem minden módszer elég megbízható, ezért külÖn kezeljük azokat a tippeket, amikre jobb és külön, amikre kevésbé megbízható módszerrel jutottunk.

megbízható:
- ahtt_exact_match
- name_exact_match
- fid_exact_match
- name_fuzzy_match

nem elég megbízható:
- fid_fuzzy_match
- indoklas_fuzzy
- name_fuzzy_fallback
- ctfidf

Az alapján, hogy melyik csoport módszerét használtuk, megkülönböztetünk "helyesként számontartott" és "átnézendő" előirányzatokat.

### Kiértékelés

A módszerek kiértékelésekor két metrikára támaszkodhatunk: pontosság és lefedettség.

A pontosság alatt azt értjük, hogy az eltalált esetek száma osztva az összes esettel, amire a modell valamilyen választ adott.

Ami kicsit megtévesztő lehet, hogy itt a módszereknél csak azokat az eseteket számoltam bele, ahol az adott módszer tudott válaszolni valamit. És ennek a magyarázatára szolgál a lefedettség.

Tehát, ha például ÁHT-T párosításnál a lefedettség 99%, akkor az azt jelenti, hogy a minták 99%-ához sikerült ÁHT-T kódot találni korábbi évekből. Ha pedig a pontosság pl. 95%, akkor az pedig azt, hogy ennek a 99%-nak a 95%-át sikerül ÁHT-T kód szerint helyesen besorolni.

A megoldásnak olyan mérőszámai vannak, hogy:
- Mekkora azoknak az előirányzatoknak az aránya, amiket "helyesként számontartott"-nak jelölünk. Vagyis már nem kell hozzányúlni.
- A "helyesként számontartott"-nak jelölt előirányzatok mekkora arányban helyesek tényleg. Ha ez túl kicsi az a végleges munka minőségére mehet.
- Mekkora azoknak az előirányzatoknak az aránya, amiket "átnézendő"-nek jelöltünk. Vagyis át kell majd nézni. Ez értelemszerűen a tuti_coverage-el* együtt adja ki a 100%-ot.
- Az "átnézendő"-nek jelölt előirányzatok mekkora arányban helyesek. Ez a felkínált tipp pontossága, nyilván hasznos ha helyes, de nem létfeltétel, mert valaki úgyis át kell nézze.

A valóságban pedig ezek az értékek (egy ponton túl) csak egymás kárára javíthatók. Ha növelni akarom a modell pontosságát, és érzékenyebbre állítom a modellt, azzal csökkentem a "helyesként számontartott" előirányzatok arányát, így több előirányzatot kell manuálisan átnézni.

Tehát a cél az, hogy megtaláljuk azt az ideális arányt, ahol még elég pontosak a kiválasztott funkciók, de nincs túl sok manuális munka az ellenőrzéssel.

Példákkal, hogy ezeknek a mérőszámoknak miért van jelentőssége:
- Lehet például egy olyan képzeletbeli megoldás, amiben a mindenre azt a jelölést teszi a modell, hogy "átnézendő" ezzel a modell által átnézett előirányzatok (0 db) mind helyes lesz.
- Lehet az is, hogy a modell mindent megold magától, és nem kell manuálisan átnézni semmit, de a megoldások pontatlanok lesznek.
- Nyilván az ideális az lenne, ha egyszerre lenne tökéletes a megoldás és minden előirányzatot lefedne, de ez gyakorlatban nem megvalósítható. Úgyhogy be kell érnünk egy gyengébb megoldással azt viszont mi döntjük el, hogy melyik szempontból legyen jobb.

| year | 2017 | 2018 | 2019 | 2020 | 2021 |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **helyesként számontartott accuracy** | 0.9772 | 0.9918 | 0.9819 | 0 | 0 |
| helyesként számontartott coverage | 0.8677 | 0.9009 | 0.92 | 0.8619 | 0.7246 |
| **helyesként számontartott sum accuracy** | 0.989 | 0.9928 | 0.9973 | 0 | 0 |
| helyesként számontartott sum coverage | 0.9596 | 0.987 | 0.9796 | 0.9212 | 0.7431 |
| **átnézendő accuracy** | 0.7323 | 0.4681 | 0.6234 | 0 | 0 |
| átnézendő coverage | 1 | 1 | 1 | 1 | 1 |
| **átnézendő sum accuracy** | 0.6486 | 0.3978 | 0.5031 | 0 | 0 |
| átnézendő sum coverage | 1 | 1 | 1 | 1 | 1 |
|  |  |  |  |  |  |
| **ahtt\_exact\_match accuracy** | 0.9865 | 0.9965 | 0.984 | 0 | 0 |
| ahtt\_exact\_match coverage | 0.849 | 0.8946 | 0.9096 | 0 | 0 |
| **ahtt\_exact\_match sum accuracy** | 0.991 | 0.9977 | 0.9974 | 0 | 0 |
| ahtt\_exact\_match sum coverage | 0.9433 | 0.9777 | 0.979 | 0 | 0 |
| **name\_exact\_match accuracy** | 0.9744 | 0.9856 | 0.9731 | 0 | 0 |
| name\_exact\_match coverage | 0.8125 | 0.8757 | 0.8898 | 0.8191 | 0.5596 |
| **name\_exact\_match sum accuracy** | 0.9897 | 0.9968 | 0.9967 | 0 | 0 |
| name\_exact\_match sum coverage | 0.9007 | 0.9357 | 0.9709 | 0.8913 | 0.6499 |
| **fid\_exact\_match accuracy** | 0.9832 | 0.9917 | 0.9723 | 0 | 0 |
| fid\_exact\_match coverage | 0.7427 | 0.8851 | 0.8264 | 0.7704 | 0.5684 |
| **fid\_exact\_match sum accuracy** | 0.9946 | 0.9927 | 0.9838 | 0 | 0 |
| fid\_exact\_match sum coverage | 0.8655 | 0.9705 | 0.9138 | 0.8796 | 0.5496 |
| **name\_fuzzy\_match accuracy** | 0.9745 | 0.9856 | 0.9732 | 0 | 0 |
| name\_fuzzy\_match coverage | 0.8156 | 0.8767 | 0.8929 | 0.82 | 0.6439 |
| **name\_fuzzy\_match sum accuracy** | 0.9897 | 0.9968 | 0.9967 | 0 | 0 |
| name\_fuzzy\_match sum coverage | 0.901 | 0.939 | 0.973 | 0.8913 | 0.7112 |
| **fid\_fuzzy\_match accuracy** | 0.6632 | 0.6562 | 0.6783 | 0 | 0 |
| fid\_fuzzy\_match coverage | 0.9031 | 0.9779 | 0.921 | 0.8589 | 0.7544 |
| **fid\_fuzzy\_match sum accuracy** | 0.6375 | 0.6291 | 0.6392 | 0 | 0 |
| fid\_fuzzy\_match sum coverage | 0.9107 | 0.9921 | 0.9543 | 0.9214 | 0.6533 |
| **indoklas\_fuzzy accuracy** | 0.9555 | 0.9769 | 0.9716 | 0 | 0 |
| indoklas\_fuzzy coverage | 0.726 | 0.7313 | 0.6954 | 0 | 0 |
| **indoklas\_fuzzy sum accuracy** | 0.9598 | 0.9843 | 0.976 | 0 | 0 |
| indoklas\_fuzzy sum coverage | 0.6295 | 0.618 | 0.6969 | 0 | 0 |
| **name\_fuzzy\_fallback accuracy** | 0.8667 | 0.9041 | 0.9096 | 0 | 0 |
| name\_fuzzy\_fallback coverage | 1 | 1 | 1 | 1 | 1 |
| **name\_fuzzy\_fallback sum accuracy** | 0.9485 | 0.9794 | 0.9797 | 0 | 0 |
| name\_fuzzy\_fallback sum coverage | 1 | 1 | 1 | 1 | 1 |
| **ctfidf accuracy** | 0.8295 | 0.8472 | 0.8356 | 0 | 0 |
| ctfidf coverage | 0.9531 | 0.9104 | 0.9293 | 0 | 0 |
| **ctfidf sum accuracy** | 0.6905 | 0.9086 | 0.8798 | 0 | 0 |
| ctfidf sum coverage | 0.9183 | 0.8659 | 0.8861 | 0 | 0 |
