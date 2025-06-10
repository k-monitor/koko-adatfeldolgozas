# költségvetés adatfeldolgozás

## előkövetelmények

- python virtuális környezet

## telepítés

```bash
pip install -r requirements.txt
```

## indoklás szövegek kinyerése

### pdf-ek előkészítése

#### régi formátum

**bemeneti adatok**

`adatok/koltsegvetesek.xlsx`

`indoklasok/nyers/2016.pdf`

`indoklasok/nyers/2017.pdf`

`indoklasok/nyers/2018.pdf`

**kimeneti adatok**

`indoklasok/feldolgozott/2016`

`indoklasok/feldolgozott/2017`

`indoklasok/feldolgozott/2018`



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


```bash
cat description_header.txt $(ls extracted_descriptions/2019*.csv | sort) | grep -v "id,indoklás szöveg" > descriptions_2019.csv
```

```bash
cat description_header.txt $(ls extracted_descriptions/2020*.csv | sort) | grep -v "id,indoklás szöveg" > descriptions_2020.csv
```

cat description_header.txt $(ls extracted_descriptions/2021*.csv | sort) | grep -v "id,indoklás szöveg" > descriptions_2021.csv
