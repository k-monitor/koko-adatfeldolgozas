# költségvetés adatfeldolgozás

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
