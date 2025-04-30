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
