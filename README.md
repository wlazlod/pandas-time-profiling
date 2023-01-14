# pandas-time-profiling
Generates profile reports from a pandas `DataFrame` regarding time variable.

## Inspiration
Package is hardly inspired by the great [Pandas Profiling](https://github.com/pandas-profiling/pandas-profiling) package and share some of the code (mostly regarding report front-end).  
It produces similiar report but focuses on checking the distribution regarding time domain. It checkes the statistics of each variable in each time period of selected variable.

## Instalation

For using in local enviroment you can install the package directly from git:

```bash
pip install git+https://github.axa.com/daniel-wlazlo-external/pandas-time-profiling.git
```
To install from cloned repo:
```bash
python setup.py install
```
Virtual enviroment is recommended for using the package.

## Report Creation

Before creating a report, you have to ensure that, the column used for breaking the dataset is a date type, otherwise, you will get an error:

```python
ValueError: Column: <column_name> is not datetime type. Please parse the variable.
```

You can do it for example by doing:
```python
df["date"]=pd.to_datetime(df["date"])
```

As in [Pandas Profiling],(https://github.com/pandas-profiling/pandas-profiling) raport is created by using: 

```python
Report = pandas_time_profiling.ProfileTimeReport(df,
            time_variable = "date",
            time_unit = "M")
```

Report can be exported to html file using `.to_html()` method.
```python
Report.to_file("./test.html")
```

## Report content

Report consist of 2 parts: summary and statistics of variables. For each type of variables several statistics are calculated to describe distribution over time.

```python
df = pd.read_csv("https://raw.githubusercontent.com/j3k0/corona-ministryinfo-gov-lb-fetcher/909dc3342e0af34f46771cac6cb2434694b6f7a4/daily_data.csv")
df['date'] = pd.to_datetime(df['date'])
Report = pandas_time_profiling.ProfileTimeReport(df, "date", "W")
Report.to_file("./example.html")
```

![Report example](/docs/Report.png)



