# formula1-season-explorer

## Environment
- Install packages needed by `pip install -r requirement.txt`.

## Static data
- Both of data is set to AWS S3 and read by boto3.
### Geo data
- [Data source](https://github.com/bacinger/f1-circuits/tree/master)

### Past data to train prediction model
- [Data source](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)

## Dynamic data
- Latest data which is not included in static data is retrieved from [Ergast](https://ergast.com/mrd/).
    - Season schedule
    - Latest grand prix result

## TODO
- Because [Ergast](https://ergast.com/mrd/) will shutdown at the end of 2024, new api or data source will be needed for 2025 season and after that, 
- Impact to the shuttingdown
    - Season schedule
    - Latest grand prix result data