import duckdb
import pandas as pd

from .load_csv_data import load_csv_data


class InmemoryDB:
    def __init__(self, bucket_name: str) -> None:
        """Initialize inmemory db"""
        self.con = duckdb.connect(database=":memory:")
        self.bucket_name = bucket_name

    def create_inmemory_db(self, dict_csv_key: dict[str, str]) -> None:
        """Create duckdb tables from csv data"""
        tables = self.con.execute("SHOW TABLES").fetchdf()
        for table_name, key in dict_csv_key.items():
            if table_name not in tables.name.values:
                if table_name == "qualify":
                    table_name = f"{table_name}ing"
                df = load_csv_data(bucket_name=self.bucket_name, key=key)  # noqa
                query = f"CREATE TABLE {table_name} AS SELECT * FROM df"
                self.con.execute(query)

    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute query"""
        return self.con.execute(query).fetchdf()

    def close(self) -> None:
        """Clone connection"""
        self.con.close()
