from pyspark.sql import SparkSession


def round_off(num, dec=2):
    factor = 10 ** dec
    return int(num * factor) / factor


def get_spark():
    return SparkSession.builder.getOrCreate()


def prepare_df(df, id_columns=None):
    """Drop the ID columns and any columns that contain only null values."""
    if id_columns:
        df = df.drop(*id_columns)
    all_null_columns = [c for c in df.columns if df.filter(df[c].isNotNull()).count() == 0]
    return df.drop(*all_null_columns)


def reset_table(spark, table_name):
    spark.sql(f"DROP TABLE IF EXISTS {table_name}")


def append_row(spark, table_name, row, schema):
    """Append a single row (tuple matching the schema) to the results table."""
    new_row = spark.createDataFrame([row], schema=schema)
    new_row.write.option("mergeSchema", "true").saveAsTable(table_name, mode='append')


def column_pairs(columns):
    """Yield every unordered pair of distinct columns."""
    for i, col1 in enumerate(columns):
        for col2 in columns[i + 1:]:
            yield col1, col2
