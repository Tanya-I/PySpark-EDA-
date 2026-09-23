from pyspark.sql.functions import var_pop
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

from .utils import round_off, get_spark, reset_table, append_row

VIF_SCHEMA = StructType([
    StructField('Feature', StringType(), True),
    StructField('VIF', DoubleType(), True)
])


def get_multivariate_analysis(df, table_name, numerical_columns, id_columns=None):
    spark = get_spark()

    if id_columns:
        df = df.drop(*id_columns)

    reset_table(spark, table_name)

    for column in numerical_columns:
        try:
            # Zero variance means every value is the same: the column carries no
            # information for multivariate analysis and can cause computational issues.
            variance = df.select(var_pop(column)).first()[0]
            if variance == 0:
                print(f"Column '{column}' has zero variance and will be skipped.")
                continue

            other_columns = [c for c in numerical_columns if c != column]
            r_squared_sum = sum(df.stat.corr(column, other_col) ** 2 for other_col in other_columns)

            # VIF is infinite under perfect multicollinearity (the column is a
            # linear combination of the others), which invalidates the analysis.
            if r_squared_sum == 1.0:
                print(f"VIF for column '{column}' is infinity, indicating perfect multicollinearity, and will be skipped.")
                continue

            vif = 1.0 / (1.0 - r_squared_sum)
            append_row(spark, table_name, (column, round_off(vif)), VIF_SCHEMA)
        except Exception as e:
            print(f"Error processing VIF for column '{column}': {e}")

    print(f"The results have been successfully saved to the table: {table_name}")
