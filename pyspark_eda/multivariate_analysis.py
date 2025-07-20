from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.sql import SparkSession
from pyspark.sql.functions import var_pop
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

from .utils import round_off

def get_multivariate_analysis(df, table_name, numerical_columns,id_columns=None):
    spark = SparkSession.builder.getOrCreate()

    # Drop the ID column if provided
    if id_columns:
        df = df.drop(*id_columns)

    vif_schema = StructType([
        StructField('Feature', StringType(), True),
        StructField('VIF', DoubleType(), True)
    ])
    # Drop the table if it exists
    spark.sql(f"DROP TABLE IF EXISTS {table_name}")

    for column in numerical_columns:
            try:
                # Check for zero variance columns using Spark
                variance = df.select(var_pop(column)).collect()[0][0]
                if variance == 0:
                    print(f"Column '{column}' has zero variance and will be skipped. ")
                    #Explanation: Zero variance means that all values in the column are the same. 
                    #Such columns do not provide any useful information for multivariate analysis and can cause computational issues.
                    continue

                # Calculate R-squared sum for the current column using PySpark
                other_columns = [c for c in numerical_columns if c != column]
                r_squared_sum = sum([df.stat.corr(column, other_col)**2 for other_col in other_columns])

                # Calculate VIF
                if r_squared_sum == 1.0:
                    vif = float('inf')
                else:
                    vif = 1.0 / (1.0 - r_squared_sum)

                if vif == float('inf'):
                    print(f"VIF for column '{column}' is infinity, indicating perfect multicollinearity, and will be skipped. ")
                    #Explanation: Perfect multicollinearity means that the column is a perfect linear combination of other columns. 
                    # This makes the VIF infinite and invalidates the analysis.
                    continue

                new_row = spark.createDataFrame([(column, round_off(vif))], schema=vif_schema)
                new_row.write.option("mergeSchema", "true").saveAsTable(table_name, mode='append')
            except Exception as e:
                print(f"Error processing VIF for column '{column}': {e}")

    print(f"The results have been successfully saved to the table: {table_name}")
