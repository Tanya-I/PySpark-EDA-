from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
import matplotlib.pyplot as plt
from .utils import round_off

def get_univariate_analysis(df, table_name, numerical_columns, categorical_columns, id_list=None, print_graphs=0):
    spark = SparkSession.builder.getOrCreate()

    # Drop the ID column if provided
    if id_list:
        df = df.drop(*id_list)

    # Drop columns with all null values
    not_null_columns = [col for col in df.columns if df.filter(df[col].isNotNull()).count() == 0]
    df = df.drop(*not_null_columns)

    # Define schema for the final summary DataFrame
    summary_schema = StructType([
        StructField('column', StringType(), nullable=False),
        StructField('total_count', IntegerType(), nullable=False),
        StructField('min', DoubleType(), nullable=True),
        StructField('max', DoubleType(), nullable=True),
        StructField('mean', DoubleType(), nullable=True),
        StructField('mode', StringType(), nullable=True),
        StructField('null_percentage', StringType(), nullable=True),
        StructField('skewness', DoubleType(), nullable=True),
        StructField('kurtosis', DoubleType(), nullable=True),
        StructField('stddev', DoubleType(), nullable=True),
        StructField('q1', DoubleType(), nullable=True),
        StructField('q2', DoubleType(), nullable=True),
        StructField('q3', DoubleType(), nullable=True),
        StructField('mean_plus_3std', DoubleType(), nullable=True),
        StructField('mean_minus_3std', DoubleType(), nullable=True),
        StructField('outlier_percentage', StringType(), nullable=True),
        StructField('frequency_distribution', StringType(), nullable=True)
    ])

    # Drop the table if it exists
    spark.sql(f"DROP TABLE IF EXISTS {table_name}")

    # Numerical columns analysis
    for column in numerical_columns:
        try:
            total_count = df.count()
            min_value = df.select(F.min(column)).first()[0]
            max_value = df.select(F.max(column)).first()[0]
            mean_value = df.filter(~df[column].isNull()).select(F.round(F.mean(column), 3)).first()[0]
            mode_value = str(df.filter(~df[column].isNull()).groupBy(column).count().orderBy(F.desc('count')).select(column).first()[0])
            null_count = df.filter(df[column].isNull()).count()
            null_percentage = round_off((null_count / total_count) * 100)
            skewness_value = df.filter(~df[column].isNull()).select(F.round(F.skewness(column), 3)).first()[0]
            kurtosis_value = df.filter(~df[column].isNull()).select(F.round(F.kurtosis(column), 3)).first()[0]
            stddev_value = df.filter(~df[column].isNull()).select(F.round(F.stddev(column), 3)).first()[0]
            q1, q2, q3 = df.filter(~df[column].isNull()).approxQuantile(column, [0.25, 0.5, 0.75], 0.001)
            q1, q2, q3 = round_off(q1), round_off(q2), round_off(q3)
            mean_plus_3std = round_off(mean_value + 3 * stddev_value)
            mean_minus_3std = round_off(mean_value - 3 * stddev_value)
            outlier_count = df.filter((F.col(column) > mean_plus_3std) | (F.col(column) < mean_minus_3std)).count()
            outlier_percentage = round_off((outlier_count / total_count) * 100)

            # Create a new row and append to the DataFrame
            new_row = spark.createDataFrame([(column, total_count, min_value, max_value, mean_value, mode_value, f"{null_percentage}%", skewness_value, kurtosis_value, stddev_value, q1, q2, q3, mean_plus_3std, mean_minus_3std, f"{outlier_percentage}%", None)], schema=summary_schema)
            new_row.write.option("mergeSchema", "true").saveAsTable(table_name, mode='append')
        except Exception as e:
            print(f"Error processing numerical column {column}: {e}")

    # Categorical columns analysis
    for column in categorical_columns:
        try:
            total_count = df.count()
            null_count = df.filter(df[column].isNull()).count()
            null_percentage = round_off((null_count / total_count) * 100, 3)
            frequencies = df.groupBy(column).count().orderBy(F.desc('count')).select(column, F.col('count').alias('frequency')).collect()
            frequencies_dict = {str(row[column]): row['frequency'] for row in frequencies}

            # Create a new row and append to the DataFrame
            new_row = spark.createDataFrame([(column, total_count, None, None, None, None, f"{null_percentage}%", None, None, None, None, None, None, None, None, None, str(frequencies_dict))], schema=summary_schema)
            new_row.write.option("mergeSchema", "true").saveAsTable(table_name, mode='append')
        except Exception as e:
            print(f"Error processing categorical column {column}: {e}")

    print(f"The results have been successfully saved to the table: {table_name}")

    if print_graphs:
        # Show histograms for numerical columns
        for column in numerical_columns:
            try:
                plt.figure(figsize=(6, 4))
                data = df.select(column).toPandas()[column].dropna()
                plt.hist(data, bins='auto', edgecolor='black')  # 'auto' will use the Freedman-Diaconis rule
                plt.title(f'{column} Histogram')
                plt.xlabel(column)
                plt.ylabel('Frequency')
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"Error generating histogram for column {column}: {e}")

        # Plot frequency distribution for categorical data
        summary_df = spark.read.table(table_name)
        for column in categorical_columns:
            try:
                frequencies_dict_str = summary_df.filter(F.col("column") == column).select("frequency_distribution").first()[0]
                frequencies_dict = eval(frequencies_dict_str)
                plt.figure(figsize=(6, 4))
                plt.bar(frequencies_dict.keys(), frequencies_dict.values())
                plt.title(f"Frequency Distribution of {column}")
                plt.xlabel(column)
                plt.ylabel("Frequency")
                plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"Error generating frequency distribution for column {column}: {e}")
