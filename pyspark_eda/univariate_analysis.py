import ast

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
import matplotlib.pyplot as plt

from .utils import round_off, get_spark, prepare_df, reset_table, append_row

SUMMARY_SCHEMA = StructType([
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


def _numerical_summary(df, column, total_count):
    non_null = df.filter(F.col(column).isNotNull())
    min_value = df.select(F.min(column)).first()[0]
    max_value = df.select(F.max(column)).first()[0]
    mean_value = non_null.select(F.round(F.mean(column), 3)).first()[0]
    mode_value = str(non_null.groupBy(column).count().orderBy(F.desc('count')).select(column).first()[0])
    null_percentage = round_off((total_count - non_null.count()) / total_count * 100)
    skewness_value = non_null.select(F.round(F.skewness(column), 3)).first()[0]
    kurtosis_value = non_null.select(F.round(F.kurtosis(column), 3)).first()[0]
    stddev_value = non_null.select(F.round(F.stddev(column), 3)).first()[0]
    q1, q2, q3 = (round_off(q) for q in non_null.approxQuantile(column, [0.25, 0.5, 0.75], 0.001))
    mean_plus_3std = round_off(mean_value + 3 * stddev_value)
    mean_minus_3std = round_off(mean_value - 3 * stddev_value)
    outlier_count = df.filter((F.col(column) > mean_plus_3std) | (F.col(column) < mean_minus_3std)).count()
    outlier_percentage = round_off(outlier_count / total_count * 100)

    return (column, total_count, min_value, max_value, mean_value, mode_value, f"{null_percentage}%",
            skewness_value, kurtosis_value, stddev_value, q1, q2, q3, mean_plus_3std, mean_minus_3std,
            f"{outlier_percentage}%", None)


def _categorical_summary(df, column, total_count):
    null_count = df.filter(F.col(column).isNull()).count()
    null_percentage = round_off(null_count / total_count * 100, 3)
    frequencies = df.groupBy(column).count().orderBy(F.desc('count')).collect()
    frequencies_dict = {str(row[column]): row['count'] for row in frequencies}

    return (column, total_count, None, None, None, None, f"{null_percentage}%",
            None, None, None, None, None, None, None, None, None, str(frequencies_dict))


def _plot_histogram(df, column):
    plt.figure(figsize=(6, 4))
    data = df.select(column).toPandas()[column].dropna()
    plt.hist(data, bins='auto', edgecolor='black')
    plt.title(f'{column} Histogram')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()


def _plot_frequency_distribution(column, frequencies_dict):
    plt.figure(figsize=(6, 4))
    plt.bar(frequencies_dict.keys(), frequencies_dict.values())
    plt.title(f"Frequency Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def get_univariate_analysis(df, table_name, numerical_columns, categorical_columns, id_list=None, print_graphs=0):
    spark = get_spark()
    df = prepare_df(df, id_list)
    reset_table(spark, table_name)
    total_count = df.count()

    for column in numerical_columns:
        try:
            append_row(spark, table_name, _numerical_summary(df, column, total_count), SUMMARY_SCHEMA)
        except Exception as e:
            print(f"Error processing numerical column {column}: {e}")

    for column in categorical_columns:
        try:
            append_row(spark, table_name, _categorical_summary(df, column, total_count), SUMMARY_SCHEMA)
        except Exception as e:
            print(f"Error processing categorical column {column}: {e}")

    print(f"The results have been successfully saved to the table: {table_name}")

    if not print_graphs:
        return

    for column in numerical_columns:
        try:
            _plot_histogram(df, column)
        except Exception as e:
            print(f"Error generating histogram for column {column}: {e}")

    summary_df = spark.read.table(table_name)
    for column in categorical_columns:
        try:
            frequencies_str = summary_df.filter(F.col("column") == column).select("frequency_distribution").first()[0]
            _plot_frequency_distribution(column, ast.literal_eval(frequencies_str))
        except Exception as e:
            print(f"Error generating frequency distribution for column {column}: {e}")
