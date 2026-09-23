from pyspark.sql import functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.sql.window import Window
from scipy.stats import f
import seaborn as sns
import matplotlib.pyplot as plt

from .utils import round_off, get_spark, prepare_df, reset_table, append_row, column_pairs

RESULT_SCHEMA = StructType([
    StructField('Column_1', StringType(), nullable=False),
    StructField('Column_2', StringType(), nullable=False),
    StructField('Pearson_Correlation', DoubleType(), nullable=True),
    StructField('Spearman_Correlation', DoubleType(), nullable=True),
    StructField('Cramers_V', DoubleType(), nullable=True),
    StructField('Anova_F_Value', DoubleType(), nullable=True),
    StructField('Anova_P_Value', DoubleType(), nullable=True)
])


def _pearson(df, col1, col2):
    '''The correlation coefficient measures the strength of a linear relationship between two variables.
    It ranges from -1 (perfect negative correlation) to 1 (perfect positive correlation);
    0 means there is no linear relationship.'''
    return df.stat.corr(col1, col2)


def _spearman(df, col1, col2):
    '''Spearman's correlation is Pearson's correlation computed on the ranks of the values.'''
    ranked_df = (df.withColumn(f"{col1}_rank", F.rank().over(Window.orderBy(col1)))
                   .withColumn(f"{col2}_rank", F.rank().over(Window.orderBy(col2))))
    return ranked_df.stat.corr(f"{col1}_rank", f"{col2}_rank")


def _cramers_v(df, col1, col2):
    '''Cramer's V measures the strength of association between two nominal variables.
    It ranges from 0 (no association) to 1 (perfect association).'''
    filtered_df = df.filter(col(col1).isNotNull() & col(col2).isNotNull())
    counts = {(row[col1], row[col2]): row['count'] for row in filtered_df.groupBy(col1, col2).count().collect()}

    categories_col1 = {key[0] for key in counts}
    categories_col2 = {key[1] for key in counts}
    row_totals = {a: sum(counts.get((a, b), 0) for b in categories_col2) for a in categories_col1}
    col_totals = {b: sum(counts.get((a, b), 0) for a in categories_col1) for b in categories_col2}
    total_count = sum(counts.values())

    chi2 = 0.0
    for a in categories_col1:
        for b in categories_col2:
            expected = row_totals[a] * col_totals[b] / total_count
            chi2 += (counts.get((a, b), 0) - expected) ** 2 / expected

    min_dim = min(len(categories_col1), len(categories_col2)) - 1
    return (chi2 / (total_count * min_dim)) ** 0.5


def _anova(df, num_col, cat_col):
    '''ANOVA tests whether there are significant differences between the means of different groups.
    F-Value: a higher value indicates greater variance between groups compared to within groups.
    P-Value: a value < 0.05 typically indicates significant differences between group means,
    suggesting the categorical variable impacts the numerical variable.'''
    df = df.withColumn(num_col, col(num_col).cast('double'))

    summary_stats = df.groupBy(cat_col).agg(F.mean(num_col).alias('mean'), F.count(num_col).alias('count'))
    num_groups = summary_stats.count()
    overall_mean = df.select(F.mean(col(num_col)).alias('mean')).first()['mean']

    # Sum of squares between groups (SSB)
    ssb = summary_stats.withColumn('ssb', (col('mean') - overall_mean) ** 2 * col('count')).agg(F.sum('ssb')).first()[0]

    # Sum of squares within groups (SSW)
    ssw = summary_stats.withColumn('ssw', (col('count') - 1) * col('mean') ** 2).agg(F.sum('ssw')).first()[0]
    ssw -= num_groups * overall_mean ** 2

    df_b = num_groups - 1
    df_w = df.count() - num_groups

    f_val = (ssb / df_b) / (ssw / df_w)
    p_val = f.cdf(f_val, df_b, df_w)
    return f_val, p_val


def _plot_scatter(df, col1, col2):
    data = df.select(col1, col2).dropna().toPandas()
    plt.figure(figsize=(5, 3))
    sns.scatterplot(data=data, x=col1, y=col2)
    plt.title(f'Scatter Plot between {col1} and {col2}')
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.show()


def get_bivariate_analysis(df, table_name, numerical_columns, categorical_columns, id_columns=None, p_correlation_analysis=0, s_correlation_analysis=0, cramer_analysis=0, anova_analysis=0, print_graphs=0):
    """
    Perform bivariate analysis on the given DataFrame and save the results in a single table.

    Parameters:
    df (DataFrame): The input DataFrame for analysis.
    table_name (str): The base table name to save the results.
    numerical_columns (list): List of numerical columns.
    categorical_columns (list): List of categorical columns.
    id_columns (list): List of ID columns to drop.
    p_correlation_analysis (bool): Whether to perform Pearsons correlation analysis.
    s_correlation_analysis (bool): Whether to perform Spearmans correlation analysis.
    cramer_analysis (bool): Whether to perform Cramer's V analysis.
    anova_analysis (bool): Whether to perform ANOVA analysis.
    print_graphs (bool): Whether to print scatter plot graphs.
    """
    spark = get_spark()
    df = prepare_df(df, id_columns)
    reset_table(spark, table_name)

    # Numerical vs numerical - Pearson's and/or Spearman's correlation
    if p_correlation_analysis or s_correlation_analysis:
        for col1, col2 in column_pairs(numerical_columns):
            try:
                pearson = round_off(_pearson(df, col1, col2)) if p_correlation_analysis else None
                spearman = round_off(_spearman(df, col1, col2)) if s_correlation_analysis else None
                append_row(spark, table_name, (col1, col2, pearson, spearman, None, None, None), RESULT_SCHEMA)
            except Exception as e:
                print(f"Error calculating correlation for columns {col1} and {col2}: {e}")

    # Categorical vs categorical - Cramer's V
    if cramer_analysis:
        for col1, col2 in column_pairs(categorical_columns):
            try:
                cramer_v = round_off(_cramers_v(df, col1, col2))
                append_row(spark, table_name, (col1, col2, None, None, cramer_v, None, None), RESULT_SCHEMA)
            except Exception as e:
                print(f"Error calculating Cramer's V for columns {col1} and {col2}: {e}")

    # Numerical vs categorical - ANOVA
    if anova_analysis:
        for num_col in numerical_columns:
            for cat_col in categorical_columns:
                try:
                    f_val, p_val = _anova(df, num_col, cat_col)
                    append_row(spark, table_name, (num_col, cat_col, None, None, None, round_off(f_val), round_off(p_val)), RESULT_SCHEMA)
                except Exception as e:
                    print(f"Error calculating ANOVA for columns {num_col} and {cat_col}: {e}")

    if print_graphs:
        for col1, col2 in column_pairs(numerical_columns):
            try:
                _plot_scatter(df, col1, col2)
            except Exception as e:
                print(f"Error generating scatter plot for columns {col1} and {col2}: {e}")

    print(f"The results have been successfully saved to the table: {table_name}")
