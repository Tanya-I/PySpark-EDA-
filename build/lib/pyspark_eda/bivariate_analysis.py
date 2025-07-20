from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.sql.functions import col
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql.window import Window
from scipy.stats import f
from .utils import round_off

def get_bivariate_analysis(df, table_name, numerical_columns, categorical_columns, id_columns=None, p_correlation_analysis=0,s_correlation_analysis=0, cramer_analysis=0, anova_analysis=0, print_graphs=0):
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

    spark = SparkSession.builder.getOrCreate()

    # Drop the ID column if provided
    if id_columns:
        df = df.drop(*id_columns)

    # Drop columns with all null values
    not_null_columns = [col for col in df.columns if df.filter(df[col].isNotNull()).count() == 0]
    df = df.drop(*not_null_columns)

    # Schema definition for result DataFrame
    result_schema = StructType([
        StructField('Column_1', StringType(), nullable=False),
        StructField('Column_2', StringType(), nullable=False),
        StructField('Pearson_Correlation', DoubleType(), nullable=True),
        StructField('Spearman_Correlation', DoubleType(), nullable=True),
        StructField('Cramers_V', DoubleType(), nullable=True),
        StructField('Anova_F_Value', DoubleType(), nullable=True),
        StructField('Anova_P_Value', DoubleType(), nullable=True)
    ])

    # Drop the table if it exists
    spark.sql(f"DROP TABLE IF EXISTS {table_name}")

    # Numerical vs numerical analysis -Pearsons correlation coefficient
    if p_correlation_analysis and not s_correlation_analysis and numerical_columns:
      for i in range(len(numerical_columns)):
            for j in range(i + 1, len(numerical_columns)):
                col1 = numerical_columns[i]
                col2 = numerical_columns[j]
                try:
                    corr = df.stat.corr(col1, col2)
                    new_row = spark.createDataFrame([(col1, col2, round_off(corr),None, None, None, None)], schema=result_schema)
                    new_row.write.option("mergeSchema", "true").saveAsTable(table_name, mode='append')
                except Exception as e:
                    print(f"Error calculating correlation for columns {col1} and {col2}: {e}")

    # Numerical vs numerical analysis -Spearmans correlation coefficient
    if s_correlation_analysis and not p_correlation_analysis and numerical_columns:
      for i in range(len(numerical_columns)):
        for j in range(i + 1, len(numerical_columns)):
            col1 = numerical_columns[i]
            col2 = numerical_columns[j]
            try:
                # Rank the columns
                window_spec1 = Window.orderBy(col1)
                ranked_df = df.withColumn(f"{col1}_rank", F.rank().over(window_spec1))
                window_spec2 = Window.orderBy(col2)
                ranked_df = ranked_df.withColumn(f"{col2}_rank", F.rank().over(window_spec2))

                # Calculate Pearson correlation on the ranks
                spearman_corr = ranked_df.stat.corr(f"{col1}_rank", f"{col2}_rank")
                new_row = spark.createDataFrame([(col1, col2, None, round_off(spearman_corr), None, None, None)], schema=result_schema)
                new_row.write.option("mergeSchema", "true").saveAsTable(table_name, mode='append')
            except Exception as e:
                print(f"Error calculating Spearman correlation for columns {col1} and {col2}: {e}")

    # Numerical vs numerical analysis -Spearmans and Pearsons correlation coefficient
    if p_correlation_analysis and  s_correlation_analysis and numerical_columns:
      for i in range(len(numerical_columns)):
        for j in range(i + 1, len(numerical_columns)):
            col1 = numerical_columns[i]
            col2 = numerical_columns[j]
            try:
                # Pearsons 
                corr = df.stat.corr(col1, col2)

                # Rank the columns
                window_spec1 = Window.orderBy(col1)
                ranked_df = df.withColumn(f"{col1}_rank", F.rank().over(window_spec1))
                window_spec2 = Window.orderBy(col2)
                ranked_df = ranked_df.withColumn(f"{col2}_rank", F.rank().over(window_spec2))

                # Calculate Pearson correlation on the ranks- so we get spearmans
                spearman_corr = ranked_df.stat.corr(f"{col1}_rank", f"{col2}_rank")
                new_row = spark.createDataFrame([(col1, col2, round_off(corr), round_off(spearman_corr), None, None, None)], schema=result_schema)
                new_row.write.option("mergeSchema", "true").saveAsTable(table_name, mode='append')
            except Exception as e:
                print(f"Error calculating Spearman correlation for columns {col1} and {col2}: {e}")

    # Categorical vs categorical analysis - Cramer's V
    if cramer_analysis and categorical_columns and len(categorical_columns) >= 2:
      for i in range(len(categorical_columns)):
        for j in range(i + 1, len(categorical_columns)):
            col1 = categorical_columns[i]
            col2 = categorical_columns[j]
            try:
                # Filter out rows with null values in either column
                filtered_df = df.filter(col(col1).isNotNull() & col(col2).isNotNull())

                # Create contingency table
                contingency_table = filtered_df.groupBy(col1, col2).count()

                # Create dictionaries for category to index mapping
                categories_col1 = {row[col1]: idx for idx, row in enumerate(filtered_df.select(col1).distinct().collect())}
                categories_col2 = {row[col2]: idx for idx, row in enumerate(filtered_df.select(col2).distinct().collect())}

                # Initialize contingency matrix
                contingency_matrix = [[0 for _ in range(len(categories_col2))] for _ in range(len(categories_col1))]

                # Populate contingency matrix
                for row in contingency_table.collect():
                    index1 = categories_col1[row[col1]]
                    index2 = categories_col2[row[col2]]
                    contingency_matrix[index1][index2] = row["count"]

                # Calculate chi-squared statistic
                chi2 = sum(
                    (contingency_matrix[i][j] - (sum(contingency_matrix[i]) * sum([contingency_matrix[k][j] for k in range(len(categories_col1))]) / sum([sum(contingency_matrix[m]) for m in range(len(categories_col1))])))**2 /
                    (sum(contingency_matrix[i]) * sum([contingency_matrix[k][j] for k in range(len(categories_col1))]) / sum([sum(contingency_matrix[m]) for m in range(len(categories_col1))]))
                    for i in range(len(categories_col1)) for j in range(len(categories_col2))
                )

                # Calculate total count
                total_count = sum([sum(row) for row in contingency_matrix])

                # Calculate Cramer's V
                min_dim = min(len(categories_col1), len(categories_col2)) - 1
                cramer_v = (chi2 / (total_count * min_dim))**0.5

                new_row = spark.createDataFrame([(col1, col2, None, round_off(cramer_v), None, None)], schema=result_schema)
                new_row.write.option("mergeSchema", "true").saveAsTable(table_name, mode='append')
            except Exception as e:
                print(f"Error calculating Cramer's V for columns {col1} and {col2}: {e}")

    # Numerical vs categorical analysis - ANOVA
    if anova_analysis and numerical_columns and categorical_columns:
      for num_col in numerical_columns:
        for cat_col in categorical_columns:
            try:
                # Ensure numerical column is cast to DoubleType
                df = df.withColumn(num_col, col(num_col).cast('double'))

                # Convert integer columns to double if they are numerical
                if cat_col not in categorical_columns and df.dtypes[cat_col] == 'int':
                    df = df.withColumn(cat_col, col(cat_col).cast('double'))

                # Group by categorical column and compute summary statistics
                summary_stats = df.groupBy(cat_col).agg(
                    F.mean(num_col).alias('mean'),
                    F.count(num_col).alias('count')
                )

                # Collect the summary statistics into a list
                summary_list = summary_stats.collect()

                # Overall mean for the numerical column
                overall_mean = df.select(F.mean(col(num_col)).alias('mean')).first()['mean']

                # Sum of squares between groups (SSB)
                ssb = summary_stats.withColumn('ssb', (col('mean') - overall_mean) ** 2 * col('count')).agg(F.sum('ssb')).first()['sum(ssb)']

                # Sum of squares within groups (SSW)
                ssw = summary_stats.withColumn('ssw', (col('count') - 1) * col('mean') ** 2).agg(F.sum('ssw')).first()['sum(ssw)']
                ssw -= len(summary_list) * overall_mean ** 2

                # Degrees of freedom
                df_b = len(summary_list) - 1
                df_w = df.count() - len(summary_list)

                # F-value
                f_val = (ssb / df_b) / (ssw / df_w)

                # P-value
                p_val = f.cdf(f_val, df_b, df_w)

                # Append ANOVA results
                new_row = spark.createDataFrame([(num_col, cat_col, None, None, round_off(f_val), round_off(p_val))], schema=result_schema)
                new_row.write.option("mergeSchema", "true").saveAsTable(table_name, mode='append')
            except Exception as e:
                print(f"Error calculating ANOVA for columns {num_col} and {cat_col}: {e}")

    # Shows scatter plot if the user wants
    if print_graphs:
        scatter_pairs = [(col1, col2) for i, col1 in enumerate(numerical_columns) for col2 in numerical_columns[i+1:]]
        def plot_scatter(pair):
            col1, col2 = pair
            try:
                data = df.select(col1, col2).dropna().toPandas()
                plt.figure(figsize=(5, 3))
                sns.scatterplot(data=data, x=col1, y=col2)
                plt.title(f'Scatter Plot between {col1} and {col2}')
                plt.xlabel(col1)
                plt.ylabel(col2)
                plt.show()
            except Exception as e:
                print(f"Error generating scatter plot for columns {col1} and {col2}: {e}")
        for pair in scatter_pairs:
            plot_scatter(pair)

    print(f"The results have been successfully saved to the table: {table_name}")