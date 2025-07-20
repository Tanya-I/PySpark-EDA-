# pyspark_eda

`pyspark_eda` is a Python library for performing exploratory data analysis (EDA) using PySpark. It offers functionalities for both univariate, bivariate analysis and multivariate analysis, handling missing values, outliers, and visualizing data distributions.

## Features

- **Univariate analysis:** Analyze numerical and categorical columns individually. Displays histogram and frequency distribution table if required.
- **Bivariate analysis:** Includes correlation, Cramer's V, and ANOVA. Displays scatter plot if required. 
- **Multivariate analysis:** Includes Variance Inflation Factor (VIF).
- **Automatic handling:** Deals with missing values and outliers seamlessly.
- **Visualization:** Provides graphical representation of data distributions and relationships.

## Installation
You can install `pyspark_eda` via pip:

```bash
pip install pyspark_eda
```
## Function
### Univariate Analysis 
### Parameters
- **df** (*DataFrame*): The input PySpark DataFrame.
- **table_name** (*str*): The base table name to save the results
- **numerical_columns** (*list*): The numerical columns of the table on which you want the analysis to be performed.
- **categorical_columns** (*list*): The categorical columns of the table on which you want the analysis to be performed.
- **id_list** (*list*, optional): List of columns to exclude from analysis.
- **print_graphs** (*int*, optional): Whether to print graphs (1 for yes, 0 for no),default value is 0.

### Description
Performs univariate analysis on the DataFrame and prints summary statistics and visualizations.
It returns a table with the following columns : column , total_count, min, max, mean , mode, null_percentage, skewness , kurtosis, stddev ( which is the standard deviation), q1,q2 q3 (quartiles), mean_plus_3std, mean_minus_3std, outlier_percentage and frequency_distribution.
You can display the table to view the results.

### Example Usage
## get_univariate_analysis
```python
from pyspark.sql import SparkSession
from pyspark_eda import get_univariate_analysis

# Initialize Spark session
spark = SparkSession.builder.appName('DataAnalysis').getOrCreate()

# Load your data into a PySpark DataFrame
df = spark.read.csv('your_data.csv', header=True, inferSchema=True)

# Identify numerical and categorical columns
numerical_columns = ['col1', 'col2', 'col3']
categorical_columns = ['col4', 'col5', 'col6']

# Perform univariate analysis
get_univariate_analysis(df, table_name="your_table_name", numerical_columns=numerical_columns, categorical_columns=categorical_columns, id_list=['id_column'], print_graphs=1)
```

## Function
### Bivariate Analysis
### Parameters
- **df** (*DataFrame*): The input PySpark DataFrame.
- **table_name** (*str*): The base table name to save the results
- **numerical_columns** (*list*): The numerical columns of the table on which you want the analysis to be performed.
- **categorical_columns** (*list*): The categorical columns of the table on which you want the analysis to be performed.
- **id_columns** (*list, optional*): List of columns to exclude from analysis.
- **p_correlation_analysis** (*int, optional*): Whether to perform Pearson's correlation analysis (1 for yes, 0 for no),default value is 0.
- **s_correlation_analysis** (*int, optional*): Whether to perform Spearman's correlation analysis (1 for yes, 0 for no),default value is 0.
- **cramer_analysis** (*int, optional*): Whether to perform Cramer's V analysis (1 for yes, 0 for no), default value is 0.
- **anova_analysis** (*int, optional*): Whether to perform ANOVA analysis (1 for yes, 0 for no),default value is 0.
- **print_graphs** (*int, optional*): Whether to print graphs (1 for yes, 0 for no),default value is 0.

### Description
Performs bivariate analysis on the DataFrame, including Pearsons Correlation,Spearmans Correlation, Cramer's V, and ANOVA.
It returns a table with the following columns: Column_1, Column_2, Pearson_Correlation,Spearman_Correlation, Cramers_V, Anova_F_Value,Anova_P_Value.
You can display the table to view the results.

### Example Usage 
### get_bivariate_analysis
```python
from pyspark.sql import SparkSession
from pyspark_eda import get_bivariate_analysis

# Initialize Spark session
spark = SparkSession.builder.appName('DataAnalysis').getOrCreate()

# Load your data into a PySpark DataFrame
df = spark.read.csv('your_data.csv', header=True, inferSchema=True)

# Identify numerical and categorical columns
numerical_columns = ['col1', 'col2', 'col3']
categorical_columns = ['col4', 'col5', 'col6']

# Perform bivariate analysis
get_bivariate_analysis(df, table_name="bivariate_analysis_results", numerical_columns=numerical_columns, categorical_columns=categorical_columns, id_columns=['id_column'], p_correlation_analysis=1,s_correlation_analysis=1, cramer_analysis=1, anova_analysis=1, print_graphs=0)
```

## Function
### Multivariate Analysis
### Parameters
- **df** (*DataFrame*): The input PySpark DataFrame.
- **table_name** (*str*): The base table name to save the results
- **numerical_columns** (*list*): The numerical columns of the table on which you want the analysis to be performed.
- **id_columns** (*list, optional*): List of columns to exclude from analysis.

### Description
Performs multivariate analysis on the DataFrame, which gives the Variance Inflation Factor (VIF) for each numerical column.
It returns a table with the following columns: Feature, VIF. You can display the table to view the results. 

### Example Usage 
### get_multivariate_analysis
```python
from pyspark.sql import SparkSession
from pyspark_eda import get_bivariate_analysis

# Initialize Spark session
spark = SparkSession.builder.appName('DataAnalysis').getOrCreate()

# Load your data into a PySpark DataFrame
df = spark.read.csv('your_data.csv', header=True, inferSchema=True)

# Identify numerical columns
numerical_columns = ['col1', 'col2', 'col3']

# Perform bivariate analysis
get_multivariate_analysis(df, table_name="multivariate_analysis_results", numerical_columns=numerical_columns, id_columns=['id_column'])
```

## Contact
- **Author:** Tanya Irani
- **Email:** tanyairani22@gmail.com