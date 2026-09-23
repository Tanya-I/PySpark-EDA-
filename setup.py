from setuptools import find_packages, setup

setup(
    name='pyspark_eda',  # Updated name to match your package
    version='1.6.0',
    packages=find_packages(),
    description='A Python package for univariate ,bivariate and multivariate data analysis using PySpark',
    author='Tanya Irani', 
    author_email='tanyairani22@gmail.com',  
    keywords='data analysis pyspark univariate bivariate multivariate statistics',  # Add relevant keywords
    install_requires=[
        'pyspark>=3.0.0',
        'matplotlib>=3.0.0',
        'scipy>=1.4.0',
        'seaborn>=0.11.0'
    ],
    python_requires='>=3.6',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)
