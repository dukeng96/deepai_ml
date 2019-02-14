from setuptools import setup

setup(
   name='deepai_ml',
   version='1.0',
   description='machine learning library developed by deepai',
   author='Duke Ng',
   author_email='knguyen4@gustavus.edu',
   packages=['deepai_ml'],
   install_requires=['sklearn_pandas', 'pandas', 'numpy', 'sklearn',
                     'statsmodels', 'mlxtend', 'xgboost', 'seaborn', 'matplotlib'], #external packages as dependencies
)