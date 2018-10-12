from setuptools import setup

setup(name = 'dlxtools',
      version = '0.1',
      description = 'Datalytyx data science tools',
      url = 'https://github.com/Datalytyx-Data-Science/dlxds',
      author = 'Datalytyx data science team',
      author_email = '',
      license = 'MIT',
      packages = ['dlxtools'],
      install_requires = ['numpy', 'pandas', 'sklearn', 'scipy'],
      long_description = open(README.md).read(),
      zip_safe=False)