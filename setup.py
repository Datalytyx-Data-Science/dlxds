from setuptools import setup, find_packages

with open('README.rst', 'r') as fh:
	long_description = fh.read()

setup(name = 'dlxds',
      version = '0.1',
      description = 'Datalytyx data science tools',
      url = 'https://github.com/Datalytyx-Data-Science/dlxds',
      author = 'Datalytyx data science team',
      author_email = '',
      license = 'MIT',
      packages = find_packages(),
      install_requires = ['numpy', 'pandas', 'sklearn', 'scipy', 'os', 'warnings'],
      long_description = long_description,
      zip_safe=False,
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"])