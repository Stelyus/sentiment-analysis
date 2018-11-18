# Google Cloud ML Engine setup


from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
'docopt',
'lxml',
'gensim',
'keras',
'beautifulsoup4',
'pandas',
'numpy',
'google-cloud-storage',
'setuptools',
'nltk',
'tensorflow']

setup(
  name='sentiment-analysis',
  version='0.1',
  author = 'Franck Thang',
  author_email = 'stelyus@outlook.com',
  include_package_data=True,
  install_requires=REQUIRED_PACKAGES,
  packages=find_packages()
  )
  
