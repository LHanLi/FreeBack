from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))


setup(name='FreeBack',
      version='0.1.3',

      description='Package for backtest',

      # URL
      url="https://github.com/LHanLi/FreeBack",

      # Author
      author="LH.Li,zzq",
      author_email='lh98lee@zju.edu.cn',

      license='License :: OSI Approved :: MIT License',

      packages=find_packages(),
      )


# python3 setup.py install --u
