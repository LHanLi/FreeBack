from setuptools import setup, find_packages
from codecs import open

with open("README.md","r",encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name="FreeBack",
    # 版本号: 第几次模块增加，第几次函数增加，第几次函数功能修改
    # (每次高级别序号增加后,低级别序号归0)
    # alpha为调试版,beta为测试版,stable为稳定版 
    version="4.6.3",
    author="LH.Li,zzq",
    author_email="lh98lee@zju.edu.cn",  
    description='Package for backtest',
    long_description=long_description, 
    # 描述文件为md格式
    long_description_content_type="text/markdown",
    url="https://github.com/LHanLi/FreeBack",
    packages=find_packages(),
    install_requires = [
        #'pandas',
        #'scipy',
        #'statsmodels',
        #'seaborn',
        #'plottable',
        #'pyecharts',
        #'numpy_ext',
        #'xlsxwriter'
    ],
    classifiers=[
         # 该软件包仅与Python3兼容
        "Programming Language :: Python :: 3",
        # 根据GPL 3.0许可证开源
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        # 与操作系统无关
        "Operating System :: OS Independent",
    ],
)

# python3 setup.py install --u
