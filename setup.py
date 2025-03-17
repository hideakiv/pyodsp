from setuptools import setup, find_packages

setup(
    name="pyodsp",
    version="0.1",
    description="Pyomo interface for Decomposition of Structured Programs",
    author="Hideaki Nakao",
    author_email="h.nakao1992blanca@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "Pyomo",
        "pytest",
    ],
)
