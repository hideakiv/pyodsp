from setuptools import setup

setup(
    name="pyodsp",
    version="0.1",
    description="Pyomo interface for Decomposition of Structured Programs",
    author="Hideaki Nakao",
    author_email="1@gmail.com",
    packages=["pyodsp"],
    install_requires=[
        "numpy",
        "pandas",
        "Pyomo",
        "mpi4py",
        "pytest",
    ],
)
