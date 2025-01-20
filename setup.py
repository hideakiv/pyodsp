from setuptools import setup

setup(
    name="pyodec",
    version="0.1",
    description="Decomposition Algorithms for Block structured problems in Pyomo",
    author="Hideaki Nakao",
    author_email="1@gmail.com",
    packages=["pyodec"],
    install_requires=[
        "numpy",
        "Pyomo",
        "mpi4py",
    ],
)
