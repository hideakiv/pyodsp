from setuptools import find_packages, setup

setup(
    name="pyodsp",
    version="0.2.0",
    description="Pyomo interface for Decomposition of Structured Programs",
    author="Hideaki Nakao",
    author_email="h.nakao1992blanca@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "Pyomo",
        "pytest",
        "highspy",
        "scipy",
    ],
    extras_require={
        # pyodsp.viz and pyodsp.model.sp.viz only; nothing else imports them
        "viz": ["matplotlib"],
        # docs/ only. mpi4py is mocked in conf.py rather than installed.
        "docs": [
            "sphinx>=7",
            "myst-parser>=2",
            "furo",
            "sphinx-copybutton",
        ],
    },
)
