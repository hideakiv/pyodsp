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
        "highspy",
        "scipy",
    ],
    extras_require={
        # pyodsp.viz and pyodsp.model.sp.viz only; nothing else imports them
        "viz": ["matplotlib"],
    },
)
