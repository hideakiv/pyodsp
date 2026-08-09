# API reference

Generated from the docstrings in the source.

The package divides in three. {mod}`pyodsp.model` is the modelling front-end
you describe a stochastic program to. {mod}`pyodsp.dec` is the decomposition
machinery it builds on, usable on its own. {mod}`pyodsp.alg`,
{mod}`pyodsp.solver` and {mod}`pyodsp.viz` are the supporting pieces: the
bundle methods and risk measures, the Pyomo wrapper, and the plotting helpers.

```{toctree}
:maxdepth: 2

model
dec
support
```
