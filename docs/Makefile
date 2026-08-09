# Minimal makefile for Sphinx documentation.

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help clean html strict livehtml Makefile

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

# Treat warnings as errors — what CI should run.
strict:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" -W --keep-going -n $(SPHINXOPTS)

clean:
	rm -rf "$(BUILDDIR)"

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
