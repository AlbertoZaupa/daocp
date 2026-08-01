PYTHON ?= $(shell command -v python >/dev/null 2>&1 && echo python || echo python3)

export PYTHONPATH := $(CURDIR)/lib:$(PYTHONPATH)

.PHONY: all run clean

all:
	@$(PYTHON) -c "import pybind11" 2>/dev/null || \
		($(PYTHON) -m pip install pybind11 || \
		 $(PYTHON) -m pip install --user pybind11)
	@mkdir -p lib build
	$(PYTHON) interfaces/python/setup.py build_ext \
		--build-lib lib --build-temp build

run: all
	$(PYTHON) examples/python/di.py

clean:
	$(RM) build/*
	$(RM) lib/daocp*.so
