PYTHON ?= $(shell command -v python >/dev/null 2>&1 && echo python || echo python3)

export PYTHONPATH := $(CURDIR)/lib:$(PYTHONPATH)

.PHONY: clean

BLASFEO = lib/libblasfeo.a

pywrapper: $(BLASFEO)
	@$(PYTHON) -c "import pybind11" 2>/dev/null || \
		($(PYTHON) -m pip install pybind11 || \
		 $(PYTHON) -m pip install --user pybind11)
	@mkdir -p lib build
	$(PYTHON) interfaces/python/setup.py build_ext \
		--build-lib lib --build-temp build

$(BLASFEO):
	@cd blasfeo && make static_library 
	@mv blasfeo/lib/libblasfeo.a lib/

clean:
	$(RM) -r build
	$(RM) lib/daocp*.so
	$(MAKE) -C blasfeo clean
	$(RM) lib/libblasfeo.a
