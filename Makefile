PYTHON ?= $(shell command -v python >/dev/null 2>&1 && echo python || echo python3)

export PYTHONPATH := $(CURDIR)/lib:$(PYTHONPATH)

# Machine-local overrides can be placed in config.mk. 
-include config.mk
BLASFEO_TARGET ?= GENERIC

.PHONY: clean

BLASFEO := lib/libblasfeo.a

pywrapper: $(BLASFEO)
	@$(PYTHON) -c "import pybind11" 2>/dev/null || \
		($(PYTHON) -m pip install pybind11 || \
		 $(PYTHON) -m pip install --user pybind11)
	@mkdir -p lib build
	$(PYTHON) interfaces/python/setup.py build_ext \
		--build-lib lib --build-temp build

$(BLASFEO): 
	@mkdir -p $(@D)
	$(MAKE) -C blasfeo TARGET="$(BLASFEO_TARGET)" static_library
	@cp blasfeo/lib/libblasfeo.a $@

clean:
	$(RM) -r build
	$(RM) lib/daocp*.so
	$(RM) lib/libblasfeo.a
	$(MAKE) -C blasfeo clean
