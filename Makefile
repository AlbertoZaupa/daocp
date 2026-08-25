PYTHON ?= $(shell command -v python >/dev/null 2>&1 && echo python || echo python3)
CC ?= cc
AR ?= ar
CFLAGS ?= -O3
CPPFLAGS += -Iinclude -Iblasfeo/include

ifeq ($(shell uname -s),Darwin)
export ARCHFLAGS ?= -arch $(shell uname -m)
ifeq ($(shell uname -m),arm64)
export MACOSX_DEPLOYMENT_TARGET ?= 11.0
endif
endif

export PYTHONPATH := $(CURDIR)/lib:$(PYTHONPATH)

# Machine-local overrides can be placed in config.mk. 
-include config.mk
BLASFEO_TARGET ?= GENERIC

.PHONY: all pywrapper run clean

BLASFEO := lib/libblasfeo.a
DAOCP := lib/libdaocp.a
DAOCP_SOURCES := $(wildcard src/*.c)
DAOCP_OBJECTS := $(patsubst src/%.c,build/src/%.o,$(DAOCP_SOURCES))

all: pywrapper

pywrapper: $(DAOCP)
	@$(PYTHON) -c "import pybind11" 2>/dev/null || \
		($(PYTHON) -m pip install pybind11 || \
		 $(PYTHON) -m pip install --user pybind11)
	@mkdir -p lib build
	$(PYTHON) interfaces/python/setup.py build_ext \
		--build-lib lib --build-temp build

run: pywrapper
	$(PYTHON) examples/python/di.py

$(DAOCP): $(BLASFEO) $(DAOCP_OBJECTS)
	@mkdir -p $(@D)
	$(AR) rcs $@ $(DAOCP_OBJECTS)

build/src/%.o: src/%.c $(BLASFEO) include/daocp.h include/internal.h include/defs.h
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -fPIC -c $< -o $@

$(BLASFEO): 
	@mkdir -p $(@D)
	$(MAKE) -C blasfeo TARGET="$(BLASFEO_TARGET)" static_library
	@cp blasfeo/lib/libblasfeo.a $@

clean:
	$(RM) -r build
	$(RM) lib/daocp*.so
	$(RM) lib/libdaocp.a lib/libblasfeo.a
	$(MAKE) -C blasfeo clean
