.PHONY: lib, pybind, clean, format, all

all: lib


lib:
	@mkdir -p build
	@cd build; cmake ..
	@cd build; $(MAKE)

format:
	python3 -m black .
	clang-format-12 -i src/*.cc src/*.cu

clean:
	rm -rf build python/needle/backend_ndarray/ndarray_backend*.so
