IMAGENAME = dcv-bench

BENCH_SCRIPT = /work/examples/bench_async_search.py

build:
	docker build -t $(IMAGENAME) -f examples/bench-search.dockerfile .

bash: build
	docker run -it --cpus=$(CPU_COUNT) --workdir /work/ -v $(shell pwd)/examples/output_data:/data $(IMAGENAME) /bin/bash

bench: build
	docker run -it --cpus=$(CPU_COUNT) --workdir /work/ -v $(shell pwd)/examples/output_data:/data $(IMAGENAME) python $(BENCH_SCRIPT) $(ARGS)
