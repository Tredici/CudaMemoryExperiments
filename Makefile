
NVCC := /usr/local/cuda/bin/nvcc
NVPROF := /usr/local/cuda/bin/nvprof
SRC := huge_op_classic.cu


all: huge_op_pageable huge_op_pinned huge_op_mapped huge_op_unified


run-all: all run_pageable run_pinned run_mapped run_unified


huge_op_pageable: $(SRC)
	$(NVCC) $(SRC) -o $@ -g -DCHOICE=1

huge_op_pinned: $(SRC)
	$(NVCC) $(SRC) -o $@ -g -DCHOICE=2

huge_op_mapped: $(SRC)
	$(NVCC) $(SRC) -o $@ -g -DCHOICE=3

huge_op_unified: $(SRC)
	$(NVCC) $(SRC) -o $@ -g -DCHOICE=4


clean:
	rm -f huge_op_pageable huge_op_pinned huge_op_mapped huge_op_unified


run_pageable: huge_op_pageable
	./huge_op_pageable

run_pinned: huge_op_pinned
	./huge_op_pinned

run_mapped: huge_op_mapped
	./huge_op_mapped

run_unified: huge_op_unified
	./huge_op_unified


prof_pageable: huge_op_pageable
	$(NVPROF) ./huge_op_pageable

prof_pinned: huge_op_pinned
	$(NVPROF) ./huge_op_pinned

prof_mapped: huge_op_mapped
	$(NVPROF) ./huge_op_mapped

prof_unified: huge_op_unified
	$(NVPROF) ./huge_op_unified


prof-all: prof_pageable prof_pinned prof_mapped prof_unified


# Advanced profiling:
#   https://developer.nvidia.com/blog/cuda-pro-tip-nvprof-your-handy-universal-gpu-profiler/
adv-prof-all: adv_prof_pageable adv_prof_pinned adv_prof_mapped adv_prof_unified


adv_prof_pageable: huge_op_pageable
	time $(NVPROF) --analysis-metrics -o huge_op_pageable.nvprof ./huge_op_pageable

adv_prof_pinned: huge_op_pinned
	time $(NVPROF) --analysis-metrics -o huge_op_pinned.nvprof ./huge_op_pinned

adv_prof_mapped: huge_op_mapped
	time $(NVPROF) --analysis-metrics -o huge_op_mapped.nvprof ./huge_op_mapped

adv_prof_unified: huge_op_unified
	time $(NVPROF) --analysis-metrics -o huge_op_unified.nvprof ./huge_op_unified

