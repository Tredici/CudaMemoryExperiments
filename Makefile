
NVCC := nvcc
SRC := huge_op_classic.cu


all: huge_op_pageable huge_op_pinned huge_op_mapped huge_op_unified


run-all: all run_pageable run_pinned run_mapped run_unified


huge_op_pageable: $(SRC)
	nvcc $(SRC) -o $@ -g -DCHOICE=1

huge_op_pinned: $(SRC)
	nvcc $(SRC) -o $@ -g -DCHOICE=2

huge_op_mapped: $(SRC)
	nvcc $(SRC) -o $@ -g -DCHOICE=3

huge_op_unified: $(SRC)
	nvcc $(SRC) -o $@ -g -DCHOICE=4



run_pageable: huge_op_pageable
	./huge_op_pageable

run_pinned: huge_op_pinned
	./huge_op_pinned

run_mapped: huge_op_mapped
	./huge_op_mapped

run_unified: huge_op_unified
	./huge_op_unified


prof_pageable: huge_op_pageable
	nvprof ./huge_op_pageable

prof_pinned: huge_op_pinned
	nvprof ./huge_op_pinned

prof_mapped: huge_op_mapped
	nvprof ./huge_op_mapped

prof_unified: huge_op_unified
	nvprof ./huge_op_unified


prof-all: prof_pageable prof_pinned prof_mapped prof_unified

