SRCS=$(wildcard *.cu)
OUTPUTS=$(patsubst %.cu,%,bin/$(SRCS))


$(OUTPUTS): bin $(SRCS)

bin/%: %.cu
	nvcc $< -o $@

bin:
	mkdir -p bin




