ifeq ($(OS),Windows_NT)
	winos := 1
else
	linos := 1
endif

empty =
space = $(empty) $(empty)

comma := ,

ifdef winos
EXE = optixTriangle.exe
else
EXE = optixTriangle
endif

OBJ = \
	optixTriangle.o \
	shader_optixTriangle.o \

HDR = \
	sphere.h \
	thing.h \
	util.h \
	v.h \

CLS = \
	sphere.cxx \
	thing.cxx \
	util.cxx \
	v.cxx \

IMG = optixTriangle.png

SHD = \
	shader_optixTriangle.cu \

PTX = \
	shader_optixTriangle.ptx \

CXXFLAGS = -Wall -Wsign-compare -Wno-multichar -funroll-loops -fPIC -msse -msse2 -msse3 -mfpmath=sse -O3 -g3 -DNDEBUG
NVCCFLAGS = -std c++11 -ccbin /usr/bin/g++ -Xcompiler $(subst $(space),$(comma),$(CXXFLAGS)) -arch sm_75 -use_fast_math

INC = \
	-I/usr/local/cuda/include \
	-I/usr/local/optix/include \
	-I/usr/local/optix/SDK \
	-I/usr/local/optix/SDK/support \

.PHONY: all clean tidy
.SUFFIXES: .cxx .cu .png .ptx

.c.o:
	gcc -c $< -o $@

.cxx.o:
	nvcc -c $< -o $@ \
		$(NVCCFLAGS) \
		$(INC) \

.cu.ptx:
	nvcc -ptx $< -o $@ \
		$(NVCCFLAGS) \
		$(INC) \

.ptx.c:
	bin2c -c -p 0 -n $(basename $<) $< > $@

all: $(EXE) $(IMG)

$(EXE): $(OBJ)
	g++ -o $@ $^ \
		-L ~/optix-samples/lib -lsutil_7_sdk -lglad \
		-L /usr/local/cuda/lib64 -lcudart -lpthread -lrt -ldl \
		-lm \



clean:
	rm -f $(OBJ)

tidy: clean
	rm -f $(EXE) $(IMG)



$(IMG): $(EXE)
	./$< | magick ppm:- $@
ifdef winos
	cmd /c $@
endif
