ifeq ($(OS),Windows_NT)
	winos := 1
else
	linos := 1
endif

empty =
space = $(empty) $(empty)

comma := ,

ifdef winos
EXE = rtwo.exe
else
EXE = rtwo
endif

OBJ = \
	rtwo.o \
	camera.o \
	optics.o \
	sphere.o \

IMG = rtwo.png

SHD = \
	camera.cu \
	optics.cu \

PTX = \
	camera.ptx \
	optics.ptx \

OPTIX_CACHE_PATH = ./OptixCache_$$USER

ifdef winos
# prerequisites:
# install VS2017 (Community)
# install Cygwin (gcc, g++, make, ...)
# unpack CUDA Toolkit somewhere
# unpack OptiX SDK somewhere
# add nvcc.exe and cl.exe to PATH

# ignored warnings (known):
# C4061: unused enumerator
# C4245: <some type> unsigned int conversion
# C4365: <some type> unsigned int conversion
# C4514: unreferenced inline function deleted
# C4571: catch semantic changed in VC 7.1
# C4625: copy constructor defined as deleted
# C4626: assign constructor defined as deleted
# C4668: undefined macro
# C4710: function not inline
# C4711: function automatically inlined
# C4774: VC header <xlocnum> warning
# C4820: n bytes added
# C4996: function or variable may be unsafe
# C5026: shift constructor defined as deleted
# C5027: shift constructor defined as deleted
# C5039: risk of undefined behaviour on exception

IGNORED_WARNINGS = \
	/wd4061 \
	/wd4245 \
	/wd4365 \
	/wd4514 \
	/wd4571 \
	/wd4625 \
	/wd4626 \
	/wd4668 \
	/wd4710 \
	/wd4711 \
	/wd4774 \
	/wd4820 \
	/wd4996 \
	/wd5026 \
	/wd5027 \
	/wd5039 \

CXXFLAGS = /Wall $(IGNORED_WARNINGS) /O2 -DNOMINMAX -DNDEBUG $$CXXFLAGS
NVCCFLAGS = -rdc true -ccbin cl.exe -Xcompiler $(subst $(space),$(comma),$(CXXFLAGS)) -arch sm_75 -use_fast_math

INC = \
	-I$$(cygpath -w /usr/lab/cudacons/cuda_11.4.0_471.11_win10/cuda_cudart/cudart/include) \
	-I$$(cygpath -w /usr/lab/cudacons/cuda_11.4.0_471.11_win10/libcurand/curand_dev/include) \
	-I$$(cygpath -w /usr/lab/cudacons/NVIDIA-OptiX-SDK-7.3.0-win64/include) \

LIB = \
	-L$$(cygpath -w /usr/lab/cudacons/cuda_11.4.0_471.11_win10/cuda_cudart/cudart/lib/x64) \

LD = nvcc
else
CXXFLAGS = -Wall -Wsign-compare -Wno-multichar -funroll-loops -fPIC -msse -msse2 -msse3 -mfpmath=sse -O3 -g3 -DNDEBUG $$CXXFLAGS
NVCCFLAGS = -rdc true -std c++11 -ccbin g++ -Xcompiler $(subst $(space),$(comma),$(CXXFLAGS)) -arch sm_75 -use_fast_math

INC = \
	-I/usr/local/cuda/include \
	-I/usr/local/optix/include \

LIB = \
	-L/usr/local/cuda/lib64 -lcudart -lpthread -lrt -ldl \
	-lm \

LD = g++
endif

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
	bin2c -c -p 0 -n $(basename $<)_ptx $< > $@

all: $(EXE) $(IMG)

$(EXE): $(OBJ)
	$(LD) -o $@ $^ \
		$(LIB) \



clean:
ifdef winos
	rm -f rtwo.lib rtwo.exp
endif
	rm -f $(OBJ)

tidy: clean
	rm -f $(EXE) $(IMG)
	rm -rf $(OPTIX_CACHE_PATH)



$(IMG): $(EXE)
	OPTIX_CACHE_PATH=$${OPTIX_CACHE_PATH:-$(OPTIX_CACHE_PATH)} ./$< | magick ppm:- $@
