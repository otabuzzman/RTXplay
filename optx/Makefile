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
	glad.o \
	rtwo.o \
	args.o \
	sphere.o \
	camera_i.o \
	optics_i.o \
	camera_r.o \
	optics_r.o \
	postproc_h.o \
	postproc_d.o \
	denoiser.o \
	paddle.o \
	simpleui.o \
	simplesm.o \
	frag.o \
	vert.o \

IMG = rtwo.png

OPTIX_CACHE_PATH = ./OptixCache_$$USER

all: $(EXE) $(IMG)

# preprocessor macros:
# -DRECURSIVE - use recursive shaders instead of iterative
# -DCURAND    - use CUDA RNG instead of rand48

ifdef winos
# prerequisites:
# install VS2017 (Community)
# install Cygwin (gcc, g++, make, ...)
# unpack CUDA Toolkit somewhere
# unpack OptiX SDK somewhere
# add nvcc.exe and cl.exe to PATH

# ignored warnings (known):
# C4061: unused enumerator
# C4100: unreferenced formal parameter
# C4191: <some type> unsafe type conversion
# C4245: <some type> unsigned int conversion
# C4365: <some type> unsigned int conversion
# C4456: declaration shadows previous
# C4505: unreferenced local function deleted
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
# C5045: Spectre mitigation inserted

IGNORED_WARNINGS = \
	/wd4061 \
	/wd4100 \
	/wd4191 \
	/wd4245 \
	/wd4365 \
	/wd4456 \
	/wd4505 \
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
	/wd5045 \

CXXFLAGS  = /Wall $(IGNORED_WARNINGS) /O2 -DNOMINMAX -DNDEBUG $$CXXDEFS
NVCCFLAGS = -ccbin cl.exe -Xcompiler $(subst $(space),$(comma),$(CXXFLAGS)) -arch sm_75 -use_fast_math -lineinfo

INC = \
	-I$$(cygpath -w /usr/lab/cudacons/cuda_11.4.0_471.11_win10/cuda_cudart/cudart/include) \
	-I$$(cygpath -w /usr/lab/cudacons/cuda_11.4.0_471.11_win10/libcurand/curand_dev/include) \
	-I$$(cygpath -w /usr/lab/cudacons/NVIDIA-OptiX-SDK-7.3.0-win64/include) \
	-I$$(cygpath -w /usr/local/include) \

LDFLAGS = -arch sm_75

LIB = \
	-L$$(cygpath -w /usr/lab/cudacons/cuda_11.4.0_471.11_win10/cuda_cudart/cudart/lib/x64) \
	-L$$(cygpath -w /usr/local/lib) -lglfw3 \

LD = nvcc

# native OpenGL image viewer
glxview.exe: glxview.cxx glad.o vert.o frag.o | stb_image.h
	g++ -o $@ $^ \
		-I. -I/usr/local/include \
		-L/usr/local/lib -lglfw3 \

# GNU getopt_long(3) missing on Windows
args.o: getopt.h

getopt.h:
	ln $$(cygpath -w /usr/include/$@) $@

args.exe: args.cxx
	g++ -DMAIN -o $@ $< \

else
CXXFLAGS  = -Wall -Wsign-compare -Wno-multichar -funroll-loops -fPIC -msse -msse2 -msse3 -mfpmath=sse -O3 -g3 -DNDEBUG $$CXXDEFS
NVCCFLAGS = -std c++11 -ccbin g++ -Xcompiler $(subst $(space),$(comma),$(CXXFLAGS)) -arch sm_75 -use_fast_math -lineinfo

INC = \
	-I/usr/local/cuda/include \
	-I/usr/local/optix/include \
	-I/usr/local/include \

LIB = \
	-L/usr/local/cuda/lib64 -lcudart -lpthread -lrt -ldl \
	-L/usr/local/lib -lglfw3 \
	-lm \

LDFLAGS =

LD = g++

# interop'ed OpenGL image viewer
glxview: glxview.cxx glad.o vert.o frag.o | stb_image.h
	nvcc -o $@ $^ \
		$(NVCCFLAGS) \
		-I. $(INC) \
		$(LIB) \

endif

vpath glad.c glad

.PHONY: all clean tidy
.SUFFIXES: .cxx .cu .glsl .png .ptx

.c.o:
	gcc -c $< -o $@ \
		-I. \

.cxx.o:
	nvcc -dc $< -o $@ \
		$(NVCCFLAGS) \
		-I. $(INC) \

# NVCC separate compilation needed (UG)
postproc_d.o: postproc_h.o
	nvcc -dlink $< -o $@ \
		$(NVCCFLAGS) \

postproc_h.o: postproc.cu
	nvcc -dc $< -o $@ \
		$(NVCCFLAGS) \
		-I. $(INC) \

.cu.ptx:
	nvcc -ptx $< -o $@ \
		$(NVCCFLAGS) \
		$(INC) \

.ptx.c:
	bin2c -c -p 0 -n $(basename $<)_ptx $< > $@

.glsl.c:
	bin2c -c -p 0 -n $(basename $<)_glsl $< > $@

$(EXE): $(OBJ)
	$(LD) -o $@ $^ \
		$(LDFLAGS) \
		$(LIB) \



clean:
ifdef winos
	rm -f rtwo.lib rtwo.exp
endif
	rm -f $(OBJ)

eclean: clean
	rm -f $(EXE) $(IMG)
	rm -rf $(OPTIX_CACHE_PATH)
ifdef winos
	rm -f glxview.exe args.exe
endif

lclean: eclean
	rm -rf glad KHR
ifdef winos
	rm -f getopt.h
endif

rclean: lclean
	rm -f glad.zip stb_image.h

tidy: rclean

# setup GLAD
glad/glad.c: glad/glad.h
	unzip -pq glad.zip src/$(@F) >$@
glad/glad.h: KHR/khrplatform.h
	mkdir -p $(@D)
	unzip -pq glad.zip include/$@ >$@
KHR/khrplatform.h: glad.zip
	mkdir -p $(@D)
	unzip -pq $< include/$@ >$@
glad.zip:
	@echo "*** 1. open GLAD in browser and click generation button:"
	@echo "***        https://glad.dav1d.de/#profile=core&language=c&specification=gl&loader=on&api=gl%3D3.3"
	@echo "*** 2. download ZIP archive and copy into $$(pwd) (this) folder"
	@echo '*** 3. re-run `make´'
	@false

# setup STBI
stb_image.h:
	wget -q https://raw.githubusercontent.com/nothings/stb/master/stb_image.h



$(IMG): $(EXE)
	OPTIX_CACHE_PATH=$${OPTIX_CACHE_PATH:-$(OPTIX_CACHE_PATH)} ./$< | magick ppm:- $@
