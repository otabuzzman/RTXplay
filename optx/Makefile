ifeq ($(OS),Windows_NT)
	winos := 1
else
	linos := 1
endif

empty =
space = $(empty) $(empty)

comma := ,

ifdef winos
o = .obj
x = .exe
else
o = .o
x =
endif

EXE = rtwo$x

OBJ = \
	glad$o \
	rtwo$o \
	args$o \
	scene$o \
	sphere$o \
	camera_i$o \
	optics_i$o \
	camera_r$o \
	optics_r$o \
	postproc_h$o \
	postproc_d$o \
	denoiser$o \
	paddle$o \
	simpleui$o \
	simplesm$o \
	frag$o \
	vert$o \

IMG = rtwo.png

OPTIX_CACHE_PATH = ./OptixCache_$$USER

# scene objects (wavefront OBJ format)
SCN = \
	sphere_3.scn \
	sphere_6.scn \
	sphere_8.scn \
	sphere_9.scn \

# default target
$(IMG): $(EXE) $(SCN)
	OPTIX_CACHE_PATH=$${OPTIX_CACHE_PATH:-$(OPTIX_CACHE_PATH)} ./$< | magick ppm:- $@

# preprocessor macros:
# -DRECURSIVE - use recursive shaders instead of iterative
# -DCURAND    - use CUDA RNG instead of rand48

ifdef winos
# prerequisites:
# install VS2017 (Community)
# install Cygwin (gcc, g++, make, ...)
# install CUDA Toolkit (use defaults)
# install OptiX SDK (use defaults)
# add cl.exe to PATH

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
LDFLAGS   = /WX /NODEFAULTLIB:MSVCRT # /VERBOSE:LIB
NVCCFLAGS = -Xcompiler $(subst $(space),$(comma),$(CXXFLAGS)) -Xlinker $(subst $(space),$(comma),$(LDFLAGS)) -arch sm_75 -use_fast_math -lineinfo

INC = \
	-I"$$(cygpath -w '/cygdrive/c/program files/nvidia gpu computing toolkit/cuda/v11.4/include')" \
	-I"$$(cygpath -w '/cygdrive/c/programdata/nvidia corporation/optix sdk 7.3.0/include')" \
	-I$$(cygpath -w /usr/local/include) \

LIB = \
	-L"$$(cygpath -w '/cygdrive/c/program files/nvidia gpu computing toolkit/cuda/v11.4/lib/x64')" \
	-L$$(cygpath -w /usr/local/lib) -lglfw3_mt -lopengl32 \
	-ladvapi32 -lshell32 -luser32 -lgdi32 -lkernel32 \

# native OpenGL image viewer
glxview.exe: glxview.cxx glad.obj vert.obj frag.obj | stb_image.h
	g++ -o $@ $^ \
		-I. -I/usr/local/include \
		-L/usr/local/lib -lglfw3 \

args.exe: args.cxx
	g++ -DMAIN -o $@ $< \

else
CXXFLAGS  = -std=c++11 -Wall -Wsign-compare -Wno-multichar -funroll-loops -fPIC -msse -msse2 -msse3 -mfpmath=sse -O3 -g3 -DNDEBUG $$CXXDEFS
LDFLAGS   = --fatal-warnings
NVCCFLAGS = -Xcompiler $(subst $(space),$(comma),$(CXXFLAGS)) -Xlinker $(subst $(space),$(comma),$(LDFLAGS)) -arch sm_75 -use_fast_math -lineinfo

INC = \
	-I/usr/local/cuda/include \
	-I/usr/local/optix/include \
	-I/usr/local/include \

LIB = \
	-L/usr/local/cuda/lib64 -lcudart -lpthread -lrt -ldl \
	-L/usr/local/lib -lglfw3 \
	-lm \

# interop'ed OpenGL image viewer
glxview: glxview.cxx glad.o vert.o frag.o | stb_image.h
	nvcc -o $@ $^ \
		$(NVCCFLAGS) \
		-I. $(INC) \
		$(LIB) \

endif

vpath glad.c glad

.PHONY: all clean tidy
.SUFFIXES: .cxx .cu .glsl .png .ptx .scn

%$o: %.c
	gcc -c $< -o $@ \
		-I. \

%$o: %.cxx
	nvcc -dc $< -o $@ \
		$(NVCCFLAGS) \
		-I. $(INC) \

# NVCC separate compilation needed (UG)
postproc_d$o: postproc_h$o
	nvcc -dlink $< -o $@ \
		$(NVCCFLAGS) \

postproc_h$o: postproc.cu
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
	nvcc -o $@ $^ \
		$(NVCCFLAGS) \
		$(LIB) \

$(SCN): sphere$x
sphere_3.scn:
	./sphere$x 1. 3 >$@
sphere_6.scn:
	./sphere$x 1. 6 >$@
sphere_8.scn:
	./sphere$x 1. 8 >$@
sphere_9.scn:
	./sphere$x 1. 9 >$@

sphere$x: sphere.cxx
	nvcc -DMAIN -o $@ $^ \
		$(NVCCFLAGS) \
		$(INC) \



clean:
	rm -f $(OBJ)
ifdef winos
	rm -f rtwo.lib rtwo.exp
	rm -f sphere.lib sphere.exp
endif

eclean: clean
	rm -f $(EXE) $(IMG)
	rm -rf $(OPTIX_CACHE_PATH)
	rm -f glxview$x sphere$x $(SCN)
ifdef winos
	rm -f args.exe
endif

lclean: eclean
	rm -rf glad KHR

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
