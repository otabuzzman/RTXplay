ifeq ($(OS),Windows_NT)
	winos := 1
else
	linos := 1
endif

ifdef winos
EXE = rtow.exe
else
EXE = rtow
endif

OBJ = \
	rtow.o \

CLS = \
	camera.h \
	optics.h \
	ray.h \
	sphere.h \
	thing.h \
	things.h \
	util.h \
	v.h \

IMG = rtow.png

.PHONY: all clean tidy
.SUFFIXES: .cxx .png

all: $(EXE) $(IMG)

$(OBJ): $(CLS)

.cxx.o:
	g++ $$CXXFLAGS -c $<

$(EXE): $(OBJ)
	g++ $$CXXFLAGS -o $@ $<



clean:
	rm -f $(OBJ)

tidy: clean
	rm -f $(EXE) $(IMG)



$(IMG): $(EXE)
	./$< | magick ppm:- $@
ifdef winos
	cmd /c $@
endif

malen-nach-strahlen.tar.gz:
	tar zcf $@ \
		camera.h \
		Makefile \
		ray.h \
		rtow.cxx \
		sphere.h \
		thing.h \
		things.h \
		util.h \
		v.h \
		optics.h \
		optx/ \
		optx/camera.cu \
		optx/camera.h \
		optx/Makefile \
		optx/optics.cu \
		optx/optics.h \
		optx/rtwo.cxx \
		optx/rtwo.h \
		optx/sphere.cxx \
		optx/sphere.h \
		optx/thing.h \
		optx/things.h \
		optx/util.h \
		optx/util_cpu.h \
		optx/util_gpu.h \
		optx/v.h \
		--transform 's,^,malen-nach-strahlen/,' \
