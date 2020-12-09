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
	v.h \
	util.h \

IMG = rtow.png

.PHONY: all clean tidy

.cu.ptx:
	echo nvcc

.ptx.c:
	echo bin2c

all: $(EXE) $(IMG)

$(OBJ): $(CLS)

$(EXE): $(OBJ)
	g++ -o $@ $<



clean:
	rm -f $(OBJ)

tidy: clean
	rm -f $(EXE) $(IMG)



$(IMG): $(EXE)
	./$< | magick ppm:- $@
ifdef winos
	cmd /c $@
endif
