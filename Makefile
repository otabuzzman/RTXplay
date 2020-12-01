ifeq ($(OS),Windows_NT)
	winos := 1
else
	linos := 1
endif

ifdef winos
EXE = main.exe
else
EXE = main
endif

CLS = \
	V.h \
	rgb.h \
	Ray.h \

IMG = rtow.png

.PHONY: all clean tidy

all: $(EXE) $(IMG)

main.o: $(CLS)

$(EXE): main.o
	g++ -o $@ $<



clean:
	rm -f *.o

tidy: clean
	rm -f $(EXE) $(IMG)



$(IMG): $(EXE)
	./$< | magick ppm:- $@
ifdef winos
	cmd /c $@
endif
