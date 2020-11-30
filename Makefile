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

.PHONY: clean

all: $(EXE)

main.o: $(CLS)

$(EXE): main.o
	g++ -o $@ $<



clean:
	rm -f *.o

tidy: clean
	rm -f *.exe *.png
