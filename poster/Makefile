BUILD := \
	poster \


DEPS := \
	beamerthemegemini.sty \
	beamercolorthemeimsa.sty \
	cityu_logo.pdf \


LATEX  := env LC_ALL=C LANG=C TEXMFVAR=/private/tmp/fsm4py-texmf-var TEXMFCONFIG=/private/tmp/fsm4py-texmf-config lualatex
BIBTEX := bibtex

TARGETS := $(patsubst %, %.pdf, $(BUILD))

# phony targets

all: $(TARGETS)

clean:
	rm -rf *.pdf *.aux *.bbl *.blg *.log *.nav *.out *.snm *.toc

.PHONY: all clean

# main targets

poster.pdf: poster.tex poster.bib $(DEPS)
	$(eval SRC_$@ = $(patsubst %.tex, %, $<))
	$(LATEX) $(SRC_$@)
	$(LATEX) $(SRC_$@)
