main.pdf: main.tex references.bib Makefile
	pdflatex main
	bibtex main
	pdflatex main
	pdflatex main

mac: main.pdf
	open main.pdf

clean:
	rm -f main.pdf main.bbl main.out *.aux *.log *.blg 
