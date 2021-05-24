

#
# maketex
#
TEX=template
LATEX=pdflatex

PDFLATEX=pdflatex
BIBTEX=bibtex
XDVI=kdvi
XPDF= okular
DVIPDF=dvipdf
DVIPS=dvips
PS2PDF=ps2pdf

$(TEX).aux:	$(TEX).tex
	$(LATEX) $(TEX)

fonts:
	pdffonts  $(TEX).pdf

dvipdf: $(TEX).aux
	$(DVIPS)  -Ppdf -G0 -ta4  -o $(TEX).ps $(TEX).dvi
# 	$(PS2PDF)   $(TEX).ps   $(TEX).pdf
# 	$(DVIPS) -Pdownload35 -Ppdf -G0 -ta4  -o $(TEX).ps $(TEX).dvi
	$(PS2PDF) -dEmbedAllFonts=true   $(TEX).ps   $(TEX).pdf
# 	gs -dSAFER -dNOPLATFONTS -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -sPAPERSIZE=letter -dCompatibilityLevel=1.4 -dPDFSETTINGS=/printer -dCompatibilityLevel=1.4 -dMaxSubsetPct=100 -dSubsetFonts=true -dEmbedAllFonts=true -sOutputFile=$(TEX)_emb.pdf -f $(TEX).pdf
# 	cp $(TEX)_emb.pdf $(TEX).pdf 
	rm  $(TEX).ps

bib:
	$(BIBTEX) $(TEX) 

clean:
	rm -f *.aux  *.lot *.bbl *.blg *.log *.dvi  *.ps *.toc *.out *.snm *.nav rss RSS *.loa *.spl

cleanup:	clean
	rm -f *~ *.gz *.zip *.4og *.4om *.4ct *.4tc *.tmp *.idv *.xref *.lg 

dvi:	
	$(XDVI) $(TEX)&

view:
	$(XDVI) $(TEX)&


pdf:
	$(XPDF)  $(TEX).pdf &

all: 
	$(LATEX)  $(TEX)
	$(BIBTEX) $(TEX)
	$(LATEX)  $(TEX)
	$(BIBTEX) $(TEX)
	$(LATEX)  $(TEX)

edit:
	kile `pwd`/$(TEX).tex &

	
complete: clean all dvipdf pdf

tar: cleanup
	tar -zcvf   $(TEX).tar.gz *

zip: 
	zip -r -9 submission.zip $(TEX).tex  *.png *.jpg *.jpeg *.bib cover_letter*  *.bbl  *.bst *.pdf


beam:
	$(LATEX)  #cil28-0398pres.tex 
	$(DVIPDF) # cil28-0398pres
pres:
	$(XPDF) #cil28-0398pres.pdf

mail: tar # envia um e-mail com a data e hora para $(MAILADRESS)
	date | mail  -s  $(TEX):06:tgz  -a  $(TEX).tar.gz  goliatt@gmail.com

sed:
	sh sed.sh
