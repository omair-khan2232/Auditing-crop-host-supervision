LaTeX build instructions

Compile from the `paper` directory containing main.tex,
supplementary_material.tex, the table_*.tex files, and the fig*.pdf/png files:

pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error supplementary_material.tex
pdflatex -interaction=nonstopmode -halt-on-error supplementary_material.tex
pdflatex -interaction=nonstopmode -halt-on-error cover_letter_IPA.tex

The manuscript uses the standard Elsevier elsarticle class. References are
embedded in main.tex, so BibTeX is not required.
