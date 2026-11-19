# Submission version for *Empirical Software Engineering* (Springer)

Generated from `../paper/main.tex` (the IEEE TSE version). Only the formatting
that EMSE requires was changed; the text, numbers, tables, and figures are the same.

| EMSE requirement (submission guidelines) | In this folder |
|---|---|
| Single-blind review: no anonymization needed | author names and links kept |
| Abstract of 150–250 words | 248 words |
| 4–6 keywords | 6 keywords |
| Citations by name and year in parentheses; DOI links in references | `natbib` author–year + `plainnat`, full `https://doi.org/` links |
| Separate title page: title, authors, affiliations, corresponding author e-mail, ORCIDs, acknowledgments, declarations | `title_page.tex` |
| Declarations (funding, competing interests, ethics, consent, data/code availability, author contributions) | in `title_page.tex` and at the end of `manuscript.tex` |
| Use of AI tools documented in the Methods section | Section "Implementation and reproducibility" |
| No subfolders in the LaTeX submission | all files sit in this folder |

## Build the PDFs

On [Overleaf](https://www.overleaf.com) (compiler: pdfLaTeX), upload all files of this
folder into one project and compile twice, once with each main document
(Menu → Main document):

1. `manuscript.tex` → manuscript PDF (uses `references.bib`)
2. `title_page.tex` → title-page PDF

Locally: `pdflatex manuscript && bibtex manuscript && pdflatex manuscript && pdflatex manuscript`
and `pdflatex title_page`.

Springer recommends, but does not require, its Springer Nature LaTeX template
(`sn-jnl`). To use it, start the "Springer Nature LaTeX Template" on Overleaf, choose an
author–year reference option (for example `sn-mathphys-ay`), and paste the body of
`manuscript.tex`.

## Upload checklist

- [ ] Manuscript: `manuscript.tex`, `references.bib`, and the 10 `.png` figures (or the compiled PDF)
- [ ] Title page: `title_page.tex` (or its compiled PDF) as a separate file
- [ ] Corresponding author: Alla Khaled Almuqadimi, al3muqadimi@gmail.com
- [ ] Check the author-contribution statement and the second ORCID before submitting
