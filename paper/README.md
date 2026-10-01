# Manuscript source

LaTeX source of the preprint (elsarticle class, Energy Economics format).

    main.tex                  document class, frontmatter, section includes, declarations
    refs.bib                  bibliography (65 entries; author-year, elsarticle-harv)
    highlights.txt            five highlights, each under 85 characters
    sections/                 sections 01-09 and appendices A-D
    figures/*.pdf             vector figures (Type 1 fonts)
    figures/pipeline_src.tex  TikZ source for Figure 1 (compile standalone)
    make_figures_v2.py        regenerates Figures 2-8 from tracked output files
    make_pde_mc.py            runs the PDE and Monte Carlo strike sweep (figures/pde_mc.json)

## Compiling

pdfLaTeX and BibTeX, main document `main.tex`; run pdfLaTeX, BibTeX, then
pdfLaTeX twice (or `latexmk -pdf main.tex`). In Overleaf: upload the folder,
set the compiler to pdfLaTeX and recompile.

## Regenerating the figures

    python paper/make_figures_v2.py . paper/figures

The script reads only files under `outputs/` and `inputs/`. With a LaTeX
installation on the path it renders text in Computer Modern; otherwise it
falls back to the matplotlib default serif font.

## Compiled PDFs

    Uzkan_Sacli_2026_forward_anchored_option_valuation_v1.pdf   first SSRN version
    Uzkan_Sacli_2026_forward_anchored_option_valuation_v2.pdf   current version

Increase the version suffix on every material revision.
