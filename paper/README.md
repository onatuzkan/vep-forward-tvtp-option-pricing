# Manuscript source

LaTeX source for the preprint (elsarticle class).

    main.tex                  document class, frontmatter, section includes
    refs.bib                  bibliography (42 entries, all cited)
    sections/                 one file per section (01-10) and appendix (A-C)
    figures/*.pdf             vector figures
    figures/pipeline_src.tex  TikZ source for Figure 1 (compile standalone)
    make_figures.py           regenerates data figures from the model repo
    make_pde_mc.py            runs the PDE/Monte Carlo verification sweep

## Compiling in Overleaf

1. Upload as a new project (Upload Project -> zip).
2. Menu -> Compiler: pdfLaTeX; Main document: main.tex
3. Recompile twice (BibTeX runs between passes).

All figures are vector PDFs; the document does not load TikZ/PGF.

## Verified state

pdfLaTeX + BibTeX: 0 errors, 0 undefined references, 0 overfull boxes,
11 figures, 6 tables, ~9,750 words (body 8,770 + appendices).

## Compiled PDF — naming convention

The current compiled PDF is committed alongside the sources as:

    paper/Uzkan_Sacli_2026_forward_anchored_option_valuation_v1.pdf

Bump the trailing `v<N>` on every material revision (`v2`, `v3`, ...).
Previous versions can be archived in `paper/archive/` before the new
version is copied in.  The build artefacts (`*.aux`, `*.log`, `*.bbl`,
etc., listed in `paper/.gitignore`) are NOT tracked; the compiled PDF
IS tracked, so any reader who pulls the repo sees the same PDF the
authors compiled.
