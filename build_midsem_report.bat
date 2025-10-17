@echo off
REM Build script for midsem_report.tex using MiKTeX (pdflatex + bibtex)
REM Place this file inside the folder that contains midsem_report.tex and run it from cmd.exe

:: Change to script directory (works even if path contains spaces)
cd /d "%~dp0"

necho === Building midsem_report.tex ===

necho Running pdflatex (1st pass)...
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error midsem_report.tex
nif errorlevel 1 (
    echo pdflatex failed. Check midsem_report.log
    goto :done
)

n:: Run bibtex if references.bib exists or if .aux references citations
nif exist references.bib (
    echo Running bibtex...
    bibtex midsem_report
    if errorlevel 1 (
        echo BibTeX failed. Check midsem_report.blg and midsem_report.log
        goto :done
    )
) else (
    echo No references.bib found; skipping bibtex.
)

necho Running pdflatex (2nd pass)...
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error midsem_report.tex
nif errorlevel 1 (
    echo pdflatex failed on 2nd pass. Check midsem_report.log
    goto :done
)

necho Running pdflatex (3rd pass for crossrefs)...
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error midsem_report.tex
nif errorlevel 1 (
    echo pdflatex failed on 3rd pass. Check midsem_report.log
    goto :done
)

necho === Build finished ===
necho Output: "%~dp0midsem_report.pdf"

n:done
necho See midsem_report.log for details if anything went wrong.
pause
