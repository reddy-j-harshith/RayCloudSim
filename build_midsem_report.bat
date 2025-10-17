@echo off
REM Build script for midsem_report.tex using MiKTeX (pdflatex + bibtex)
REM Place this file inside the folder that contains midsem_report.tex and run it from cmd.exe

:: Change to script directory (works even if path contains spaces)
cd /d "%~dp0"

echo === Building midsem_report.tex ===

echo Running pdflatex (1st pass)...
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error midsem_report.tex
if errorlevel 1 (
    echo pdflatex failed. Check midsem_report.log
    goto :done
)

:: Run bibtex if references.bib exists
if exist references.bib (
    echo Running bibtex...
    bibtex midsem_report
    if errorlevel 1 (
        echo BibTeX failed. Check midsem_report.blg and midsem_report.log
        goto :done
    )
) else (
    echo No references.bib found; skipping bibtex.
)

echo Running pdflatex (2nd pass)...
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error midsem_report.tex
if errorlevel 1 (
    echo pdflatex failed on 2nd pass. Check midsem_report.log
    goto :done
)

echo Running pdflatex (3rd pass for crossrefs)...
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error midsem_report.tex
if errorlevel 1 (
    echo pdflatex failed on 3rd pass. Check midsem_report.log
    goto :done
)

echo === Build finished ===
echo Output: "%~dp0midsem_report.pdf"

:done
echo See midsem_report.log for details if anything went wrong.
pause
