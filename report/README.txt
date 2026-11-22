DACN_HK261 — Specialized Project Report

Layout: specialized_report_EN from Downloads, based on the HCMUT thesis template.
The original class is in template-source/hcmut-thesis.cls. The active class uses
T1 fonts in English mode, retaining the template layout without requiring vntex.
The template license is in LICENSE-template.txt.

Edit cover information in report-info.tex.
Edit report content in chapters/main/*.tex and references in refs/references.tex.
The existing wording, equations, captions, and result placeholders were retained.
The template sample chapters and front-matter prose were not added to this report.

Build: latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
