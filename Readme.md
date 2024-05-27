## PASTA-4-PHT

This repository contains supplementary material for the paper titled ["PASTA-4-PHT: A Pipeline for Automated Security and Technical Audits for the Personal Health Train" by Welten et al.](https://example.com). The available material includes source code in form of Dockerfiles for each step of the pipeline as well as screenshots of the evaluation results that the pipeline generates.

## Related Work
- ["Multi-Institutional Breast Cancer Detection Using a Secure On-Boarding Service for Distributed Analytics"](https://www.mdpi.com/1603526)
- ["Distributed Skin Lesion Analysis Across Decentralised Data Sources"](https://ebooks.iospress.nl/volumearticle/56886)
- ["Will it run? -A Proof of Concept for Smoke Testing Decentralized Data Analytics Experiments" by Welten et al.](https://www.frontiersin.org/articles/10.3389/fmed.2023.1305415/abstract)

## Screenshots
The security audit pipeline generates a PDF report that summarizes the findings of the individual pipeline steps.

PDF Report Code Annotations (1)       | PDF Report Static Analysis (2)          |  PDF Report Vulnerabilities (3) | PDF Report Summary (4)
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![Image](/img/pdf_code.png "PDF Report Code Annotations") |  ![Image](/img/pdf_sast.png "PDF Report Static Analysis")   |  ![Image](/img/pdf_image.png "PDF Report  Vulnerabilities") | ![Image](/img/pdf_summary.png "PDF Report Summary")

The PDF report is provided through the trainwiki frontend. Here, users can find a comprehensible summary of the audit findings as well as a preview of the more extensive PDF report with the option to download the pdf file.

![Image](/img/trainwiki.png "Trainwiki Screenshot")