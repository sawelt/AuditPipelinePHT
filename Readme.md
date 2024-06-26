[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11505228.svg)](https://doi.org/10.5281/zenodo.11505228)

## PASTA-4-PHT

This repository contains supplementary material for the paper titled "PASTA-4-PHT: A Pipeline for Automated Security and Technical Audits for the Personal Health Train" by Welten et al. The available material includes source code in form of Dockerfiles for each step of the pipeline as well as source code for the container trains used to evaluate the security pipeline. Alongside the source code we provide screenshots of the evaluation results that the pipeline generates within each subfolder of the _trains_ directory. The following Diagram shows the infrastructure of the PHT framework and the security audit pipeline, which is subject of this paper:

![Image](/img/PaperRealization.png "Paper Realization Diagram")

## Related Work
- ["Multi-Institutional Breast Cancer Detection Using a Secure On-Boarding Service for Distributed Analytics"](https://www.mdpi.com/1603526)
- ["Distributed Skin Lesion Analysis Across Decentralised Data Sources"](https://ebooks.iospress.nl/volumearticle/56886)
- ["Will it run? -A Proof of Concept for Smoke Testing Decentralized Data Analytics Experiments" by Welten et al.](https://www.frontiersin.org/articles/10.3389/fmed.2023.1305415/abstract)

## Screenshots
The security audit pipeline generates a PDF report that summarizes the findings of the individual pipeline steps.

PDF Report Code Annotations (1)       | PDF Report Static Analysis (2)          |  PDF Report Vulnerabilities (3) | PDF Report Summary (4)
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![Image](/img/pdf_code.png "PDF Report Code Annotations") |  ![Image](/img/pdf_sast.png "PDF Report Static Analysis")   |  ![Image](/img/pdf_image.png "PDF Report  Vulnerabilities") | ![Image](/img/pdf_summary.png "PDF Report Summary")

The PDF report is provided through the trainwiki frontend. Here, users can find a comprehensible summary of the audit findings as well as a preview of the more extensive PDF report with the option to download the pdf file. We provide the full pdf report for each train within the _trains_ directory.

![Image](/img/trainwiki.png "Trainwiki Screenshot")

## Setup
In order to set up the project, please follow these guidelines:

1. Create a folder for each train in the root directory.
2. Place the YAML file in the root directory.

For detailed information on the pipeline steps, refer to the [`/pipeline/Readme.md`](/pipeline/Readme.md) file. In the [`/pipeline/Readme.md`](/pipeline/Readme.md) file, you will find code examples for each step. Feel free to choose the steps that are suitable for your needs and create your own file.

## Disclaimer
Certain components or tools included in this project are provided by third-party vendors and are subject to their respective licenses. Please refer to the documentation or license files accompanying each third-party tool for details on their terms of use and licensing conditions. The licenses of these third-party tools are solely governed by their respective vendors and not by the terms of this project.