<p align="center"><h1 align="center">1000_SOIL_CHEMODIVERSITY</h1></p>
<p align="center">
    <img src="https://img.shields.io/github/license/ArjunChakrawal/1000_soil_chemodiversity?style=default&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
    <img src="https://img.shields.io/github/last-commit/ArjunChakrawal/1000_soil_chemodiversity?style=default&logo=git&logoColor=white&color=0080ff" alt="last-commit">
    <img src="https://img.shields.io/github/languages/top/ArjunChakrawal/1000_soil_chemodiversity?style=default&color=0080ff" alt="repo-top-language">
    <img src="https://img.shields.io/github/languages/count/ArjunChakrawal/1000_soil_chemodiversity?style=default&color=0080ff" alt="repo-language-count">
    <a href="https://doi.org/10.5281/zenodo.15225637"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.15225637.svg" alt="DOI"></a>
</p>
<p align="center"><!-- default option, no dependency badges. -->
</p>
<p align="center">
    <!-- default option, no dependency badges. -->
</p>
<br>

##  Table of Contents

- [ Overview](#-overview)
- [ Project Structure](#-project-structure)
  - [ Project Index](#-project-index)
- [ Getting Started](#-getting-started)
  - [ Prerequisites](#-prerequisites)
  - [ Installation](#-installation)
  - [ Usage](#-usage)
  - [ Testing](#-testing)
- [ Project Roadmap](#-project-roadmap)
- [ Contributing](#-contributing)
- [ License](#-license)
- [ Acknowledgments](#-acknowledgments)

---

##  Overview

<p> This repository contains all scripts and analysis workflows associated with our study:“Dissolved Organic Matter Chemodiversity as a Predictor of Microbial Metabolism and Soil Respiration.” In this study, we investigated how the molecular complexity of dissolved organic matter (DOM) influences microbial metabolism and soil respiration across diverse soil ecosystems. We integrated high-resolution DOM chemistry, characterized using Fourier Transform Ion Cyclotron Resonance Mass Spectrometry (FTICR-MS), with potential soil respiration rates from sites across the United States. </p>

Using data from the Molecular Observation Network (MONet), specifically the [1000 Soils Pilot Dataset](https://zenodo.org/records/7706774), we pursued two main objectives:

1. To evaluate statistical relationships between DOM chemodiversity indices and microbial respiration.

2. To test the predictive performance of empirical and kinetic models incorporating DOM chemistry and soil biogeochemical variables.


---

##  Project Structure

```sh
└── 1000_soil_chemodiversity/
    ├── 1000Soil_data
    │   ├── 1000S_Dataset_Biogeochem_Biomass_Tomography_WEOM_2023_06_12.xlsx
    │   ├── 1000Soils_Metadata_Site_Mastersheet_v1.xlsx
    │   ├── Readme File_v2.xlsx
    │   ├── icr_v2_corems2.csv
    │   └── md5sums.txt
    ├── Figure3_4.R
    ├── LICENSE
    ├── README.md
    ├── SI_figres.py
    ├── figs
    │   ├── Figure1A.html
    │   ├── Figure1B.svg
    │   ├── Figure1C.png
    │   ├── Figure2.png
    │   ├── Figure5.png
    │   ├── figure3.png
    │   ├── figure4.png
    │   └── SI
    │  		├── FigureS1-12
    │  		├── regression_table_R1.docx
    ├── figures_1_2_5.py
    ├── my_functions.py
    └── processed_data
        ├── df_BG_ICR.csv
        └── icr_by_class.csv
```


###  Project Index
<details open>
    <summary><b><code>1000_SOIL_CHEMODIVERSITY/</code></b></summary>
    <details> <!-- __root__ Submodule -->
        <summary><b>__root__</b></summary>
        <blockquote>
            <table>
            <tr>
                <td><b><a href='https://github.com/ArjunChakrawal/1000_soil_chemodiversity/blob/master/my_functions.py'>my_functions.py</a></b></td>
                <td><code>❯ REPLACE-ME</code></td>
            </tr>
            <tr>
                <td><b><a href='https://github.com/ArjunChakrawal/1000_soil_chemodiversity/blob/master/Figure3_4.R'>Figure3_4.R</a></b></td>
                <td><code>❯ REPLACE-ME</code></td>
            </tr>
            <tr>
                <td><b><a href='https://github.com/ArjunChakrawal/1000_soil_chemodiversity/blob/master/SI_figres.py'>SI_figres.py</a></b></td>
                <td><code>❯ REPLACE-ME</code></td>
            </tr>
            <tr>
                <td><b><a href='https://github.com/ArjunChakrawal/1000_soil_chemodiversity/blob/master/figures_1_2_5.py'>figures_1_2_5.py</a></b></td>
                <td><code>❯ REPLACE-ME</code></td>
            </tr>
            </table>
        </blockquote>
    </details>
    <details> <!-- figs Submodule -->
        <summary><b>figs</b></summary>
        <blockquote>
            <table>
            <tr>
                <td><b><a href='https://github.com/ArjunChakrawal/1000_soil_chemodiversity/blob/master/figs/Figure1A.html'>Figure1A.html</a></b></td>
                <td><code>❯ REPLACE-ME</code></td>
            </tr>
            </table>
        </blockquote>
    </details>
</details>

---
##  Getting Started

###  Prerequisites

Before getting started with 1000_soil_chemodiversity, ensure your runtime environment meets the following requirements:

- **Programming Language:** Python


###  Installation

Install 1000_soil_chemodiversity using one of the following methods:

**Build from source:**

1. Clone the 1000_soil_chemodiversity repository:
```sh
❯ git clone https://github.com/ArjunChakrawal/1000_soil_chemodiversity
```

2. Navigate to the project directory:
```sh
❯ cd 1000_soil_chemodiversity
```

---

##  Contributing

- **💬 [Join the Discussions](https://github.com/ArjunChakrawal/1000_soil_chemodiversity/discussions)**: Share your insights, provide feedback, or ask questions.
- **🐛 [Report Issues](https://github.com/ArjunChakrawal/1000_soil_chemodiversity/issues)**: Submit bugs found or log feature requests for the `1000_soil_chemodiversity` project.
- **💡 [Submit Pull Requests](https://github.com/ArjunChakrawal/1000_soil_chemodiversity/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your github account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone https://github.com/ArjunChakrawal/1000_soil_chemodiversity
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to github**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="left">
   <a href="https://github.com{/ArjunChakrawal/1000_soil_chemodiversity/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=ArjunChakrawal/1000_soil_chemodiversity">
   </a>
</p>
</details>

---

##  License

This project is protected under the [MIT License](https://choosealicense.com/licenses/mit/#) License.

---

##  Acknowledgments

“Soil data were provided by the Molecular Observation Network (MONet) at the Environmental Molecular Sciences Laboratory (https://ror.org/04rc0xn13), a DOE Office of Science user facility sponsored by the Biological and Environmental Research program under Contract No. DE-AC05-76RL01830. The work (proposal: 10.46936/10.25585/60008970) conducted by the U.S. Department of Energy, Joint Genome Institute (https://ror.org/04xm1d337), a DOE Office of Science user facility, is supported by the Office of Science of the U.S. Department of Energy operated under Contract No. DE-AC02-05CH11231. The Molecular Observation Network (MONet) database is an open, FAIR, and publicly available compilation of the molecular and microstructural properties of soil. Data in the MONet open science database can be found at https://sc-data.emsl.pnnl.gov/.The National Ecological Observatory Network is a program sponsored by the National Science Foundation and operated under cooperative agreement by Battelle. A portion of soil samples collected for this research were obtained through NEON Research Support Services."

---
