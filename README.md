# Examining the consistency of Earth Observation-based Machine Learning models for predicting poverty in sub-Saharan Africa

<a target="_blank" href="https://datalumina.com/">
    <img src="https://img.shields.io/badge/Datalumina-Project%20Template-2856f7" alt="Datalumina Project" />
</a>

This repository contains the code required to reproduce the figures, tables, and supplementary visualizations associated with the manuscript submitted by Mlambo et al. (2026).

## Raw data

All the data used in these analyses are available from online repositories and can be downloaded for individual countries. 
- <a href=https://dhsprogram.com/data/>Demographic and Health Surveys (DHS) </a>can be obtained upon registration.
- <a href=https://data.humdata.org/dataset/relative-wealth-index>Relative wealth index (RWI)</a> data by Chi et al.(2022).
- <a href=https://doi.org/10.7910/DVN/5OGWYM>International wealth index (IWI) </a>poverty maps by Lee and Braithwaite (2022).
- <a href=https://github.com/sustainlab-group/africa_poverty>Harmonized wealth index </a>by Yeh et al.’s(2020).

## Instructions
Install the required packages from `requirements.txt` into a virtual environment.
Download the source data from the links provided and extract into ./data/raw.
Run the scripts in ./src/ in order from 1 - 12 to generate the data for figures.
Run ./notebooks/Figures.ipynb to reproduce the figures in the article.


## Project Organization

```
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data.
│   ├── processed      <- The final datasets for reporting and generating figures.
│   └── raw            <- Folders DHS, Chi, Lee and Yeh containing the source data.
│
├── notebooks          <- Jupyter notebooks for generating figures
│
├── reports            <- Generated analysis
│   └── figures        <- Generated figures to be used in the manuscript.
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment.
│
└── src                         <- Source code for this project.
    │
    ├── config.py               <- Store variables and configuration
```

--------
