Generalized Mixed Model-based Clustering of Grouped Data with Application in Education
================


# Data

### Abstract

The case study considered in this paper is implemented using a subset of the OECD'S PISA survey data of 2018 to cluster countries standing on their innumeracy levels, i.e., the levels of mathematical illiteracy. 
The OECD’s PISA measures 15-year-olds’ knowledge and skills in reading, mathematics, and science to handle real-life challenges. 
Our focus is on mathematical performance, which evaluates students’ ability to apply math in various contexts. 
The global indicators for the United Nations Sustainable Development Goals identify a minimum Level of Proficiency - computed on the obtained scores - 
that all children should acquire by the end of secondary education: students below this level are considered low-achieving students. 
We aim to investigate the effect the countries involved in the OECD’s PISA 2018 survey have on the rate of low-achieving students in mathematics.
The data are preprocessed as described in Section 3.1 of the paper.

### Availability

Raw data are publicly available at https://www.oecd.org/pisa/data/2018database/.

### Description

Cleaned data that are produced by processing raw input data are placed in `data`. Specifically:

* `data`/`raw_data_import_and_preprocessing.R` is an R script containing the code for importing, preprocessing and merging the raw data input (downloadable from https://webfs.oecd.org/pisa2018/SPSS_STU_QQQ.zip and https://webfs.oecd.org/pisa2018/SPSS_SCH_QQQ.zip);

* `data`/`df_level2.csv` is the cleaned dataset produced by processing the raw input data mentioned above;

* `data`/`iso_countries.xlsx` is the excel retrieved from the Codebook (https://webfs.oecd.org/pisa2018/PISA2018_CODEBOOK.xlsx) that associates each country with its ISO code.




# Code

### Abstract

The code includes all functions necessary to implement and run the results found in Sections 3 and 4 of the paper (and S4 and S5 of Supplementary Materials).

Scripts that execute the overall workflow to carry out an analysis and generate results for the manuscript are placed in the main directory.
The folder `code` contains the core code to implement the GLMMDRE and various utility/auxiliary functions.


### Description

The contents of the folder `code` are as follows:
1. `code`/`DataGeneration_class.py`: class for generating data, to be used both for real case studies and for simulations, both for Bernoulli and Poisson responses;
2. `code`/`AlgorithmComp_class.py`: class that implements all the utilities for running the GLMMDRE algorithm;
3. `code`/`Auxiliary_functions.py`: auxiliary function for the GLMMDRE algorithm;
4. `code`/`algorithm_alpha.py`: function for GLMMDRE algorithm with $\alpha$-criterion, to be used both for real case studies and for simulations, both for Bernoulli and Poisson responses;
5. `code`/`algorithm_t.py`: function for GLMMDRE algorithm with t-criterion, to be used both for real case studies and for simulations, both for Bernoulli and Poisson responses.

In the main we can find:
1. `GLMMDRE_case_study.ipynb`: for running the GLMMDRE (case study), takes `data`/`df_level2.csv` in input and produces pickles files within `output`/`case_study_results`;
2. `GLMM_case_study.R`: for running the parametric GLMM (case study), takes `data`/`df_level2.csv` in input and produces `output`/`df_level2_pred.csv`;
3. `Check_Poisson_distribution.ipynb`: for testing within the case study whether Y_MATH is Poisson-distributed;
5. `Analysis_case_study_results.ipynb`: script for producing case study results reported within Sections 3 of the paper and S4 of Supplementary Materials;
6. `GLMMDRE_simulation_study.ipynb`: for running the GLMMDRE (simulation study), produces pickles files within `output`/`simulation_study_results`;
7. `Analysis_simulation_study_results.ipynb`: script for producing some results reported within Sections 4 of the paper and S5 of Supplementary Materials;
8. `DG_GLMMDRE_comparison_state_of_art.ipynb`: for generating data (.csv files are saved in folders `output`/`comparison_state_of_art`/`Bernoulli_DG_output` and `output`/`comparison_state_of_art`/`Poisson_DG_output`, respectively for Bernoulli and Poisson responses) and running the GLMMDRE on those files (.csv files with results are saved in `output`/`comparison_state_of_art`/`Bernoulli_GLMMDRE_output` and `output`/`comparison_state_of_art`/`Poisson_GLMMDRE_output`, respectively for Bernoulli and Poisson responses);
10. `GLMM_comparison_state_of_art.R`: script for producing the other results reported within Sections 4 of the paper and S5 of Supplementary Materials.

The `output` directory holds objects derived from computations, including results of simulations or real data analyses. The contents of the folder are as follows, as already mentioned:
1. `output`/`df_level2_pred.csv`: .csv with results of `GLMM_case_study.R`;
2. `output`/`simulation_study_results`: folder with pickles results of `GLMMDRE_simulation_study.ipynb`;
3. `output`/`comparison_state_of_art`: folder with other folders result of `DG_GLMMDRE_comparison_state_of_art.ipynb`.


# Instructions for use

### Reproducibility

Results of the paper reported in Sections 3 and 4 (and S4 and S5 of Supplementary Materials) can be reproduced by running `Analysis_case_study_results.ipynb`, `Analysis_simulation_study_results.ipynb` and `GLMM_comparison_state_of_art.R`

