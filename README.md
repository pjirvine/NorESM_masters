This folder contains all that is needed to produce figures 1-4 and table 1 from Irvine et al. 2020. The code for figure 5 is present but is not functioning at present and figure 6 was produced offline.

Contents:
- Scripts = contains all plotting scripts, including the jupyter notebook which generates all figures and plots.
- processing_scripts = contains scripts specific to producing the data on Harvard's Odyssey system. All processed data is available and these scripts do not need to be run.
- Figures = final figures produced from code in "scripts/figure_sections"
- plots = additional plots not used in paper produced from code in scripts/plot_sections"
- tables = tables for paper and summary data for plotting.
- archive = contains scripts specific to producing the input data on Harvard's Odyssey system. All processed data is available and these scripts do not need to be run.
- irvine_erl_20.yml = python environment specification for project.

conda env create -f irvine_erl_20.yml
