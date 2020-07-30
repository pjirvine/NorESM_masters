This folder contains all that is needed to produce figures 1-4 and table 1 from Irvine et al. 2020. The code for figure 5 is present but is not functioning at present and figure 6 was produced offline.

Contents:
- Scripts = contains all plotting scripts, including the jupyter notebook which generates all figures and plots.
- Figures = final figures produced from code in "scripts/figure_sections"
- plots = additional plots not used in paper produced from code in scripts/plot_sections"
- tables = tables for paper and summary data for plotting.
- archive = contains scripts specific to producing the input data on Harvard's Odyssey system. All processed data is available and these scripts do not need to be run.
- irvine_erl_20.yml = python environment specification for project.

To Produce the figures from this project:

1. Install anaconda / python on your linux machine
2. run this command in the terminal in the main folder to create the project environment:
  conda env create -f irvine_erl_20.yml
3. open jupyter lab by running this command:
  jupyter lab
4. jupyter lab should have opened up in a new browser window
5. in jupyter lab's navigation panel open the "scripts" directory then open the "Figure_Notebook.ipynb"
6. run each code box in the notebook. There are a couple of warnings but if it works you should see a set of figures being shown.
7. start adapting the code!
