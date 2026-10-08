# Draws every figure of the main text from the intermediate results in
# output/, in seconds and without a GPU.
# The commands that rerun the experiments themselves are in README.md.

PY ?= python

.PHONY: all figures figure1 figure2 figure3 figure4 figure5 figure6 figure7

all: figures

figures: figure1 figure2 figure3 figure4 figure5 figure6 figure7

figure1:
	$(PY) code/plotting/fig1_toy.py

figure2:
	$(PY) code/plotting/fig2_lg_grid.py

figure3:
	$(PY) code/plotting/fig3_lg_stability.py

figure4:
	$(PY) code/plotting/fig4_lg_cost.py

figure5:
	$(PY) code/plotting/fig5_hd_cost.py

figure6:
	$(PY) code/plotting/fig6_image_metrics.py

figure7:
	$(PY) code/plotting/fig7_samples.py
