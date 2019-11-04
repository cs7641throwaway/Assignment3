from sklearn.neural_network import MLPClassifier
import pandas as pd
import graphviz
from joblib import dump, load
import sklearn.model_selection as ms
import matplotlib.pyplot as plt
import numpy as np
import data_funcs
import random
import scipy.sparse
import scipy.linalg
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.random_projection import SparseRandomProjection
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import plot


random.seed(0)
run_chess = True
run_fmnist = True

run_rp_sweep = True
run_nn = False
run_image = False
run_reconstruction_plots = False

data_prop = 1.0
test_prop = 0.2

if run_fmnist:
	fmnist_trgX, fmnist_tstX, fmnist_trgY, fmnist_tstY =  data_funcs.get_data('fmnist', data_prop, test_prop)

if run_chess:
	chess_trgX, chess_tstX, chess_trgY, chess_tstY = data_funcs.get_data('chess', data_prop, test_prop)

chess_n_components = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
fmnist_n_components = [1, 10, 20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750]

# Note: have to specify n_components because of conservative limit from:
# Dimensionality of the target projection space.
# n_components can be automatically adjusted according to the number of samples in the dataset and the bound
# given by the Johnson-Lindenstrauss lemma. In that case the quality of the embedding is controlled by the eps parameter.
# It should be noted that Johnson-Lindenstrauss lemma can yield very conservative estimated of the required
# number of components as it makes no assumption on the structure of the dataset.

if run_rp_sweep:
	if run_chess:
		data_funcs.run_rp(chess_n_components, "chess", chess_trgX)
	if run_fmnist:
		data_funcs.run_rp(fmnist_n_components, "fmnist", fmnist_trgX)

if run_nn:
	if run_chess:
		data_funcs.run_dim_red_nn("chess", "RP", chess_trgX, chess_trgY, chess_tstX, chess_tstY)

	if run_fmnist:
		data_funcs.run_dim_red_nn("fmnist", "RP", fmnist_trgX, fmnist_trgY, fmnist_tstX, fmnist_tstY)

# Just for visualization
if run_image:
	dr = SparseRandomProjection(n_components=1000, random_state=0)
	dr_trgX = dr.fit_transform(fmnist_trgX)
	fmnist_reconstructedX = dr.inverse_transform(dr_trgX)
	plot.plot_fmnist_image(fmnist_trgX[0], fmnist_trgY[0], "Original Image")
	plot.plot_fmnist_image(fmnist_reconstructedX[0], fmnist_trgY[0], "SRP Reconstructed Image")


if run_reconstruction_plots:
	dr = SparseRandomProjection(n_components=data_funcs.best_comp_count("fmnist", "RP"), random_state=0)
	dr.fit(fmnist_trgX)
	plot.show_RP(dr, fmnist_trgX)
