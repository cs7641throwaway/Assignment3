from sklearn.neural_network import MLPClassifier
import pandas as pd
import graphviz
from joblib import dump, load
import sklearn.model_selection as ms
import matplotlib.pyplot as plt
import numpy as np
import data_funcs
import random
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import FastICA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import plot


random.seed(0)
run_chess = True
run_fmnist = True

run_ica_sweep = False
run_nn = False
run_image = False
run_component_plots = True

data_prop = 1.0
test_prop = 0.2

fmnist_trgX, fmnist_tstX, fmnist_trgY, fmnist_tstY =  data_funcs.get_data('fmnist', data_prop, test_prop)
chess_trgX, chess_tstX, chess_trgY, chess_tstY = data_funcs.get_data('chess', data_prop, test_prop)

n_components_chess = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
n_components_fmnist = [2, 5, 10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750]

if run_ica_sweep:
	if run_chess:
		ica_chess = data_funcs.run_ica(n_components_chess, "chess", chess_trgX)

	if run_fmnist:
		ica_fmnist = data_funcs.run_ica(n_components_fmnist, "fmnist", fmnist_trgX)


# Really good tutorial on pipelines here: https://scikit-learn.org/stable/modules/compose.html#pipeline
if run_nn:
	if run_chess:
		data_funcs.run_dim_red_nn("chess", "ICA", chess_trgX, chess_trgY, chess_tstX, chess_tstY)

	if run_fmnist:
		data_funcs.run_dim_red_nn("fmnist", "ICA", fmnist_trgX, fmnist_trgY, fmnist_tstX, fmnist_tstY)


if run_image:
	dr = FastICA(n_components=700, random_state=0)
	dr.fit(fmnist_trgX)
	dr_trgX = dr.transform(fmnist_trgX)
	fmnist_reconstructedX = dr.inverse_transform(dr_trgX)
	print("Original:")
	print(dr_trgX.shape)
	print("Reconstructed:")
	print(fmnist_reconstructedX.shape)
	plot.plot_fmnist_image(fmnist_trgX[0], fmnist_trgY[0], "Original Image")
	plot.plot_fmnist_image(fmnist_reconstructedX[0], fmnist_trgY[0], "ICA Reconstructed Image")


if run_component_plots:
	ica = FastICA(n_components=data_funcs.best_comp_count("fmnist", "ICA"), random_state=0)
	ica.fit(fmnist_trgX)
	plot.show_ICA(ica)

