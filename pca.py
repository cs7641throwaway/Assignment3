from sklearn.neural_network import MLPClassifier
import pandas as pd
import graphviz
from joblib import dump, load
import sklearn.model_selection as ms
import matplotlib.pyplot as plt
import plot
import numpy as np
import data_funcs
import random
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

random.seed(0)
run_chess = False
run_fmnist = True

run_pca_sweep = False
run_nn = False
run_image = False
run_component_plots = True

data_prop = 1.0
test_prop = 0.2

if run_fmnist:
	fmnist_trgX, fmnist_tstX, fmnist_trgY, fmnist_tstY =  data_funcs.get_data('fmnist', data_prop, test_prop)
if run_chess:
	chess_trgX, chess_tstX, chess_trgY, chess_tstY = data_funcs.get_data('chess', data_prop, test_prop)

if run_pca_sweep:
	if run_chess:
		pca_chess = data_funcs.run_pca(None, "chess", chess_trgX)
		print(pca_chess.explained_variance_ratio_)

	if run_fmnist:
		pca_fmnist = data_funcs.run_pca(None, "fmnist", fmnist_trgX)
		print(pca_fmnist.explained_variance_ratio_)

if run_nn:
	if run_chess:
		data_funcs.run_dim_red_nn("chess", "PCA", chess_trgX, chess_trgY, chess_tstX, chess_tstY)

	if run_fmnist:
		data_funcs.run_dim_red_nn("fmnist", "PCA", fmnist_trgX, fmnist_trgY, fmnist_tstX, fmnist_tstY)


# Just for visualization
if run_image:
	plot.plot_fmnist_image(fmnist_trgX[0], fmnist_trgY[0], "Original Image")
	for n in [20, 200, 700]:
		dr = PCA(n_components=n, random_state=0)
		dr_trgX = dr.fit_transform(fmnist_trgX)
		fmnist_reconstructedX = dr.inverse_transform(dr_trgX)
		plot.plot_fmnist_image(fmnist_reconstructedX[0], fmnist_trgY[0], "PCA No Scale Reconstructed Image n_comp = "+str(n))

if run_component_plots:
	pca = PCA(n_components=data_funcs.best_comp_count("fmnist", "PCA"), random_state=0)
	pca.fit(fmnist_trgX)
	plot.show_PCA(pca)


