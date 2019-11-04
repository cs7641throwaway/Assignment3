from sklearn.neural_network import MLPClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import data_funcs
import random
import scipy.sparse
import scipy.linalg
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import plot


random.seed(0)
run_chess = False
run_fmnist = True

run_rf_sweep = False
run_nn = False
plot_fmnist_importance = True

data_prop = 1.0
test_prop = 0.2

if run_fmnist:
	fmnist_trgX, fmnist_tstX, fmnist_trgY, fmnist_tstY =  data_funcs.get_data('fmnist', data_prop, test_prop)

if run_chess:
	chess_trgX, chess_tstX, chess_trgY, chess_tstY = data_funcs.get_data('chess', data_prop, test_prop)

if run_rf_sweep:
	if run_chess:
		data_funcs.run_rf("chess", chess_trgX, chess_trgY)

	if run_fmnist:
		data_funcs.run_rf("fmnist", fmnist_trgX, fmnist_trgY)

if run_nn:
	if run_chess:
		data_funcs.run_dim_red_nn("chess", "RF", chess_trgX, chess_trgY, chess_tstX, chess_tstY)

	if run_fmnist:
		data_funcs.run_dim_red_nn("fmnist", "RF", fmnist_trgX, fmnist_trgY, fmnist_tstX, fmnist_tstY)

if plot_fmnist_importance:
	# Get features
	dim_red = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1).fit(fmnist_trgX, fmnist_trgY)
	plot.plot_feature_importance(dim_red.feature_importances_)
	# Plot 28x28 but with each pixel being its importance instead of value
