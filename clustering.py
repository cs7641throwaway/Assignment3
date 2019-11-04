from sklearn.neural_network import MLPClassifier
import pandas as pd
import graphviz
from joblib import dump, load
import sklearn.model_selection as ms
import matplotlib.pyplot as plt
import numpy as np
import data_funcs
import random
import data_funcs
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
import plot

random.seed(0)
run_fmnist = False
run_chess = True

# Runs kmeans/em
run_kmeans = False
run_em = True

run_sweeps = False
run_nn = False
run_cluster_analysis = True

# Used only if run_cluster_analysis; applies best DR to each clustering method to observe impact
best_dr = False


# One hot variables to add PCA/ICA/RP/RF feature selection before kmeans/em (only for sweeps)
dim_red = None
#dim_red = "PCA"
#dim_red = "ICA"
#dim_red = "RP"
#dim_red = "RF"

if dim_red is not None:
	if dim_red is not "ICA" and dim_red is not "RP" and dim_red is not "PCA" and dim_red is not "RF":
		print("ERROR: dim_red must be PCA, ICA, RP, RF, or None")
		exit()

data_prop = 1.0
test_prop = 0.2

if run_fmnist:
	fmnist_trgX, fmnist_tstX, fmnist_trgY, fmnist_tstY =  data_funcs.get_data('fmnist', data_prop, test_prop)
if run_chess:
	chess_trgX, chess_tstX, chess_trgY, chess_tstY = data_funcs.get_data('chess', data_prop, test_prop)

# print(chess_trgX.shape)
# print(chess_tstY)

# 2 or 60 (similar) for k; 10 for EM
clusters_chess = [2, 5, 10, 20, 30, 40, 50, 60, 70]
# ~50ish for k; 20 for EM
clusters_fmnist = [2, 5, 10, 20, 50, 100, 150, 200]

# Selection:
	# Chess
		# PCA: 30
		# ICA: 40
		# RP: 60
		# RF: 50

	# FMNIST
    	# PCA: 500
		# ICA: 200
		# RP: 700
		# RF: 600
if run_sweeps:
	if run_chess:
		if run_kmeans:
			data_funcs.sweep_k(clusters_chess, "chess", chess_trgX, chess_trgY, dim_red)
		if run_em:
			data_funcs.em_sweep_clusters(clusters_chess, "chess", chess_trgX, chess_trgY, dim_red)

	if run_fmnist:
		if run_kmeans:
			data_funcs.sweep_k(clusters_fmnist, "fmnist", fmnist_trgX, fmnist_trgY, dim_red)
		if run_em:
			clusters_fmnist = [2, 5, 10, 20, 50] # Reduced because EM takes forever
			data_funcs.em_sweep_clusters(clusters_fmnist, "fmnist", fmnist_trgX, fmnist_trgY, dim_red)

if run_nn:
	if run_chess:
		if run_kmeans:
			data_funcs.run_cluster_nn("chess", "km", chess_trgX, chess_trgY, chess_tstX, chess_tstY, False)
			data_funcs.run_cluster_nn("chess", "km", chess_trgX, chess_trgY, chess_tstX, chess_tstY, True)
		if run_em:
			data_funcs.run_cluster_nn("chess", "em", chess_trgX, chess_trgY, chess_tstX, chess_tstY, False)
			data_funcs.run_cluster_nn("chess", "em", chess_trgX, chess_trgY, chess_tstX, chess_tstY, True)

	if run_fmnist:
		if run_kmeans:
			data_funcs.run_cluster_nn("fmnist", "km", fmnist_trgX, fmnist_trgY, fmnist_tstX, fmnist_tstY, False)
			data_funcs.run_cluster_nn("fmnist", "km", fmnist_trgX, fmnist_trgY, fmnist_tstX, fmnist_tstY, True)
		if run_em:
			data_funcs.run_cluster_nn("fmnist", "em", fmnist_trgX, fmnist_trgY, fmnist_tstX, fmnist_tstY, False)
			data_funcs.run_cluster_nn("fmnist", "em", fmnist_trgX, fmnist_trgY, fmnist_tstX, fmnist_tstY, True)


if run_cluster_analysis:
	if run_chess:
		tx_data = TSNE(random_state=0).fit_transform(chess_trgX)
		if run_kmeans:
			clusters = data_funcs.best_cluster_count("chess", "km")
			if best_dr:
				comp_count = data_funcs.best_comp_count("chess", "RF")
				dr = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
				dr.fit(chess_trgX, chess_trgY)
				dr_chess_trgX = data_funcs.selectKImportance(dr, chess_trgX, comp_count)
				title_cluster = 'Chess KM Clusters with RF DR'
				file_cluster = 'plots/chess_km_cluster_w_RF_DR.png'
			else:
				dr_chess_trgX = chess_trgX
				title_cluster = 'Chess KM Clusters'
				file_cluster = 'plots/chess_km_cluster.png'
			km = KMeans(n_clusters=clusters, random_state=0).fit(dr_chess_trgX)
			# Plot histogram of cluster purities
			plot.plot_tsne(tx_data, km.predict(dr_chess_trgX), title_cluster, file_cluster)
			plot.plot_tsne(tx_data, chess_trgY, "Chess KM Real Labels", "plots/chess_km_cluster_real_labels.png")
		if run_em:
			clusters = data_funcs.best_cluster_count("chess", "em")
			if best_dr:
				comp_count = data_funcs.best_comp_count("chess", "PCA")
				title_cluster = 'Chess EM Clusters with PCA DR'
				file_cluster = 'plots/chess_em_cluster_w_PCA_DR.png'
				dr = PCA(n_components=comp_count, random_state=0)
				dr_chess_trgX = dr.fit_transform(chess_trgX)
			else:
				dr_chess_trgX = chess_trgX
				title_cluster = 'Chess EM Clusters'
				file_cluster = 'plots/chess_em_cluster.png'
			em = GaussianMixture(n_components=clusters, random_state=0).fit(dr_chess_trgX)
			plot.plot_tsne(tx_data, em.predict(dr_chess_trgX), title_cluster, file_cluster)
			plot.plot_tsne(tx_data, chess_trgY, "Chess EM Real Labels", "plots/chess_em_cluster_real_labels.png")
	if run_fmnist:
		tx_data = TSNE(random_state=0).fit_transform(fmnist_trgX)
		if run_kmeans:
			clusters = data_funcs.best_cluster_count("fmnist", "km")
			if best_dr:
				comp_count = data_funcs.best_comp_count("fmnist", "RF")
				title_cluster = 'FMNIST KM Clusters with RF DR'
				file_cluster = 'plots/fmnist_km_cluster_w_RF_DR.png'
				dim_red = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
				dim_red.fit(fmnist_trgX, fmnist_trgY)
				dr_fmnist_trgX = data_funcs.selectKImportance(dim_red, fmnist_trgX, comp_count)
			else:
				title_cluster = 'FMNIST KM Clusters'
				file_cluster = 'plots/fmnist_em_cluster.png'
				dr_fmnist_trgX = fmnist_trgX
			km = KMeans(n_clusters=clusters, random_state=0).fit(dr_fmnist_trgX)
			plot.plot_tsne(tx_data, km.predict(dr_fmnist_trgX), title_cluster, file_cluster)
			plot.plot_tsne(tx_data, fmnist_trgY, "FMNIST KM Real Labels", "plots/fmnist_km_cluster_real_labels.png")
		if run_em:
			clusters = data_funcs.best_cluster_count("fmnist", "em")
			if best_dr:
				comp_count = data_funcs.best_comp_count("fmnist", "PCA")
				title_cluster = 'FMNIST EM Clusters with PCA DR'
				file_cluster = 'plots/fmnist_em_cluster_w_PCA_DR.png'
				dim_red = PCA(n_components=comp_count, random_state=0)
				dr_fmnist_trgX = dim_red.fit_transform(fmnist_trgX)
			else:
				dr_fmnist_trgX = fmnist_trgX
				title_cluster = 'FMNIST EM Clusters'
				file_cluster = 'plots/fmnist_em_cluster.png'
			em = GaussianMixture(n_components=clusters, random_state=0).fit(dr_fmnist_trgX)
			plot.plot_tsne(tx_data, em.predict(dr_fmnist_trgX), title_cluster, file_cluster)
			plot.plot_tsne(tx_data, fmnist_trgY, "FMNIST EM Real Labels", "plots/fmnist_em_cluster_real_labels.png")


	#chess_score = chess_km.score(chess_trgX)
		#print("KM Clusters: ", cluster, " Chess score: ", chess_score)
		#data_funcs.calc_similarity(chess_km.predict(chess_trgX), chess_trgY)
		#chess_em = GaussianMixture(n_components=cluster, random_state=0).fit(chess_trgX)
		#chess_score = chess_em.score(chess_trgX)
		#print("EM Clusters: ", cluster, " Chess score: ", chess_score)
		#data_funcs.calc_similarity(chess_em.predict(chess_trgX), chess_trgY)
