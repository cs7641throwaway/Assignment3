import pandas as pd
import graphviz
from joblib import dump, load
import sklearn.model_selection as ms
import matplotlib.pyplot as plt
import numpy as np
import os
import data_funcs
from sklearn.manifold import TSNE


def plot_fmnist_image(arr, label, title):
    if label == 0:
        item = "T-shirt"
    elif label == 1:
        item = "Trouser"
    elif label == 2:
        item = "Pullover"
    elif label == 3:
        item = "Dress"
    elif label == 4:
        item = "Coat"
    elif label == 5:
        item = "Sandal"
    elif label == 6:
        item = "Shirt"
    elif label == 7:
        item = "Sneaker"
    elif label == 8:
        item = "Bag"
    elif label == 9:
        item = "Ankle boot"
    else:
        print("ERROR: Label is ", label, "but should be int from 0 to 9")
        return
    file = 'plots/'+title+'_'+item+'.png'
    two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
    plt.imshow(two_d, interpolation='nearest')
    plt.title(title+' '+item)
    plt.savefig(file)
    return plt


# https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
def show_PCA(pca):
    fig, axes = plt.subplots(2, 8, figsize=(9, 4),
                             subplot_kw={'xticks': [], 'yticks': []},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(pca.components_[i].reshape(28, 28), cmap='bone')
    fig.savefig('plots/PCA_components.png')
    return


def show_ICA(ica):
    fig, axes = plt.subplots(2, 8, figsize=(9, 4),
                             subplot_kw={'xticks': [], 'yticks': []},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(ica.components_[i].reshape(28, 28), cmap='bone')
    fig.savefig('plots/ICA_components.png')
    return


def show_RP(rp, dataX):
    reconstructed_dataX = data_funcs.reconstruct_arr(rp, dataX)
    fig, axes = plt.subplots(2, 8, figsize=(9, 4),
                             subplot_kw={'xticks': [], 'yticks': []},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        if i >= 8:
            ax.imshow(dataX[i-8].reshape(28, 28), cmap='bone')
        else:
            ax.imshow(reconstructed_dataX[i].reshape(28, 28), cmap='bone')
    fig.savefig('plots/RP_reconstructed.png')
    return


def plot_feature_importance(arr):
    file = 'plots/RF_Feature_Importance.png'
    two_d = np.reshape(arr, (28, 28))
    plt.imshow(two_d, interpolation='nearest')
    plt.title("RF Feature Importance")
    plt.savefig(file)
    return plt


def plot_tsne(data, clusters, title, file):
    plt.figure()
    plt.title(title)
    plt.scatter(data[:,0],data[:,1], c=clusters)
    plt.savefig(file)


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = ms.learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=1, shuffle = True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    #plt.show()
    plt.savefig(title+'.png')
    return plt


def perform_grid_search(estimator, type, dataset, params, trg_X, trg_Y, tst_X, tst_Y, cv=5, n_jobs=-1, train_score=True):
    cv = ms.GridSearchCV(estimator, n_jobs=n_jobs, param_grid= params, refit=True, verbose=2, cv=cv, return_train_score=train_score)
    cv.fit(trg_X, trg_Y)
    test_score = cv.score(tst_X, tst_Y)
    regTable = pd.DataFrame(cv.cv_results_)
    regTable.to_csv('./results/{}_{}_reg.csv'.format(type,dataset),index=False)
    with open('./results/test_results.csv','a') as f:
        f.write('{},{},{},{}\n'.format(type,dataset,test_score,cv.best_params_))


def plot_distribution():
    data = pd.read_hdf('datasets_full.hdf', 'fmnist')
    d = data['Class'].value_counts()
    d.plot.bar()
    plt.ylabel('Frequency')
    plt.xlabel('Class')
    plt.title('FMNIST Class Distribution')
    plt.savefig('FMNIST_class_dist.png')


def plot_complexity(title, param_str, df, ylim=None):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel(param_str)
    plt.ylabel("Score")
    plt.grid()

    # TODO: Need data in form of param, value, training mean, training std, validation mean, validation std
    param_values = df['param_'+param_str]
    train_scores_mean = df['mean_train_score']
    train_scores_std = df['std_train_score']
    test_scores_mean = df['mean_test_score']
    test_scores_std = df['std_test_score']


    plt.fill_between(param_values, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(param_values, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(param_values, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(param_values, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    #plt.show()
    plt.savefig(title+'.png')
    return plt


def get_model_complexity_data(file, param=None, param_value=None):
    df = pd.read_csv(file)
    if param is None:
        return df
    df2 = df[df['param_'+param] == param_value]
    # Open file
    # Read into df
    # Slice to get df
    # Format accordingly
    return df2

def plot_DT_complexity():
    file = 'results/DT_FMNIST_reg.csv'
    df = get_model_complexity_data(file, 'criterion', 'entropy')
    df = df[df['param_max_depth']==20] ; # Slice off max depth
    df = df[df['param_min_samples_leaf']==5] ; # Slice off max depth
    plot_complexity("FMNIST_DT_entropy_max_depth=20_min_samples_leaf=5", 'min_impurity_decrease', df)
    df = get_model_complexity_data(file, 'criterion', 'entropy')
    df = df[df['param_min_impurity_decrease']==0.0005] ; # Slice off max depth
    df = df[df['param_min_samples_leaf']==5] ; # Slice off max depth
    plot_complexity("FMNIST_DT_entropy_min_impurity_decr=0.0005_min_samples_leaf=5", 'max_depth', df)
    df = get_model_complexity_data(file, 'criterion', 'entropy')
    df = df[df['param_min_impurity_decrease']==0.0005] ; # Slice off max depth
    df = df[df['param_max_depth']==20] ; # Slice off max depth
    plot_complexity("FMNIST_DT_entropy_min_impurity_decr=0.0005_max_depth=20", 'min_samples_leaf', df)
    file = 'results/DT_chess_reg.csv'
    df = get_model_complexity_data(file, 'criterion', 'entropy')
    df = df[df['param_max_depth']==100] ; # Slice off max depth
    df = df[df['param_min_samples_leaf']==1] ; # Slice off max depth
    plot_complexity("chess_DT_entropy_max_depth=100_min_samples_leaf=1", 'min_impurity_decrease', df)
    df = get_model_complexity_data(file, 'criterion', 'entropy')
    df = df[df['param_min_impurity_decrease']==0.0005] ; # Slice off max depth
    df = df[df['param_min_samples_leaf']==1] ; # Slice off max depth
    plot_complexity("chess_DT_entropy_min_impurity_decr=0.0005_min_samples_leaf=1", 'max_depth', df)
    df = get_model_complexity_data(file, 'criterion', 'entropy')
    df = df[df['param_min_impurity_decrease']==0.0005] ; # Slice off max depth
    df = df[df['param_max_depth']==100] ; # Slice off max depth
    plot_complexity("chess_DT_entropy_min_impurity_decr=0.0005_max_depth=100", 'min_samples_leaf', df)

def plot_SVM_complexity():
    file = 'results/SVM_Linear_chess_reg.csv'
    df = get_model_complexity_data(file, 'penalty', 'l1')
    plot_complexity("chess_SVM_Linear_alpha", 'alpha', df)
    file = 'results/SVM_RBF_chess_reg.csv'
    df = get_model_complexity_data(file)
    plot_complexity("chess_SVM_RBF_C", 'C', df)
    file = 'results/SVM_Linear_FMNIST_reg.csv'
    df = get_model_complexity_data(file, 'penalty', 'l1')
    plot_complexity("FMNIST_SVM_Linear_alpha", 'alpha', df)
    file = 'results/SVM_RBF_FMNIST_reg.csv'
    df = get_model_complexity_data(file)
    plot_complexity("FMNIST_SVM_RBF_C", 'C', df)

def plot_kNN_complexity():
    file = 'results/kNN_chess_reg.csv'
    df = get_model_complexity_data(file, 'weights', 'uniform')
    plot_complexity("chess_kNN_uniform_n_neighbors", 'n_neighbors', df)
    df = get_model_complexity_data(file, 'weights', 'distance')
    plot_complexity("chess_kNN_distance_n_neighbors", 'n_neighbors', df)
    file = 'results/kNN_FMNIST_reg.csv'
    df = get_model_complexity_data(file, 'weights', 'uniform')
    plot_complexity("FMNIST_kNN_uniform_n_neighbors", 'n_neighbors', df)
    df = get_model_complexity_data(file, 'weights', 'distance')
    plot_complexity("FMNIST_kNN_distance_n_neighbors", 'n_neighbors', df)

def plot_NN_complexity():
    file = 'results/NN_chess_reg.csv'
    df = get_model_complexity_data(file, 'hidden_layer_sizes', "(50, 10)")
    plot_complexity("chess_NN_relu_hidden_layers=(50,10)_tol", 'tol', df)
    file = 'results/NN_FMNIST_reg.csv'
    df = get_model_complexity_data(file, 'tol', 10**-3)
    plot_complexity("FMNIST_NN_tol_10e-3_hidden_layers", 'hidden_layer_sizes', df)
    df = get_model_complexity_data(file, 'hidden_layer_sizes', "(50,)")
    plot_complexity("FMNIST_NN_hidden_layers_50_tol", 'tol', df)

def plot_NN_iteration_curve():
    file = 'results/NN_iteration_curve_chess_reg.csv'
    df = get_model_complexity_data(file)
    plot_complexity("chess_NN_iteration_curve", 'max_iter', df)
    file = 'results/NN_iteration_curve_FMNIST_reg.csv'
    df = get_model_complexity_data(file)
    plot_complexity("FMNIST_NN_iteration_curve", 'max_iter', df)

def plot_boosting_complexity():
    file = 'results/boosting_chess_reg.csv'
    df = get_model_complexity_data(file, 'base_estimator__splitter', 'random')
    df2 = df[df['param_n_estimators'] == 200]
    plot_complexity('chess_boosting_splitter_random_n_estimators_200_max_depth', 'base_estimator__max_depth', df2)
    df = get_model_complexity_data(file, 'base_estimator__splitter', 'random')
    df2 = df[df['param_base_estimator__max_depth'] == 5]
    plot_complexity('chess_boosting_splitter_random_max_depth_5_n_estimators', 'n_estimators', df2)
    df = get_model_complexity_data(file, 'base_estimator__splitter', 'best')
    df2 = df[df['param_n_estimators'] == 200]
    plot_complexity('chess_boosting_splitter_best_n_estimators_200_max_depth', 'base_estimator__max_depth', df2)
    df = get_model_complexity_data(file, 'base_estimator__splitter', 'best')
    df2 = df[df['param_base_estimator__max_depth'] == 5]
    plot_complexity('chess_boosting_splitter_best_max_depth_5_n_estimators', 'n_estimators', df2)
    file = 'results/boosting_FMNIST_reg.csv'
    df = get_model_complexity_data(file, 'n_estimators', 100)
    plot_complexity('FMNIST_boosting_splitter_random_100_estimators_max_depth', 'base_estimator__max_depth', df)
    df = get_model_complexity_data(file, 'base_estimator__max_depth', 20)
    plot_complexity('FMNIST_boosting_splitter_random_max_depth_20_n_estimators', 'n_estimators', df)
    df = get_model_complexity_data(file, 'base_estimator__max_depth', 10)
    plot_complexity('FMNIST_boosting_splitter_random_max_depth_10_n_estimators', 'n_estimators', df)
    file = 'results/boosting_FMNIST_best_reg.csv'
    df = get_model_complexity_data(file, 'n_estimators', 100)
    plot_complexity('FMNIST_boosting_splitter_best_100_estimators_max_depth', 'base_estimator__max_depth', df)
    df = get_model_complexity_data(file, 'base_estimator__max_depth', 20)
    plot_complexity('FMNIST_boosting_splitter_best_max_depth_20_n_estimators', 'n_estimators', df)
    df = get_model_complexity_data(file, 'base_estimator__max_depth', 10)
    plot_complexity('FMNIST_boosting_splitter_best_max_depth_10_n_estimators', 'n_estimators', df)

# Plots y vs. k for dataset
def plot_k_selection(title, dataset, y, ylabel, dim_red=None):
    if dim_red is None:
        file = 'results/kmeans_clusters_'+dataset+'.csv'
    else:
        file = 'results/'+dim_red+'_kmeans_clusters_'+dataset+'.csv'
    print ("Will open file: ", file, " from directory:", os.getcwd())
    df = pd.read_csv(file)
    plt.figure()
    plt.title(title)
    plt.xlabel("k")
    plt.ylabel(ylabel)
    plt.grid()

    plt.plot(df["k"], df[y], 'o-', color="g")
    #plt.show()
    plt.savefig('plots/'+title+'.png')
    return plt


# Plots y vs. n_components for dataset
def plot_em_cluster_selection(title, dataset, y, ylabel, dim_red=None):
    if dim_red is None:
        file = 'results/em_clusters_'+dataset+'.csv'
    else:
        file = 'results/'+dim_red+'_em_clusters_'+dataset+'.csv'
    print("Will open file: ", file, " from directory:", os.getcwd())
    df = pd.read_csv(file)
    plt.figure()
    plt.title(title)
    plt.xlabel("n_components")
    plt.ylabel(ylabel)
    plt.grid()

    plt.plot(df["n_components"], df[y], 'o-', color="g")
    #plt.show()
    plt.savefig('plots/'+title+'.png')
    return plt

def plot_all_em_dim_red_selections(dim_red):
    plot_em_cluster_selection("Chess "+dim_red+" BIC vs. n_components", "chess", "bic", "BIC", dim_red)
    plot_em_cluster_selection("Chess "+dim_red+" AIC vs. n_components", "chess", "aic", "AIC", dim_red)
    plot_em_cluster_selection("Chess "+dim_red+" Score vs. n_components", "chess", "score", "Score", dim_red)
    plot_em_cluster_selection("FMNIST "+dim_red+" BIC vs. n_components", "fmnist", "bic", "BIC", dim_red)
    plot_em_cluster_selection("FMNIST "+dim_red+" AIC vs. n_components", "fmnist", "aic", "AIC", dim_red)
    plot_em_cluster_selection("FMNIST "+dim_red+" Score vs. n_components", "fmnist", "score", "Score", dim_red)


def plot_all_em_selections():
    plot_em_cluster_selection("Chess BIC vs. n_components", "chess", "bic", "BIC")
    plot_em_cluster_selection("Chess AIC vs. n_components", "chess", "aic", "AIC")
    plot_em_cluster_selection("Chess Score vs. n_components", "chess", "score", "Score")
    plot_em_cluster_selection("FMNIST BIC vs. n_components", "fmnist", "bic", "BIC")
    plot_em_cluster_selection("FMNIST AIC vs. n_components", "fmnist", "aic", "AIC")
    plot_em_cluster_selection("FMNIST Score vs. n_components", "fmnist", "score", "Score")
    # Dim red then cluster
    plot_all_em_dim_red_selections("PCA")
    plot_all_em_dim_red_selections("ICA")
    plot_all_em_dim_red_selections("RP")
    plot_all_em_dim_red_selections("RF")


def plot_all_k_selections():
    plot_k_selection("Chess Silhouette Average vs. K", "chess", "sil_avg", "Silhouette Average")
    plot_k_selection("Chess Sum of Squared Distances vs. K", "chess", "ssd", "Sum of Squared Distances")
    plot_k_selection("FMNIST Silhouette Average vs. K", "fmnist", "sil_avg", "Silhouette Average")
    plot_k_selection("FMNIST Sum of Squared Distances vs. K", "fmnist", "ssd", "Sum of Squared Distances")
    # After PCA
    plot_k_selection("Chess PCA Silhouette Average vs. K", "chess", "sil_avg", "Silhouette Average", "PCA")
    plot_k_selection("Chess PCA Sum of Squared Distances vs. K", "chess", "ssd", "Sum of Squared Distances", "PCA")
    plot_k_selection("FMNIST PCA Silhouette Average vs. K", "fmnist", "sil_avg", "Silhouette Average", "PCA")
    plot_k_selection("FMNIST PCA Sum of Squared Distances vs. K", "fmnist", "ssd", "Sum of Squared Distances", "PCA")
    # After ICA
    plot_k_selection("Chess ICA Silhouette Average vs. K", "chess", "sil_avg", "Silhouette Average", "ICA")
    plot_k_selection("Chess ICA Sum of Squared Distances vs. K", "chess", "ssd", "Sum of Squared Distances", "ICA")
    plot_k_selection("FMNIST ICA Silhouette Average vs. K", "fmnist", "sil_avg", "Silhouette Average", "ICA")
    plot_k_selection("FMNIST ICA Sum of Squared Distances vs. K", "fmnist", "ssd", "Sum of Squared Distances", "ICA")
    # After RP
    plot_k_selection("Chess RP Silhouette Average vs. K", "chess", "sil_avg", "Silhouette Average", "RP")
    plot_k_selection("Chess RP Sum of Squared Distances vs. K", "chess", "ssd", "Sum of Squared Distances", "RP")
    plot_k_selection("FMNIST RP Silhouette Average vs. K", "fmnist", "sil_avg", "Silhouette Average", "RP")
    plot_k_selection("FMNIST RP Sum of Squared Distances vs. K", "fmnist", "ssd", "Sum of Squared Distances", "RP")
    # After RF
    plot_k_selection("Chess RF Silhouette Average vs. K", "chess", "sil_avg", "Silhouette Average", "RF")
    plot_k_selection("Chess RF Sum of Squared Distances vs. K", "chess", "ssd", "Sum of Squared Distances", "RF")
    plot_k_selection("FMNIST RF Silhouette Average vs. K", "fmnist", "sil_avg", "Silhouette Average", "RF")
    plot_k_selection("FMNIST RF Sum of Squared Distances vs. K", "fmnist", "ssd", "Sum of Squared Distances", "RF")


def plot_pca_selection(title, dataset):
    file = 'results/pca_'+dataset+'.csv'
    df = pd.read_csv(file)
    df['cum_evr'] = df['explained_variance_ratio'].cumsum()
    plt.figure()
    plt.title(title)
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.grid()
    plt.plot(df['feature'], df["cum_evr"], 'o-', color='g')
    plt.savefig('plots/'+title+'.png')
    return plt


def plot_all_pca_selections():
    plot_pca_selection("Chess PCA Number of Components Selection", "chess")
    plot_pca_selection("FMNIST PCA Number of Components Selection", "fmnist")


def plot_ica_selection(title, dataset):
    file = 'results/ica_'+dataset+'.csv'
    df = pd.read_csv(file)
    plt.figure()
    plt.title(title)
    plt.xlabel("Number of Components")
    plt.ylabel("Average Kurtosis")
    plt.grid()
    plt.plot(df['n_components'], df["avg_kurtosis"], 'o-', color='g')
    plt.savefig('plots/'+title+'.png')
    return plt


def plot_all_ica_selections():
    plot_ica_selection("Chess ICA Number of Components Selection", "chess")
    plot_ica_selection("FMNIST ICA Number of Components Selection", "fmnist")


def plot_srp_selection(title, dataset):
    file = 'results/srp_'+dataset+'.csv'
    df = pd.read_csv(file)
    plt.figure()
    plt.title(title)
    plt.xlabel("Number of Random Projections")
    plt.ylabel("Reconstruction Error")
    plt.grid()
    plt.fill_between(df['n_components'], df['reconstruction_error_mean'] - df['reconstruction_error_sigma'],
                     df['reconstruction_error_mean'] + df['reconstruction_error_sigma'], alpha=0.1,
                     color="g")
    plt.plot(df['n_components'], df['reconstruction_error_mean'], 'o-', color="g",
             label="Cross-validation score")
    plt.savefig('plots/'+title+'.png')
    return plt


def plot_all_srp_selections():
    plot_srp_selection("Chess SRP Number of Random Projections Selection", "chess")
    plot_srp_selection("FMNIST SRP Number of Random Projections Selection", "fmnist")


def plot_rf_selection(title, dataset):
    file = 'results/rf_'+dataset+'.csv'
    df = pd.read_csv(file)
    df['cum_fi'] = df['feature_importance'].cumsum()
    plt.figure()
    plt.title(title)
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Feature Importance")
    plt.grid()
    plt.plot(df['feature'], df["cum_fi"], 'o-', color='g')
    plt.savefig('plots/'+title+'.png')
    return plt


def plot_all_rf_selections():
    plot_rf_selection("Chess RF Number of Components Selection", "chess")
    plot_rf_selection("FMNIST RF Number of Components Selection", "fmnist")

# Part 1: Selection (also includes part 3 dr then cluster)
#plot_all_k_selections()
#plot_all_em_selections()

# Part 1: Validation

# Part 2: Selection
#plot_all_pca_selections()
#plot_all_ica_selections()
#plot_all_srp_selections()
#plot_all_rf_selections()

# Part 3: Clustering after DR
# Rerun part 1 graphs (includes DR)

# Part 4: NN after DR
# No graphs, just tables

# Part 5: NN after Clustering
# No graphs, just tables

