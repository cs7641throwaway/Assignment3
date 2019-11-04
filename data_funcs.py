import pandas as pd
import numpy as np
import math
import scipy as sp
from sklearn import metrics
import sklearn.model_selection as ms
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from joblib import dump, load
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from sklearn.metrics import silhouette_samples, silhouette_score, normalized_mutual_info_score
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
from scipy.linalg import pinv
import scipy.sparse
import time
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import plot


def best_comp_count(dataset, dim_red):
    if dataset is "chess":
        if dim_red is "PCA":
            return 35
        if dim_red is "ICA":
            return 15
        if dim_red is "RP":
            return 60
        if dim_red is "RF":
            return 50
    if dataset is "fmnist":
        if dim_red is "PCA":
            return 300
        if dim_red is "ICA":
            return 150
        if dim_red is "RP":
            return 700
        if dim_red is "RF":
            return 600
    return 0


def best_cluster_count(dataset, type):
    if dataset is "chess":
        if type is "km":
            return 50
        if type is "em":
            return 5
    if dataset is "fmnist":
        if type is "km":
            return 50
        if type is "em":
            return 10
    return 0


def get_data(name, data_prop, test_prop):
    if name == "fmnist":
        data = pd.read_hdf('datasets_full.hdf', 'fmnist')
        dataX = data.drop('Class', 1).copy().values
        dataX = StandardScaler().fit_transform(dataX)
        dataY = data['Class'].copy().values
    elif name == "chess":
        data = pd.read_hdf('datasets_full.hdf', 'chess')
        dataX = data.drop('win', 1).copy().values
        dataX = StandardScaler().fit_transform(dataX)
        dataY = data['win'].copy().values
    else:
        print("ERROR: Unexpected value of name: ", name)
        return

    if data_prop == 1:
        data_dsX = dataX
        data_dsY = dataY
    else:
        data_dsX, data_dummyX, data_dsY, data_dummyY = ms.train_test_split(dataX, dataY, test_size = 1.0 - data_prop,
                                                                               random_state=0, stratify=dataY)
    if test_prop == 0:
        data_trainX = data_dsX
        data_trainY = data_dsY
        data_testX = None
        data_testY = None
    else:
        data_trainX, data_testX, data_trainY, data_testY = ms.train_test_split(data_dsX, data_dsY, test_size = test_prop,
                                                                           random_state=0, stratify=data_dsY)
    return data_trainX, data_testX, data_trainY, data_testY

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


def perform_grid_search(estimator, type, dataset, params, trg_X, trg_Y, tst_X, tst_Y, cv=5, n_jobs=-1):
    cv = ms.GridSearchCV(estimator, n_jobs=n_jobs, param_grid= params, refit=True, verbose=2, cv=cv)
    cv.fit(trg_X, trg_Y)
    test_score = cv.score(tst_X, tst_Y)
    regTable = pd.DataFrame(cv.cv_results_)
    regTable.to_csv('./results/{}_{}_reg.csv'.format(type,dataset),index=False)
    with open('./results/test_results.csv','a') as f:
        f.write('{},{},{},{}\n'.format(type,dataset,test_score,cv.best_params_))

def write_best_results(type, dataset, train_score, test_score, time, clf, save=False):
    params = clf.get_params()
    if save:
        dump(clf, type+'_'+dataset+'.joblib')
    with open('./results/best_results.csv','a') as f:
        f.write('{},{},{},{},{}\n'.format(type,dataset,train_score, test_score, time, params))


# https://datascience.stackexchange.com/questions/6683/feature-selection-using-feature-importances-in-random-forests-with-scikit-learn
def selectKImportance(model, X, k=5):
    return X[:, model.feature_importances_.argsort()[::-1][:k]]


def sweep_k(clusters, dataset, data, data_labels, dim_red=None):
    if dim_red is None:
        file = './results/kmeans_clusters_' + dataset + '.csv'
    else:
        file = './results/'+dim_red+'_kmeans_clusters_' + dataset + '.csv'
    with open(file, 'w') as f:
        f.write('{},{},{},{},{},{}\n'.format("k", "sil_avg", "ssd", "norm_mutual_info", "purity", "fit_time"))
    if dim_red is not None:
        comp_count = best_comp_count(dataset, dim_red)
        if dim_red is "PCA":
            dim_red = PCA(n_components=comp_count, random_state=0)
            transformed_data = dim_red.fit_transform(data)
        if dim_red is "ICA":
            dim_red = FastICA(n_components=comp_count, random_state=0)
            transformed_data = dim_red.fit_transform(data)
        if dim_red is "RP":
            dim_red = SparseRandomProjection(n_components=comp_count, random_state=0)
            transformed_data = dim_red.fit_transform(data)
        if dim_red is "RF":
            dim_red = RandomForestClassifier(n_estimators=comp_count, random_state=0, n_jobs=-1).fit(data, data_labels)
            transformed_data = selectKImportance(dim_red, data, comp_count)
    else:
        transformed_data = data
    for cluster in clusters:
        if dim_red is not None and comp_count < cluster:
            continue
        start = time.time()
        print("Transformed data.  Orig shape: ", data.shape, " new shape: ", transformed_data.shape)
        km = KMeans(n_clusters=cluster, random_state=0)
        km.fit(transformed_data)
        end = time.time()
        elapsed = end - start
        print("Fit clusters of: ", cluster, " on ", dataset, " in ", elapsed)
        ssd = km.inertia_
        print("For clusters of: ", cluster, " on ", dataset, " data set, got sum of squared error of: ", ssd)
        data_cluster_labels = km.predict(transformed_data)
        silhouette_avg = silhouette_score(transformed_data, data_cluster_labels)
        nmi = normalized_mutual_info_score(data_labels, data_cluster_labels)
        purity = purity_score(data_labels, data_cluster_labels)
        print("For clusters of ", cluster, " on ", dataset, " data set, got silhouette average of :", silhouette_avg)
        print("\tValidation: got nmi of:", nmi, " and purity of: ", purity)
        with open(file, 'a') as f:
            f.write('{},{},{},{},{},{}\n'.format(cluster, silhouette_avg, ssd, nmi, purity, elapsed))
    return


def em_sweep_clusters(clusters, dataset, data, data_labels, dim_red=None):
    if dim_red is None:
        file = './results/em_clusters_' + dataset + '.csv'
    else:
        file = './results/'+dim_red+'_em_clusters_' + dataset + '.csv'
    if dim_red is not None:
        comp_count = best_comp_count(dataset, dim_red)
        if dim_red is "PCA":
            dim_red = PCA(n_components=comp_count, random_state=0)
            transformed_data = dim_red.fit_transform(data)
        if dim_red is "ICA":
            dim_red = FastICA(n_components=comp_count, random_state=0)
            transformed_data = dim_red.fit_transform(data)
        if dim_red is "RP":
            dim_red = SparseRandomProjection(n_components=comp_count, random_state=0)
            transformed_data = dim_red.fit_transform(data)
        if dim_red is "RF":
            dim_red = RandomForestClassifier(n_estimators=comp_count, random_state=0, n_jobs=-1).fit(data, data_labels)
            transformed_data = selectKImportance(dim_red, data, comp_count)
    else:
        transformed_data = data
    print("Transformed data.  Orig shape: ", data.shape, " new shape: ", transformed_data.shape)
    with open(file, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format("n_components", "score", "bic", "aic", "norm_mutual_info", "purity", "fit_time"))
    for cluster in clusters:
        if dim_red is not None and comp_count < cluster:
            continue
        start = time.time()
        em = GaussianMixture(n_components=cluster, random_state=0).fit(transformed_data)
        end = time.time()
        elapsed = end - start
        print("Fit clusters of: ", cluster, " on ", dataset, " in ", elapsed)
        aic = em.aic(transformed_data)
        print("For clusters of: ", cluster, " on ", dataset, " data set, got aic of: ", aic)
        bic = em.bic(transformed_data)
        print("For clusters of: ", cluster, " on ", dataset, " data set, got bic of: ", bic)
        score = em.score(transformed_data)
        print("For clusters of ", cluster, " on ", dataset, " data set, got score of :", score)
        data_cluster_labels = em.predict(transformed_data)
        nmi = normalized_mutual_info_score(data_labels, data_cluster_labels)
        purity = purity_score(data_labels, data_cluster_labels)
        print("\tValidation: got nmi of:", nmi, " and purity of: ", purity)
        with open(file, 'a') as f:
            f.write('{},{},{},{},{},{},{}\n'.format(cluster, score, bic, aic, nmi, purity, elapsed))
    return


# Stack overflow
# https://stackoverflow.com/questions/15450192/fastest-way-to-compute-entropy-in-python
def __calc_entropy(labels):
    value, counts = np.unique(labels, return_counts=True)
    probs = counts/len(labels)

    entropy = 0.0
    # Compute entropy
    for p in probs:
        entropy -= p * math.log(p, math.e)
    return entropy


# For PCA, we can simply run once and dump the corresponding variances
def run_pca(n_components, dataset, data):
    pca = PCA(n_components=n_components, random_state=0)
    pca.fit(data)
    file = './results/pca_'+dataset+'.csv'
    df = pd.DataFrame(pca.explained_variance_ratio_, columns=['explained_variance_ratio'])
    df.to_csv(file, index_label='feature')
    return pca


def run_ica(n_components, dataset, data):
    file = './results/ica_' + dataset + '.csv'
    with open(file, 'w') as f:
        f.write('{},{},{}\n'.format("n_components", "avg_kurtosis", "runtime"))
    for n in n_components:
        start = time.time()
        ica = FastICA(n_components=n, random_state=0).fit_transform(data)
        end = time.time()
        elapsed = end - start
        ica_df = pd.DataFrame(ica)
        avg_kurt = ica_df.kurt(axis=0).abs().mean()
        print("For ICA n_components of: ", n, " on ", dataset, " data set got avg kurtosis of: ", avg_kurt, " in time: ", elapsed)
        with open(file, 'a') as f:
            f.write('{},{},{}\n'.format(n, avg_kurt, elapsed))
    return


def reconstruction_error(rp, orig_x):
    reconstructed = reconstruct_arr(rp, orig_x)
    errors = np.square(orig_x-reconstructed)
    return np.mean(errors)


def reconstruct_arr(rp, orig_x):
    components = rp.components_
    if scipy.sparse.issparse(components):
        components = components.todense()
    pseudo_inverse_components = pinv(components)
    reconstructed = ((pseudo_inverse_components@components)@(orig_x.T)).T  # Product of psuedo inverse with itself, multipled by the transpose of x, then transpose of that
    return reconstructed


def run_rp(n_projections, dataset, data):
    file = './results/srp_' + dataset + '.csv'
    with open(file, 'w') as f:
        f.write('{},{},{},{}\n'.format("n_components", "reconstruction_error_mean", "reconstruction_error_sigma", "runtime"))
    for n in n_projections:
        errors = []
        for i in range(1,5):
            start = time.time()
            srp = SparseRandomProjection(n_components=n)
            # srp = GaussianRandomProjection(n_components=n)
            trans_x = srp.fit_transform(data)
            end = time.time()
            elapsed = end - start
            error = reconstruction_error(srp, data)
            errors.append(error)
            # print("For SRP n_components of: ", n, " on ", dataset, " data set got reconstruction error of: ", error, " in time: ", elapsed)
        error_mean = np.mean(errors)
        error_sigma = np.std(errors)
        print("For SRP n_components of: ", n, " on ", dataset, " data set got mean reconstruction error of: ", error_mean, " with sigma of: ", error_sigma)
        with open(file, 'a') as f:
            f.write('{},{},{},{}\n'.format(n, error_mean, error_sigma, elapsed))
    return

# From https://stackoverflow.com/questions/34047540/python-clustering-purity-metric
def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def run_lda(n_components, dataset, data, labels):
    params = {'n_components':n_components, 'solver':['lsqr', 'svd']}
    cv = ms.GridSearchCV(LinearDiscriminantAnalysis(), param_grid=params, n_jobs=-1, verbose=2, cv=5)
    cv.fit(data, labels)
    regTable = pd.DataFrame(cv.cv_results_)
    regTable.to_csv('./results/lda_{}_reg.csv'.format(dataset),index=False)
    return


def run_rf(dataset, data, labels):
    file = './results/rf_'+dataset+'.csv'
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
    features = np.sort(rf.fit(data, labels).feature_importances_)[::-1]
    df = pd.DataFrame(features, columns=['feature_importance'])
    df.to_csv(file, index_label='feature')
    return rf


def run_dim_red_nn(dataset, dim_red, trgX, trgY, tstX, tstY):
    n_comp = best_comp_count(dataset, dim_red)
    if dataset is "chess":
        clf = MLPClassifier(hidden_layer_sizes=(50, 10), activation='relu')
    elif dataset is "fmnist":
        clf = MLPClassifier(hidden_layer_sizes=(50,), activation='relu', tol=0.002)
    else:
        print("Error: dataset is ", dataset, " but must be either chess or fmnist")
        return
    if dim_red is "PCA":
        dr = PCA(n_components=n_comp, random_state=0)
        dr_trgX = dr.fit_transform(trgX)
        dr_tstX = dr.transform(tstX)
    elif dim_red is "ICA":
        dr = FastICA(n_components=n_comp, random_state=0)
        dr_trgX = dr.fit_transform(trgX)
        dr_tstX = dr.transform(tstX)
    elif dim_red is "RP":
        dr = SparseRandomProjection(n_components=n_comp, random_state=0)
        dr_trgX = dr.fit_transform(trgX)
        dr_tstX = dr.transform(tstX)
    elif dim_red is "RF":
        dr = RandomForestClassifier(n_estimators=n_comp, random_state=0, n_jobs=-1).fit(trgX, trgY)
        dr_trgX = selectKImportance(dr, trgX, n_comp)
        dr_tstX = selectKImportance(dr, tstX, n_comp)
    else:
        print("Error: dim_red is ", dim_red, " but must be PCA, ICA, RP, or RF")
        return
    start = time.time()
    clf.fit(dr_trgX, trgY)
    end = time.time()
    elapsed = end-start
    train_score = clf.score(dr_trgX, trgY)
    test_score = clf.score(dr_tstX, tstY)
    print("For dataset: ", dataset, " got train_score: ", train_score, " and test_score: ", test_score, " in time", elapsed)
    write_best_results('NN_'+dim_red, dataset, train_score, test_score, elapsed, clf)
    return


# If replace is false, will append features; if replace is true, will replace features
def run_cluster_nn(dataset, cluster_method, trgX, trgY, tstX, tstY, replace=False):
    n_cluster = best_cluster_count(dataset, cluster_method)
    if dataset is "chess":
        clf = MLPClassifier(hidden_layer_sizes=(50, 10), activation='relu')
    elif dataset is "fmnist":
        clf = MLPClassifier(hidden_layer_sizes=(50,), activation='relu', tol = 0.002)
    else:
        print("Error: dataset is ", dataset, " but must be either chess or fmnist")
        return
    if cluster_method is "km":
        cm = KMeans(n_clusters=n_cluster, random_state=0)
        cm.fit(trgX)
        trg_clusters = cm.transform(trgX)
        tst_clusters = cm.transform(tstX)
    elif cluster_method is "em":
        cm = GaussianMixture(n_components=n_cluster, random_state=0)
        cm.fit(trgX)
        trg_clusters = cm.predict_proba(trgX)
        tst_clusters = cm.predict_proba(tstX)
    else:
        print("Error: cluster_method is ", cluster_method, " but must be km or em")
        return

    # Calculate distances from cluster centers

    if replace:
        # Replace features
        trgX = trg_clusters
        tstX = tst_clusters
    else:
        # Append features
        trgX = np.concatenate((trgX, trg_clusters), axis=1)
        tstX = np.concatenate((tstX, tst_clusters), axis=1)

    start = time.time()
    clf.fit(trgX, trgY)
    end = time.time()
    elapsed = end-start
    train_score = clf.score(trgX, trgY)
    test_score = clf.score(tstX, tstY)
    print("For dataset: ", dataset, " got train_score: ", train_score, " and test_score: ", test_score, " in time", elapsed)
    if replace:
        tag = 'NN_'+cluster_method+'_replace'
    else:
        tag = 'NN_'+cluster_method
    write_best_results(tag, dataset, train_score, test_score, elapsed, clf)
    return


