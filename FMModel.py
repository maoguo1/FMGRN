import numpy as np
import pandas as pd
import time
import scipy.sparse as sp
from fastFM import sgd
from scipy import sparse
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import math
import pymrmr
from sklearn.model_selection import train_test_split


def get_links(VIM, gene_names, regulators, sort=True, file_name=None):
    idx = [i for i, gene in enumerate(gene_names) if gene in regulators]
    pred_edges = [(gene_names[j], gene_names[i], score) for (i, j), score in np.ndenumerate(VIM) if i != j and j in idx]
    pred_edges = pd.DataFrame(pred_edges)
    if sort is True:
        pred_edges.sort_values(2, ascending=False, inplace=True)
    if file_name is None:
        print(pred_edges)
    else:
        pred_edges.to_csv(file_name, sep='\t', header=None, index=None)

 
# ???????????
def prenormal(X_train):
    minf_X = [0.0] * X_train.shape[1]
    maxf_X = [0.0] * X_train.shape[1]
    for f in range(0, X_train.shape[1]):
        maxf_X[f] = X_train[0][f]
        minf_X[f] = X_train[0][f]
        for i in range(1, X_train.shape[0]):
            maxf_X[f] = max(maxf_X[f], X_train[i][f])
            minf_X[f] = min(minf_X[f], X_train[i][f])
        if (maxf_X[f] > minf_X[f]):
            for i in range(0, X_train.shape[0]):
                X_train[i][f] = (X_train[i][f] - minf_X[f]) / (maxf_X[f] - minf_X[f])
        else:
            for i in range(0, X_train.shape[0]):
                X_train[i][f] = 0
        maxf_X[f] = X_train[0][f]
        minf_X[f] = X_train[0][f]
        for i in range(1, X_train.shape[0]):
            maxf_X[f] = max(maxf_X[f], X_train[i][f])
            minf_X[f] = min(minf_X[f], X_train[i][f])
    return minf_X, maxf_X, X_train


# ??????????

def subcode(X_train, b, minf, maxf):
    F = len(X_train[0])
    X_train_o = np.zeros((X_train.shape[0], F * b))

    a = []
    p = []
    for f in range(0, len(X_train[0])):
        fl = []
        a.append([])
        p.append([])
        for l in range(0, X_train.shape[0]):
            fl.append(X_train[l][f])

        a[f], p[f] = np.unique(fl, return_inverse=True)
    for l in range(0, X_train.shape[0]):
        f_en = 0
        for f in range(0, len(X_train[l])):
            if len(a[f]) == 1:
                f_en = f_en
            elif len(a[f]) >= b:
                if (X_train[l][f] - minf[f]) == (maxf[f] - minf[f]):
                    X_train_o[l][f_en + b - 1] = 1
                else:
                    X_train_o[l][f_en + int(float(X_train[l][f] - minf[f]) / (maxf[f] - minf[f]) * b)] = 1
                f_en += b
            else:
                X_train_o[l][f_en + p[f][l]] = 1
                f_en += len(a[f])
    if f_en < F * b:
        X_train_b = np.zeros((len(X_train), f_en))
        for i in range(len(X_train)):
            for j in range(f_en):
                X_train_b[i][j] = X_train_o[i][j]
                X_train_o = X_train_b

    return X_train_o


def FMpredict(data_x, data_y, b,selF1):

    minf_X, maxf_X, train_x1 = prenormal(data_x)
    #last_col = np.array([data_x.shape[0] * [1]])
    #train_x1 = np.c_[train_x1, last_col.T]
    #minf_X.append(1)
    #maxf_X.append(1)
    X_train_o = subcode(train_x1, b, minf_X, maxf_X)
    last_col = np.array([X_train_o.shape[0]*[1]])
    X_train_o = np.c_[X_train_o, last_col.T]
    train_X = sp.csc_matrix(np.array(X_train_o), dtype=np.float64)

    # test_X = sp.csc_matrix(np.array(X_test_o), dtype=np.float64)
    # print('data_y',data_y)
    fm = sgd.FMRegression(n_iter=10000,
                          init_stdev=0.0001, l2_reg_w=0.1, l2_reg_V=10000, rank=10,
                          step_size=0.0001)  # ???

    fm.fit(train_X, data_y)
    print(train_X.shape)
    print('w_bin5', fm.w_)
    return fm.w_, fm.V_,


def estimate_degradation_rates(TS_data, time_points):
    """
    For each gene, the degradation rate is estimated by assuming that the gene expression x(t) follows:
    x(t) =  A exp(-alpha * t) + C_min,
    between the highest and lowest expression values.
    C_min is set to the minimum expression value over all genes and all samples.
    The function is available at the study named dynGENIE3.
    Huynh-Thu, V., Geurts, P. dynGENIE3: dynamical GENIE3 for the inference of
    gene networks from time series expression data. Sci Rep 8, 3384 (2018) doi:10.1038/s41598-018-21715-0
    """

    ngenes = TS_data[0].shape[1]
    nexp = len(TS_data)

    C_min = TS_data[0].min()
    if nexp > 1:
        for current_timeseries in TS_data[1:]:
            C_min = min(C_min, current_timeseries.min())

    alphas = np.zeros((nexp, ngenes))

    for (i, current_timeseries) in enumerate(TS_data):
        current_time_points = time_points[i]

        for j in range(ngenes):
            idx_min = np.argmin(current_timeseries[:, j])
            idx_max = np.argmax(current_timeseries[:, j])

            xmin = current_timeseries[idx_min, j]
            xmax = current_timeseries[idx_max, j]

            tmin = current_time_points[idx_min]
            tmax = current_time_points[idx_max]

            xmin = max(xmin - C_min, 1e-6)
            xmax = max(xmax - C_min, 1e-6)

            xmin = np.log(xmin)
            xmax = np.log(xmax)

            alphas[i, j] = (xmax - xmin) / abs(tmin - tmax)

    alphas = alphas.max(axis=0)

    return alphas


def get_importances(TS_data, time_points, alpha="from_data", SS_data=None, gene_names=None, regulators='all', b=1):
    time_start = time.time()

    ngenes = TS_data[0].shape[1]

    if alpha is "from_data":
        alphas = estimate_degradation_rates(TS_data, time_points)
    else:
        alphas = [alpha] * ngenes

    # Get the indices of the candidate regulators
    idx = [i for i, gene in enumerate(gene_names) if gene in regulators]

    # Learn an ensemble of trees for each target gene, and compute scores for candidate regulators
    VIM = np.zeros((ngenes, ngenes))
    # print('ngenes:',ngenes)
    for i in range(ngenes):
        # print('i:',i)
        input_idx = idx.copy()
        if i in input_idx:
            input_idx.remove(i)
        vi = get_importances_single(TS_data, time_points, alphas[i], input_idx, i, SS_data, b)#note: imput_idx
        # print('vi:',vi)
        VIM[i, :] = vi

    time_end = time.time()
    #print('W_var', VIM)
    #print("Elapsed time: %.2f seconds" % (time_end - time_start))

    return VIM


def get_importances_single(TS_data, time_points, alpha, input_idx, output_idx, SS_data, b):
    h = 1  # define the value of time step

    ngenes = TS_data[0].shape[1]
    nexp = len(TS_data)
    nsamples_time = sum([expr_data.shape[0] for expr_data in TS_data])
    ninputs = len(input_idx)

    # Construct training sample

    # Time-series data
    input_matrix_time = np.zeros((nsamples_time - h * nexp, ninputs))
    output_vect_time = np.zeros(nsamples_time - h * nexp)

    nsamples_count = 0
    for (i, current_timeseries) in enumerate(TS_data):
        current_time_points = time_points[i]
        npoints = current_timeseries.shape[0]
        time_diff_current = current_time_points[h:] - current_time_points[:npoints - h]
        current_timeseries_input = current_timeseries[:npoints - h, input_idx]
        current_timeseries_output = (current_timeseries[h:, output_idx] - current_timeseries[:npoints - h,
                                                                          output_idx]) / time_diff_current + alpha * current_timeseries[
                                                                                                                     :npoints - h,
                                                                                                                     output_idx]
        nsamples_current = current_timeseries_input.shape[0]
        input_matrix_time[nsamples_count:nsamples_count + nsamples_current, :] = current_timeseries_input
        output_vect_time[nsamples_count:nsamples_count + nsamples_current] = current_timeseries_output
        nsamples_count += nsamples_current

    # Steady-state data
    if SS_data is not None:
        input_matrix_steady = SS_data[:, input_idx]
        output_vect_steady = SS_data[:, output_idx] * alpha

        # Concatenation
        input_all = np.vstack([input_matrix_steady, input_matrix_time])
        output_all = np.concatenate((output_vect_steady, output_vect_time))
    else:
        input_all = input_matrix_time
        output_all = output_vect_time
    #mrmr feature selection
    top_gene = 9
    out = output_all.reshape(output_all.shape[0],1)
    output = pd.DataFrame(out)
    output.columns = ['G' + str(output_idx) ]
    input1 = pd.DataFrame(input_all)
    input1.columns =  ['G'+str(i) for i in input_idx]
    df3 = pd.concat([output, input1], axis=1)
    selF1 = np.array(pymrmr.mRMR(df3, 'MIQ', top_gene))
    input_all = input1[selF1]
    # Compute importance scores
    w, v = FMpredict(input_all.values, output_all, b, selF1)
    input_idx = [int(i.replace('G','')) for i in selF1]
    vi = getvar_w(w, top_gene, b)
    # vi = getMAXMIN_w(w, ngenes-1, b)
    # vi = getabsmax_w(w, ngenes-1, b)
    # vi = getabssum_w(w, ngenes-1, b)
    print('w_var',vi)
    print('input_idx',input_idx)
    v_i = np.zeros(ngenes)
    v_i[input_idx] = vi
    return v_i

def get_importances_single1( alpha, input_idx, output_idx, SS_data, b):

    ngenes = SS_data[0].shape[1]


    # Construct training sample

    # Steady-state data

    input_matrix_steady = SS_data[:, input_idx]
    output_vect_steady = SS_data[:, output_idx] * alpha

    # Compute importance scores
    w, v = FMpredict(input_matrix_steady, output_vect_steady, b)

    vi = getvar_w(w, 3, b)
    # vi = getMAXMIN_w(w, ngenes-1, b)
    # vi = getabsmax_w(w, ngenes-1, b)
    # vi = getabssum_w(w, ngenes-1, b)
    # print('w_var',vi)
    v_i = np.zeros(ngenes)
    v_i[input_idx] = vi
    return v_i

def getabssum_w(w, ngenes, bin):
    vi = np.zeros(ngenes)
    for i in range(0, ngenes):
        max_w = 0
        for j in range(i * bin, (i + 1) * bin):
            if (i + 1) * bin > w.shape[0]:
                break;
            max_w += abs(w[j])
        vi[i] = max_w

    return vi


def getMAXMIN_w(w, ngenes, bin):
    vi = np.zeros(ngenes)
    for i in range(0, ngenes):
        max_w = 0
        min_w = 0
        for j in range(i * bin, (i + 1) * bin):
            if (i + 1) * bin > w.shape[0]:
                break;
            max_w = max(max_w, w[j])
            min_w = min(min_w, w[j])
        vi[i] = max_w - min_w

    return vi


def getabsmax_w(w, ngenes, bin):
    vi = np.zeros(ngenes)
    for i in range(0, ngenes):
        max_w = 0
        for j in range(i * bin, (i + 1) * bin):
            if (i + 1) * bin > w.shape[0]:
                break;
            max_w = max(max_w, abs(w[j]))
        vi[i] = max_w

    return vi


def getvar_w(w, ngenes, bin):
    vi = np.zeros(ngenes)
    for i in range(0, ngenes):
        arrra = []
        if i == ngenes:
            break;
        arrra = w[i * bin:(i + 1) * bin]
        arr_var = np.var(arrra)
        if math.isnan(arr_var):
            arr_var = 0
        vi[i] = arr_var

    return vi


#####
def get_scores(VIM, gold_edges, gene_names, regulators):
    idx = [i for i, gene in enumerate(gene_names) if gene in regulators]
    pred_edges = [(gene_names[j], gene_names[i], score) for (i, j), score in np.ndenumerate(VIM) if i != j and j in idx]
    pred_edges = pd.DataFrame(pred_edges)
    # Take the top 100,000 predicated results
    pred_edges = pred_edges.iloc[:100000]
    final = pd.merge(pred_edges, gold_edges, on=[0, 1], how='inner')
    # np.set_printoptions(threshold=10000)
    # print('2_y', final['2_y'])
    # sprint('2_x', final['2_x'])
    auroc = roc_auc_score(final['2_y'], final['2_x'])

    fpr, tpr, thresholds = roc_curve(final['2_y'], final['2_x'], pos_label=2)
    # print("fpr",fpr,"tpr",tpr,"thresholds",thresholds )
    aupr = average_precision_score(final['2_y'], final['2_x'])

    return auroc, aupr, final





