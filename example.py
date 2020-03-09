#from xgbgrn import *
from FMModel import *
import pandas as pd
import os
import matplotlib.pyplot as plt

###***
#This example combines the time-series data of insilico_size10 and the steady-state data of  insilico_size10
#to infer gene regulatory networks.
###***
dir = os.getcwd()
TS_data = pd.read_csv(dir+"/data/insilico_size100_4_timeseries.tsv", sep='\t').values
SS_data_1 = pd.read_csv(dir+"/data/insilico_size100_4_knockouts.tsv", sep='\t').values
SS_data_2 = pd.read_csv(dir+"/data/insilico_size100_4_knockdowns.tsv", sep='\t').values

# get the steady-state data
SS_data = np.vstack([SS_data_1, SS_data_2])

i = np.arange(0, 85, 21)
j = np.arange(21, 106, 21)

# get the time-series data
TS_data = [TS_data[i:j] for (i, j) in zip(i, j)]
# get time points
time_points = [np.arange(0, 1001, 50)] * 5

ngenes = TS_data[0].shape[1]
gene_names = ['G'+str(i+1) for i in range(ngenes)]
regulators = gene_names.copy()

gold_edges = pd.read_csv(dir+"/data/insilico_size100_4_goldstandard.tsv", '\t', header=None)

AUROC = []
AUPR = []

for i in range(100,200):
    #print('i=',i)
    VIM = get_importances(TS_data, time_points, alpha=0.0214, SS_data=SS_data, gene_names=gene_names,
                      regulators=regulators, b=i)
    auroc, aupr = get_scores(VIM, gold_edges, gene_names, regulators)
    AUROC.append(auroc)
    AUPR.append(aupr)
    #print("AUROC:", auroc, "AUPR:", aupr)


print(AUROC)
print(AUPR)
