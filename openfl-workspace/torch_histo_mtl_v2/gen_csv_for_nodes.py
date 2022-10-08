import sys, os, glob
import pandas as pd
import numpy as np
from sklearn.utils import shuffle


def gen_splits(df):
    counts = {
        'PAM50_and_Claudin-low_(CLOW)_Molecular_Subtype': 5,
        'HistoAnno': 3,
        'IHC_HER2': 3
    }
    dfs = None
    while True:
        df = shuffle(df)
        dfs = np.array_split(df, num_nodes)
        good = [np.all([len(dfs[i][key].unique()) == count for i in range(num_nodes)])
                for key, count in counts.items()]
        if np.all(good):
            break

    return dfs


split_num = int(sys.argv[1])
num_nodes = int(sys.argv[2])

if not os.path.exists('/data/zhongz2/BigData/TCGA-BRCA/splits_HistoAnno/train-{}-node-0.csv'.format(split_num)):
    csv_filename = '/data/zhongz2/BigData/TCGA-BRCA/splits_HistoAnno/train-{}.csv'.format(split_num)
    df = pd.read_csv(csv_filename)

    dfs = gen_splits(df)
    for i in range(num_nodes):
        dfs[i].to_csv('/data/zhongz2/BigData/TCGA-BRCA/splits_HistoAnno/train-{}-node-{}.csv'.format(split_num, i))






