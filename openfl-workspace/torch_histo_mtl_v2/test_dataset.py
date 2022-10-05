
import sys,os,glob,shutil,json
import pandas as pd
import pdb
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


dataset_root = '/data/zhongz2/BigData/TCGA-BRCA/'
csv_filename = '/data/zhongz2/BigData/TCGA-BRCA/tcga_brca_one_case.csv'
df = pd.read_csv(csv_filename)

invalid_CLIDs = []
removed_columns = [col for col in df.columns if '_cnv' in col or '_rnaseq' in col]
df = df.drop(columns=removed_columns)

if 'DX_filename' not in df.columns:
    dx_filenames = []
    for slideid in df['slide_id'].values:
        dx_filenames.append(os.path.join(dataset_root, 'svs', slideid))
    df['DX_filename'] = dx_filenames

df = df[df.DX_filename.isnull() == False]
if len(invalid_CLIDs) > 0:
    df = df.drop(invalid_CLIDs)
# df = df[df[df.columns[10]] != 'CLOW']  # PAM50_and_Claudin-low_(CLOW)_Molecular_Subtype
# df = df.dropna()

# remove those rows with `X` values 20220213 from Peng
# df = df[df['ajcc_pathologic_stage'].isin(STAGES_DICT.keys())]
# df['ajcc_pathologic_stage'] = df['ajcc_pathologic_stage'].map(STAGES_DICT)

SUBTYPES_DICT = {
    'LumA': 0,
    'LumB': 1,
    'Basal': 2,
    'HER2E': 3,
    'normal-like': 4
}
df = df[df['PAM50_and_Claudin-low_(CLOW)_Molecular_Subtype'].isin(SUBTYPES_DICT.keys())]
df['PAM50_and_Claudin-low_(CLOW)_Molecular_Subtype'] = \
    df['PAM50_and_Claudin-low_(CLOW)_Molecular_Subtype'].map(SUBTYPES_DICT).fillna(4)

# df = df[df['IHC_HER2'].isin(['Positive', 'Negative'])]
IHC_HER2_dict = {k: 2 for k in df['HER2.newly.derived'].value_counts().index}
IHC_HER2_dict['Positive'] = 1
IHC_HER2_dict['Negative'] = 0
df['IHC_HER2'] = df['HER2.newly.derived'].map(IHC_HER2_dict).fillna(2)

# try oncotree_code
HistoAnno_dict = {k: 2 for k in df['2016_Histology_Annotations'].value_counts().index}
HistoAnno_dict['Invasive ductal carcinoma'] = 1
HistoAnno_dict['Invasive lobular carcinoma'] = 0
df['HistoAnno'] = df['2016_Histology_Annotations'].map(HistoAnno_dict).fillna(2)

# STAGES_DICT = {
#     'Stage I': 0,
#     'Stage IA': 1,
#     'Stage IB': 2,
#     'Stage II': 3,
#     'Stage IIA': 4,
#     'Stage IIB': 5,
#     'Stage III': 6,
#     'Stage IIIA': 7,
#     'Stage IIIB': 8,
#     'Stage IIIC': 9,
#     'Stage IV': 10
# }
Stage_dict = {k: 1.0 for k in df['ajcc_pathologic_stage'].value_counts().index}
Stage_dict['Stage I'] = 0
Stage_dict['Stage IA'] = 0.1
Stage_dict['Stage IB'] = 0.2
Stage_dict['Stage II'] = 0.3
Stage_dict['Stage IIA'] = 0.4
Stage_dict['Stage IIB'] = 0.5
Stage_dict['Stage III'] = 0.6
Stage_dict['Stage IIIA'] = 0.7
Stage_dict['Stage IIIB'] = 0.8
Stage_dict['Stage IIIC'] = 0.9
df['Stage'] = df['ajcc_pathologic_stage'].map(Stage_dict).fillna(1.0)

# dropna for some columns
df = df[df['PAM50_and_Claudin-low_(CLOW)_Molecular_Subtype'].notna()]
df = df[df['Stage'].notna()]
df = df[df['IHC_HER2'].notna()]
df = df[df['HistoAnno'].notna()]

# pdb.set_trace()

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
X = np.random.rand(len(df), 1)
y = df['HistoAnno'].values
for train_index, test_index in sss.split(X, y):
    print('train_index', train_index)
    print('test_index', test_index)









