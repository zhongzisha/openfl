import sys,os,glob,shutil
import pandas as pd

if __name__ == '__main__':
    shard_num = sys.argv[1]
    csv_filename = '/data/zhongz2/BigData/TCGA-BRCA/openfl_split_{}.csv'.format(shard_num)
    svs_root = './data/svs/'
    os.makedirs(svs_root, exist_ok=True)
    df = pd.read_csv(csv_filename)
    for dx_filename in df['DX_filename'].values:
        real_path = os.path.realpath(dx_filename)
        os.system('cp {} {}'.format(real_path, svs_root))
        print(dx_filename)









