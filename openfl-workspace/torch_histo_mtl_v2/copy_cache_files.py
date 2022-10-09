import sys, os, glob, shutil
import pandas as pd
import tarfile

if __name__ == '__main__':
    split_num = sys.argv[1]
    shard_num = sys.argv[2]
    num_nodes = sys.argv[3]
    cache_data_type = 'img'

    csv_filenames = [
        '/data/zhongz2/BigData/TCGA-BRCA/splits_HistoAnno/train-{}-node-{}-of-{}.csv'.format(split_num, shard_num, num_nodes),
        '/data/zhongz2/BigData/TCGA-BRCA/splits_HistoAnno/val-{}.csv'.format(split_num)
    ]
    for csv_filename in csv_filenames:
        df = pd.read_csv(csv_filename)
        cache_root = './data/all/'
        os.makedirs(cache_root, exist_ok=True)
        if cache_data_type == 'pth':
            cache_root_original = '/data/zhongz2/tcga_brca_256/patch_images_cache_part1_0_np32_pth/all'
            for dx_filename in df['DX_filename'].values:
                real_path = os.path.realpath(dx_filename)
                svs_prefix = os.path.basename(dx_filename).replace('.svs', '')
                os.system('cp -R {} {}'.format(os.path.join(cache_root_original, svs_prefix),
                                               cache_root))
                print(dx_filename)
        else:
            cache_root_original = '/data/zhongz2/tcga_brca_256/patch_images_cache_part1_0_np2048/all/'
            for dx_filename in df['DX_filename'].values:
                real_path = os.path.realpath(dx_filename)
                svs_prefix = os.path.basename(dx_filename).replace('.svs', '')
                os.system('cp -R {} {}'.format(os.path.join(cache_root_original, svs_prefix + '.txt'),
                                               cache_root))
                tar = tarfile.open(os.path.join(cache_root_original, svs_prefix + '.tar.gz'))
                tar.extractall(cache_root)
                tar.close()
                print(dx_filename)
