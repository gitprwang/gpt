# 本文件负责数据集处理
# zinc数据集smiles正则化
# 反应数据集smiles正则化，并写入json文件
# 反应数据集和zinc数据集tokenize
# 反应数据集拼接成序列数据
# zinc和反应数据集的生成物计算属性
# 属性和序列数据拼接组成数据集

from .rdkit_tools import *

# zinc
def process_zinc(path, save_dir):
    '''
    该函数将每个smiles保存为json文件
    path：txt路径，该txt包含zinc中所有smiles
    save_dir：保存json文件的路径
    '''
    data = open(path, 'r')
    for i, line in enumerate(data.readlines()):
        if line[-1] == '\n':
            line = line[:-1]
        smiles = norm_smiles(line)
        props = cal_props(get_mol(smiles))
    pass