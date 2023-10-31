from config import cfg
from util.model import PropEncoder, ChemGPT
from util.dataloader import ChemData, ChemDataSet
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


cfg = cfg()
cfg.get_args()
cfgs = cfg.update_train_configs()

zinc_data_path = '/home/data/wd/zinc_data.txt'
uspto_data_path = '/home/data/wd/USPTO/uspto_data.txt'
rxn_data_path = '/home/data/wd/rxn.txt'


zinc_data_path = '/home/Zhouyu/MODEL/task1/test/asd/a.txt'
uspto_data_path = '/home/Zhouyu/MODEL/task1/test/asd/b.txt'
rxn_data_path = '/home/Zhouyu/MODEL/task1/test/asd/c.txt'

all_data = ChemData([zinc_data_path], [uspto_data_path, rxn_data_path])
train_data = ChemDataSet(all_data.train_data)
val_data = ChemDataSet(all_data.val_data)
test_data = ChemDataSet(all_data.test_data)

# padding
data = test_data + train_data + val_data
print(train_data.__getitem__(10))
# max_length = max(len(sequence) for sequence in data)
max_length = max(len(sequence) for sequence in data)
print("最大序列长度:", max_length)


padded_train = pad_sequence(train_data, batch_first=True, padding_value=0)
print(padded_train[0].shape)

model = ChemGPT(cfgs)