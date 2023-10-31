from config import cfg
from util.model import PropEncoder, ChemGPT
from util.dataloader import ChemData, ChemDataSet
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


cfg = cfg()
cfg.get_args()
cfgs = cfg.update_train_configs()

zinc_data_path = '/home/Zhouyu/MODEL/task1/test/asd/a.txt'
uspto_data_path = '/home/Zhouyu/MODEL/task1/test/asd/b.txt'
rxn_data_path = '/home/Zhouyu/MODEL/task1/test/asd/c.txt'

all_data = ChemData([zinc_data_path], [uspto_data_path, rxn_data_path])
train_data = ChemDataSet(all_data.train_data)
val_data = ChemDataSet(all_data.val_data)
test_data = ChemDataSet(all_data.test_data)

model = ChemGPT(cfgs)

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=2)
for batch in train_dataloader:
    out, shape = model(batch)
    print(out, shape)
   

