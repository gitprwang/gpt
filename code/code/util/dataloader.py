import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import re
from rdkit_tools import cal_props, cal_scaffold, get_mol
import json
import tqdm
import random

'''
pattern =  "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
regex = re.compile(pattern)
smiles += str('<')*(self.max_len - len(regex.findall(smiles)))
'''

def process_zinc(path, save_file):
    '''
    将ZINC数据集中的SMILES保存到save_file文件中
    '''
    all_smiles_list = open(save_file, '+w')
    for data_file in os.listdir(path):
        data_file = os.path.join(path, data_file)
        data = pd.read_csv(data_file, sep='\t')
        smiles_list = data['smiles'].tolist()
        for smiles in smiles_list:
            all_smiles_list.write(smiles+'\n')
    all_smiles_list.close()

def process_ustpo(path, save_file):
    '''
    *.rsmi文件 ReactionSmiles	PatentNumber	ParagraphNum	Year	TextMinedYield	CalculatedYield
    '''
    reaction_smiles_file = open(save_file, '+w')
    for data_file in os.listdir(path):
        if data_file[-4:]!='rsmi':
            continue
        data = pd.read_csv(os.path.join(path, data_file), sep='\t')
        reaction_smiles = data['ReactionSmiles']
        reaction_smiles_list = reaction_smiles.tolist()
        for smiles in reaction_smiles_list:
            reaction_smiles_file.write(smiles.split(' ')[0]+'\n')
    reaction_smiles_file.close()

def process_rxn(path, save_file):
    '''
    rid	tid	reactants	products	cats	sols
    '''
    data = pd.read_csv(path, sep='\t')
    def map(string):
        string = str(string).split('"')
        mol_list = []
        for item in string:
            if item in "[], ":
                continue
            mol_list.append(item)
        return mol_list
    reaction_smiles_file = open(save_file, '+w')
    reactants = data['reactants'].apply(map).tolist()
    products = data['products'].apply(map).tolist()
    cats = data['cats'].apply(map).tolist()
    sols = data['cats'].apply(map).tolist() 
    for i in range(len(reactants)):
        item = ''
        for rec in reactants[i]:
            item += rec + '.'
        if item[-1]!='.':
            continue
        item = item[:-1]+'>'
        for cat in cats[i]:
            item += cat + '.'
        for sol in sols[i]:
            item += sol + '.'
        if item[-1]=='.':
            item = item[:-1]+'>'
        else:
            item += '>'
        for prod in products[i]:
            item += prod + '.'
        if item[-1]!='.':
            continue
        item = item[:-1]
        reaction_smiles_file.write(item+'\n')
    reaction_smiles_file.close()

def process_reaction_smiles(smiles):
    '''
    smiles:  reactant.reactant>reagent>product.product
    '''
    temp = smiles.split('>')
    if len(temp)==3:
        reactants, reagents, products = temp[0], temp[1], temp[2]
    else:
        # print(smiles)
        return None
    try:
        props = cal_props(get_mol(products.split('.')[-1]))
    except:
        props = None
        # print(smiles)
    return {
        'type':'reaction',
        'reactants':reactants.split('.'),
        'reagents':reagents.split('.'),
        'products':products.split('.'),
        'props':props
    }

def process_smiles(smiles):
    '''
    将SMILES组织称目标数据格式   props + scaffold + smiles
    '''
    mol = get_mol(smiles)
    props = cal_props(mol)
    scaffold = cal_scaffold(mol)
    return {
        'type':'molecular',
        'props':props,
        'scaffold':scaffold,
        'smiles':smiles
    }

def generate_reaction_json(txt_files, save_file):
    save_file = open(save_file, '+w')
    data = []
    for txt_file in txt_files:
        print('processing '+txt_file)
        txt_data = open(txt_file, 'r')
        for i, line in enumerate(tqdm.tqdm(txt_data)):
            if line[-1]=='\n':
                line = line[:-1]
                item = process_reaction_smiles(line)
                if item is not None:
                    data.append(item)
        txt_data.close()

    text = json.dumps(data)
    save_file.write(text)
    save_file.close()
    

def generate_smiles_json(txt_files, save_file, summary):
    # save_file = open(save_file, '+w')
    file_list = []
    data = []
    total = 0
    for txt_file in txt_files:
        txt_data = open(txt_file, 'r')
        for i, line in enumerate(tqdm.tqdm(txt_data)):
            if line[-1]=='\n':
                line = line[:-1]
            data.append(process_smiles(line))
            if len(data)==10000:
                text = json.dumps(data)
                save_file_i = save_file+'_'+str(i)+'_'+str(i+len(data))+'.json'
                file_list.append(save_file_i)
                save_file_i = open(save_file_i, '+w')
                save_file_i.write(text)
                data = []
            total+=10000
        if len(data)>0:
            text = json.dumps(data)
            save_file_i = save_file+'_'+str(total)+'_'+str(total+len(data))+'.json'
            file_list.append(save_file_i)
            save_file_i = open(save_file_i, '+w')
            save_file_i.write(text)
            data = []
        txt_data.close()

    s = open(summary, '+w')
    for file in file_list:
        s.write(file+'\n')
    s.close()

    # text = json.dumps(data)
    # save_file.write(text)
    # save_file.close() 


class ChemData():
    '''
    ChemData负责将raw txt style数据集转化成 json数组形式的格式化数据用于模型训练。
    json attributes:
    {
        'type':'reaction',
        'reactants':[reactants],
        'reagents':[reagents],
        'products':[products],
        'props':[props],
        'scaffold':scaffold,
        'smiles':smiles
    }
    '''
    def __init__(self, mol_data, reaction_data, ratios=[0.1,1,1], train_val_test_split=[0.9, 0.91, 1], tasks=['mol_gen','reaction','retro','prop']):
        self.data = []

        # 读原始数据 list of dicts
        molecules = self.generate_mol_data(mol_data)
        reactions, retro = self.generate_reaction_retro_data(reaction_data)
        
        # 调整比例----
        random.shuffle(molecules)
        random.shuffle(reactions)
        random.shuffle(retro)
        print(len(molecules),len(reactions),len(retro))
        molecules = molecules[:int(ratios[0]*len(molecules))]
        reactions = reactions[:int(ratios[1]*len(reactions))]
        retro = reactions[:int(ratios[2]*len(retro))]
        print(len(molecules),len(reactions),len(retro))
        # -----------

        # 合并
        self.train_data = molecules[:int(train_val_test_split[0]*len(molecules))] + \
                            reactions[:int(train_val_test_split[0]*len(reactions))] + \
                            retro[:int(train_val_test_split[0]*len(retro))]
        
        self.val_data = molecules[int(train_val_test_split[0]*len(molecules)):int(train_val_test_split[1]*len(molecules))] + \
                            reactions[int(train_val_test_split[0]*len(reactions)):int(train_val_test_split[1]*len(reactions))] + \
                            retro[int(train_val_test_split[0]*len(retro)):int(train_val_test_split[1]*len(retro))]
        
        self.test_data = molecules[int(train_val_test_split[1]*len(molecules)):] + \
                            reactions[int(train_val_test_split[1]*len(reactions)):] + \
                            retro[int(train_val_test_split[1]*len(retro)):]



    def generate_mol_data(self, txt_files):
        molecules = []
        for txt_file in txt_files:
            txt_data = open(txt_file, 'r')
            for i, line in enumerate(tqdm.tqdm(txt_data)):
                # if i==1000:
                #     break
                if line[-1]=='\n':
                    line = line[:-1]
                molecules.append({'type':'molecular','smiles':line}) # process_smiles(line)
        return molecules

    def generate_reaction_retro_data(self, txt_files):
        reactions = []
        retro = []
        for txt_file in txt_files:
            print('processing '+txt_file)
            txt_data = open(txt_file, 'r')
            for i, line in enumerate(tqdm.tqdm(txt_data)):
                # if i==1000:
                #     break
                if line[-1]=='\n':
                    line = line[:-1]
                    item = process_reaction_smiles(line)
                    if item is not None and len(item['products'])==1:
                        # print('adding sample')
                        reactions.append(item)
                        item['type'] = 'retro'
                        try:
                            props = cal_props(get_mol(item['reactants'][-1]))
                        except:
                            props = None
                        item['props'] = props
                        retro.append(item)
        return reactions, retro
    

class ChemDataSet(Dataset):
    def __init__(self, data):
        self.data = data
        self.tokenizer = MyTokenizer()
    
    def pad_sequence(self, sequence, max_length, padding_value=0):
        if len(sequence) < max_length:
            padded_sequence = sequence + [padding_value] * (max_length - len(sequence))
        else:
            padded_sequence = sequence[:max_length]
        return padded_sequence


    def __getitem__(self, index):
        item = self.data[index]
        # preprocess 将dict组织成字符串
        task = item['type']
        if task=='reaction':
            props = item['props']
            train_str = ''
            for reactant in item['reactants']:
                train_str += reactant+'.'
            if train_str[-1]=='.':
                train_str = train_str[:-1]+'>'
            for reactant in item['reagents']:
                train_str += reactant+'.'
            if train_str[-1]=='.':
                train_str = train_str[:-1]+'>'
            for reactant in item['products']:
                train_str += reactant+'.'
            if train_str[-1]=='.':
                train_str = train_str[:-1]+'!'
            # return props, train_str
        elif task=='retro':
            props = item['props']
            train_str = ''
            for reactant in item['products']:
                train_str += reactant+'.'
            if train_str[-1]=='.':
                train_str = train_str[:-1]+'<'
            for reactant in item['reagents']:
                train_str += reactant+'.'
            if train_str[-1]=='.':
                train_str = train_str[:-1]+'<'
            for reactant in item['reactants']:
                train_str += reactant+'.'
            if train_str[-1]=='.':
                train_str = train_str[:-1]+'!'
            # return props, train_str
        elif task=='molecular':
            item = process_smiles(item['smiles'])
            props = item['props']
            train_str = item['scaffold']+'.'+item['smiles']+'!'
            # return props, train_str
        else:
            raise KeyError('Wrong task!!!')
        train_str = self.tokenizer.encode(train_str)
        props = list(props.values())
        sequence = props + train_str

        max_length = 400  # 指定填充\截断后的序列最大长度
        padded_sequence = self.pad_sequence(sequence, max_length)

        return torch.tensor(padded_sequence, dtype=torch.long)

        # print(len(self.processed_data))
        # self.processed_data = self.pad_sequence(sequence)
        # 这里的prop直接序列化为4个元素，与字典props中的顺序一一对应。
        # 根据ChemGPT的结构，应该后续将props和seq分开分别传入对应的encoder
        # 上述分离考虑放在ChemGPT的forward中，因为dataloader不方便分离。
        # return sequence
    
    def __len__(self):
        return len(self.data)

   

class MyTokenizer():
    def __init__(self, charset_file='/home/data/wd/all_chars.txt'):
        self.char2num = {}
        cnt = 0
        charset_file = open(charset_file, 'r')
        for char in charset_file.readlines():
            char = char[:-1] if char[-1]=='\n' else char
            # print(char)
            self.char2num[char] = cnt
            cnt += 1
        self.num2char = {}
        for char in self.char2num.keys():
            self.num2char[self.char2num[char]] = char

    def encode(self, string):
        return list(map(lambda x:self.char2num[x], string))

    def decode(self, index):
        return ''.join(list(map(lambda x:self.num2char[x], index)))


if __name__ == '__main__':
    print('in main')
    zinc_data_path = '/home/Zhouyu/MODEL/task1/test/asd/a.txt'
    uspto_data_path = '/home/Zhouyu/MODEL/task1/test/asd/b.txt'
    rxn_data_path = '/home/Zhouyu/MODEL/task1/test/asd/c.txt'
    all_data = ChemData([zinc_data_path], [uspto_data_path, rxn_data_path])
    
    print(len(all_data.train_data))
    dataset = ChemDataSet(all_data.train_data)
    print(dataset[0], len(dataset))
