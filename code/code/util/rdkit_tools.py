from rdkit import Chem
from rdkit.Contrib.SA_Score.sascorer import calculateScore
from rdkit.Chem.Descriptors import MolLogP, TPSA
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem.QED import qed
from rdkit.Chem.Scaffolds import MurckoScaffold
import tqdm
import atomInSmiles

# 屏蔽rdkit warning
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Chem.MolFromSmiles(smiles)

def norm_smiles(smiles):
    return atomInSmiles.decode(atomInSmiles.encode(smiles))

def tokenize(smiles):
    return atomInSmiles.encode(smiles)

def cal_props(mol): # see MolGPT
    return {
        'logp':MolLogP(mol), 
        'tpsa':TPSA(mol), 
        'qed':qed(mol), 
        'sas':calculateScore(mol)
    }

def cal_inchikey(mol):
    return Chem.MolToInchiKey(mol)

def cal_scaffold(mol):
    return MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))

def get_mol(smiles):
    return MolFromSmiles(smiles)


def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    import re
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)

def get_chars(data_files, charset_file): # smiles tokenizer, may produce too many tokens
    charset = set()
    
    charset_writer = open(charset_file, '+w')
    import re
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    
    for data_file in data_files:
        data = open(data_file, 'r')
        for i, line in enumerate(tqdm.tqdm(data)):
            if line[-1]=='\n':
                line = line[:-1]
            tokens = [token for token in regex.findall(line)]
            for token in tokens:
                charset.add(token)
        data.close()
    print('find {} chars\n'.format(len(charset)))
    for char in charset:
        charset_writer.write(char+'\n')
    charset_writer.close()

def char_wise_tokenizer(data_files, charset_file): # char-wise tokenizer
    charset = set()
    
    charset_writer = open(charset_file, '+w')
    for data_file in data_files:
        data = open(data_file, 'r')
        for i, line in enumerate(tqdm.tqdm(data)):
            if line[-1]=='\n':
                line = line[:-1]
            tokens = [token for token in line]
            for token in tokens:
                charset.add(token)
        data.close()
    print('find {} chars\n'.format(len(charset)))
    for char in charset:
        charset_writer.write(char+'\n')
    charset_writer.close()


if __name__ == '__main__':
    # smiles1 = 'O=C1N2CN3C(=O)N4CN5C(=O)N6CN7C(=O)N8CN9C(=O)NCNC(=O)NCN1C1C2N2CNC(=O)N(CNC(=O)N(CNC(=O)[N:3](CNC(=O)N(CNC(=O)N(CN1C2=O)CC)CC9)[CH:4]8[CH:5]7)C6C5)C4C3'
    # smiles2 = 'O=C1N2CN3C(=O)N4CN5C(=O)N6CN7C(=O)N8CN9C(=O)N%10CN%11C(=O)N%12CN1C1C2N2CN%13C(=O)N(CN%14C(=O)N(CN%15C(=O)[N:3](CN%16C(=O)N(CN%17C(=O)N(CN1C2=O)C%12C%11%17)C%10C9%16)[CH:4]8[CH:5]7%15)C6C5%14)C4C3%13'
    # mol1 = MolFromSmiles(smiles1)
    # mol2 = MolFromSmiles(smiles2)
    # print(cal_props(mol1))
    # print(cal_props(mol2))
    zinc_data_path = '/home/data/wd/zinc_data.txt'
    uspto_data_path = '/home/data/wd/USPTO/uspto_data.txt'
    rxn_data_path = '/home/data/wd/rxn.txt'
    char_wise_tokenizer([uspto_data_path, rxn_data_path, zinc_data_path], '/home/data/wd/all_chars.txt')
 