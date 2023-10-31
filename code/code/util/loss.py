import torch
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch import nn
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import Descriptors

import selfies as sf
from rdkit.Chem import QED

class Loss_log():
    def __init__(self):
        self.loss = [999999.]
        self.acc = [0.]
        self.flag = 0
        self.token_right_num = []
        self.token_all_num = []
        self.use_top_k_acc = 0

    def acc_init(self, topn=[1]):
        self.loss = []
        self.token_right_num = []
        self.token_all_num = []
        self.topn = topn
        self.use_top_k_acc = 1
        self.top_k_word_right = {}
        for n in topn:
            self.top_k_word_right[n] = []
    
    def get_token_acc(self):
        if len(self.token_all_num) == 0:
            return 0.
        elif self.use_top_k_acc == 1:
            res = []
            for n in self.topn:
                res.append(round((sum(self.top_k_word_right[n]) / sum(self.token_all_num)) * 100 , 3))
            return res
        else:
            return [sum(self.token_right_num)/sum(self.token_all_num)]
    
    def update_token(self, token_num, token_right):
        self.token_all_num.append(token_num)
        if isinstance(token_right, list):
            for i, n in enumerate(self.topn):
                self.top_k_word_right[n].append(token_right[i])
        self.token_right_num.append(token_right)
        
    def update(self, case):
        self.loss.append(case)

    def update_acc(self, case):
        self.acc.append(case)

    def get_loss(self):
        return self.loss[-1]

    def get_acc(self):
        return self.acc[-1]

    def get_min_loss(self):
        return min(self.loss)

    def get_loss(self):
        if len(self.loss) == 0:
            return 500.
        return np.mean(self.loss)
    
    def early_stop(self):
        if self.loss[-1] > min(self.loss):
            self.flag += 1
        else:
            self.flag = 0

        if self.flag > 1000:
            return True
        else:
            return False

    def torch_accuracy(output, target, topk=(1,)):
        '''
        param output, target: should be torch Variable
        '''

        topn = max(topk)
        batch_size = output.size(0)

        _, pred = output.topk(topn, 1, True, True) 
        pred = pred.t() 

        is_correct = pred.eq(target.view(1, -1).expand_as(pred))

        ans = []
        ans_num = []
        for i in topk:
            is_correct_i = is_correct[:i].contiguous().view(-1).float().sum(0, keepdim=True)
            ans_num.append(int(is_correct_i.item()))
            ans.append(is_correct_i.mul_(100.0 / batch_size))

        return ans, ans_num