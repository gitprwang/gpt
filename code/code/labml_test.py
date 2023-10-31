import torch
from torch import nn

from labml import experiment
from labml.configs import option
from labml_helpers.module import Module
from labml_nn.experiments.nlp_autoregression import NLPAutoRegressionConfigs
from labml_nn.optimizers.configs import OptimizerConfigs
from labml_nn.transformers import TransformerConfigs, Encoder
from labml_nn.transformers.utils import subsequent_mask

class GPT(Module):
    def __init__(self, encoder: Encoder, src_embed: Module, generator: Module):
        super().__init__()
        self.src_embed = src_embed
        self.encoder = encoder
        self.generator = generator
        self.mask = None
    
    def forward(self, x: torch.Tensor):
        if self.mask is None or self.mask.size(0) != len(x):
            self.mask = subsequent_mask(len(x)).to(x.device)
        x = self.src_embed(x)
        x = self.encoder(x, self.mask)
        x = self.generator(x)
        return x, None
    
class Configs(NLPAutoRegressionConfigs):
    model: GPT
    transformer: TransformerConfigs
    weight_decay: float = 0.1
    warmup_steps: int = 128 * 128 * 20
    optimizer = 'transformer_optimizer'

@option(Configs.transformer, 'GPT')
def _transformer_configs(c: Configs):
    conf = TransformerConfigs()
    

    


    


        
