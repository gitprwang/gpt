from transformers import GPT2Model, GPT2Config, GPT2PreTrainedModel # , CausalLMOutputWithCrossAttentions
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, L1Loss

class ChemGPTConfig(GPT2Config):
    def __init__(
        self,
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_props = 3,
        n_inner=None,
        alpha=1,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        scale_attn_by_inverse_layer_idx=False,
        reorder_and_upcast_attn=False,
        **kwargs,
    ):
        
        self.n_props = n_props
        self.alpha = alpha
        super().__init__(
            vocab_size = vocab_size,
            n_positions = n_positions,
            n_layer = n_layer,
            n_head = n_head,
            n_inner = n_inner,
            n_embd = n_embd,
            activation_function = activation_function,
            resid_pdrop = resid_pdrop,
            embd_pdrop = embd_pdrop,
            attn_pdrop = attn_pdrop,
            layer_norm_epsilon = layer_norm_epsilon,
            initializer_range = initializer_range,
            summary_type = summary_type,
            summary_use_proj = summary_use_proj,
            summary_activation = summary_activation,
            summary_first_dropout = summary_first_dropout,
            summary_proj_to_labels = summary_proj_to_labels,
            scale_attn_weights = scale_attn_weights,
            use_cache = use_cache,
            scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx,
            reorder_and_upcast_attn = reorder_and_upcast_attn,
            bos_token_id=bos_token_id, 
            eos_token_id=eos_token_id, **kwargs)

class ChemGPT(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.num_props = config.n_props
        self.num_embed = config.n_embd
        self.prop_in_linear = nn.Linear(self.num_props, self.num_embed)  # 性质到embed
        self.prop_out_linear = nn.Linear(self.num_embed, self.num_props) # hidden state到性质
        self.lm_head = nn.Linear(self.num_embed, config.vocab_size, bias=False) # hidden state到token
        self.alpha = config.alpha # 性质预测loss的权重

    def forward(self, data, labels=None):
        ''' 
        data: [props, seqs]    不带终止符 b, t, d
        labels: [seqs, props]  带终止符 b, t+1, d
        '''
        props, seqs = data[0], data[1]
        prop_embeds = self.prop_in_linear(props) # b, d
        prop_embeds = prop_embeds.unsqueeze(1) # b, 1, d

        seq_embeds = self.transformer.wte(seqs)
        print(seq_embeds.shape)

        inputs_embeds = torch.cat([prop_embeds, seq_embeds], dim=1) # b, t, d

        transformer_outputs = self.transformer(inputs_embeds=inputs_embeds)

        hidden_states = transformer_outputs[0]
        seq_hidden = hidden_states[:,:,:]
        props_hidden = hidden_states[:,-1,:]

        lm_logits = self.lm_head(seq_hidden)
        props_pred = self.prop_out_linear(props_hidden).squeeze(1) # b, num_props


        # 算loss
        loss = None
        if labels is not None:
            seq_label = labels[0]
            props_label = labels[1]
            shift_logits = lm_logits[..., 1:, :].contiguous()
            shift_labels = seq_label[..., 1:].contiguous()
            # Flatten the tokens
            loss_seq_fct = CrossEntropyLoss()
            loss_prop_fct = L1Loss()

            print(shift_logits.shape, shift_labels.shape)

            loss_seq = loss_seq_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss_prop = loss_prop_fct(props_pred, props_label)
            loss = loss_seq + self.alpha * loss_prop

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
    
if __name__=='__main__':
    config = ChemGPTConfig()
    model = ChemGPT(config)
    props = torch.randn([2, 3])
    seqs = torch.ones([2, 10]).long()
    labels = [seqs, props]
    data = [props, seqs[..., :-1]]

    output = model(data, labels)
    
    