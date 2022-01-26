import torch
import torch.nn as nn

from model import Seq2SeqModel, Decoder

class ConcatAttention(nn.Module):
    """
    Attention scores computed through a fully-connected layer
    """
    def __init__(self, hidden_size, enc_hidden_size, dec_hidden_size):
        super(ConcatAttention, self).__init__()
        self.enc_compr = torch.nn.Linear(enc_hidden_size, hidden_size, bias=False)
        self.dec_compr = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.attention_score = torch.nn.Sequential(*[
            torch.nn.Linear(hidden_size, 1, bias=False), # feed-forward nn ==> bias needed?
            torch.nn.Tanh(),
            torch.nn.Softmax(dim=1)
        ])

    def forward(self, h_t, h_enc):
        # h_t --> vector of previous state, of size (1, batch_size, hidden_size)
        # h_enc --> list of hidden vectors from encoder over the sequence, of size seq_len x (batch_size, hidden_size)
        seq_len = h_enc.shape[0]

        # putting batch dimension first
        h_enc = h_enc.permute(1,0,2)
        h_t = h_t.permute(1,0,2)

        compr_h_enc = self.enc_compr(h_enc)
        compr_h_t = self.dec_compr(h_t)
        compr_h_t = torch.cat([compr_h_t for j in range(seq_len)], dim=1) #concatenating along sequence length

        alphas = self.attention_score(compr_h_t + compr_h_enc)
        context = torch.sum(h_enc * alphas, dim=1).unsqueeze(0)
        return context, alphas


class GlobalAttention(nn.Module):
    """
    Attention using dot product as scoring function.
    """

    def __init__(self, decoder_hidden_size):
        super(GlobalAttention, self).__init__()
        self.hidden_size = decoder_hidden_size

        # final concatenation layer
        self.concat = torch.nn.Sequential(*[
            torch.nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False),
            torch.nn.Tanh()
        ])
    
    def forward(self, ht, h_enc):
        # ht --> (1, batch, hidden)
        # h_enc --> (sequence, batch, hidden)
        h_enc = h_enc.permute(1,0,2)
        ht = ht.permute(1,0,2) 
        alphas = torch.bmm(ht, h_enc.transpose(1,2)).softmax(dim=-1).transpose(1,2)
        context = torch.sum(h_enc * alphas, dim=1).unsqueeze(0)
        return context, alphas


class LocalAttention(nn.Module):
    def __init__(self, decoder_hidden_size, alignment_type='global', att_hidden_size=None):
        super(LocalAttention, self).__init__()
        self.hidden_size = decoder_hidden_size
        self.att_hidden_size = att_hidden_size if att_hidden_size is not None else decoder_hidden_size
        self.D = 3
        self.sigma = self.D / 2

        sw = {
            'concat': "ConcatAttention(self.hidden_size, self.hidden_size, self.hidden_size)",
            'global': 'GlobalAttention(self.hidden_size)'
        }
        self.alignment = eval(sw[alignment_type])

        self.compute_pt = nn.Sequential(*[
            nn.Linear(self.hidden_size, self.att_hidden_size, bias=False),
            nn.Tanh(),
            nn.Linear(self.att_hidden_size, 1, bias=False),
            nn.Sigmoid()
        ])
    
    def forward(self, h_t, h_enc, sequence_lengths):
        _, alphas = self.alignment(h_t, h_enc)
        
        pt = (sequence_lengths != 0).int().sum(dim=-1) * self.compute_pt(h_t).squeeze()
        num = (sequence_lengths - pt.unsqueeze(-1)) ** 2
        den = 2 * (torch.tensor(self.sigma) ** 2)
        coeff = torch.exp(-1 * (num / den))
        alphas = coeff * alphas.squeeze()
        context = torch.sum(h_enc.permute(1,0,2) * alphas.unsqueeze(-1), dim=1).unsqueeze(0)
        return context, alphas



class AttentionDecoder(Decoder):
    
    def __init__(self, input_size, output_size, attention_mode='fcn', **kwargs):
        super(AttentionDecoder, self).__init__(input_size, output_size, **kwargs)
        self.attention_mode = attention_mode

        if self.attention_mode == 'concat':
            # needs to readapt the model for concatenating embedding and context
            if self.use_embedder: decoder_input_size = self.embedder.weight.shape[1]
            else: decoder_input_size = self.output_size
            decoder_input_size = decoder_input_size + self.hidden_size
            self.model = eval(self.unit_map[self.unit_name])(decoder_input_size, self.hidden_size, num_layers=self.model.num_layers, batch_first=False)

        sw = {
            'concat': "ConcatAttention(self.hidden_size, self.hidden_size, self.hidden_size)",
            'local': 'LocalAttention(self.hidden_size)',
            'global': 'GlobalAttention(self.hidden_size)'
        }
        self.attention = eval(sw[attention_mode])
        self.global_attn_concat = torch.nn.Sequential(*[
            torch.nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False),
            torch.nn.Tanh()
        ])
    
    def forward(self, d_in, prev_h, h_encoder, sequence_lengths=None):
        if self.use_embedder: d_in = self.embedder(d_in)
        d_in = self.dropout(d_in)

        if self.unit_name == 'lstm': att_h = prev_h[0]
        else: att_h = prev_h
        
        if self.attention_mode == 'concat':
            context, alphas = self.attention(att_h, h_encoder)
            x = torch.cat((d_in, context), dim=-1)
            out, h = self.model(x, prev_h)
        elif self.attention_mode == 'global':
            out, h = self.model(d_in, prev_h)
            if att_h.shape[0] > 1: h_t = att_h[-1].unsqueeze(0)
            else: h_t = att_h
            context, alphas = self.attention(h_t, h_encoder) 
            out = self.global_attn_concat(torch.cat([out, context], dim=-1))
        elif self.attention_mode == 'local':
            out, h = self.model(d_in, prev_h)
            if att_h.shape[0] > 1: h_t = att_h[-1].unsqueeze(0)
            else: h_t = att_h
            context, alphas = self.attention(h_t, h_encoder, sequence_lengths)

        out = self.dropout(out)
        res = self.classifier(out.squeeze(0))
        return res.unsqueeze(0), h



class AttentionSeq2SeqModel(Seq2SeqModel):

    def __init__(self, attention_mode='fcn', max_seq_len=48, internal_dropout=0.0, **kwargs):
        super(AttentionSeq2SeqModel, self).__init__(**kwargs)
        self.attention_mode = attention_mode

        if self.decoder_input_mode == 'sentence':
            self.decoder = AttentionDecoder(self.input_size, self.output_size, hidden_size=self.decoder_hidden_size, 
                dropout=internal_dropout, num_layers=self.num_layers, 
                unit=self.unit_name, attention_mode=self.attention_mode
                ).to(self.device)

        elif self.decoder_input_mode == 'label_embed':
            # add one embedding vector for start of sentence 
            self.decoder = AttentionDecoder(self.output_size + 1, self.output_size, hidden_size=self.decoder_hidden_size, 
            dropout=internal_dropout, num_layers=self.num_layers, unit=self.unit_name, 
            use_embedder=True, attention_mode=self.attention_mode
            ).to(self.device)

        else:
            self.decoder = AttentionDecoder(self.output_size, self.output_size, hidden_size=self.decoder_hidden_size, 
            dropout=internal_dropout, num_layers=self.num_layers, 
            unit=self.unit_name, attention_mode=self.attention_mode
            ).to(self.device)

