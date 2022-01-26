
import pickle
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from nlp_init import get_embedder


def apply_padding(sentence, label=None, padded_size=None, right_pad=True):
    
    res_sentence = sentence.copy()

    if sentence[0] != 'BOS': res_sentence.insert(0, 'BOS')
    if sentence[-1] != 'EOS': res_sentence.append('EOS')

    res_sentence[0] = '<BOS>'
    res_sentence[-1] = '<EOS>'
    
    if label is not None:
        res_label = label.copy()
        if len(res_label) < len(res_sentence):
            res_label.insert(0, 'O')
            res_label.append('O')
    
    if padded_size is not None:
        to_pad = padded_size - len(sentence)

        if right_pad:
            #right padding
            res_sentence = res_sentence + ['<PAD>' for i in range(to_pad)] 
            res_label = res_label + ['O' for i in range(to_pad)]
        else: 
            #left padding
            res_sentence = ['<PAD>' for i in range(to_pad)] + res_sentence
            res_label = ['O' for i in range(to_pad)] + res_label

    if label is None: return res_sentence
    else: return res_sentence, res_label
        

class Encoder(nn.Module):

    unit_map = {
        "gru": "nn.GRU",
        "lstm": "nn.LSTM",
        "rnn": "nn.RNN"
    }

    def __init__(self, input_size, unit="gru", num_layers=2, hidden_size=256, dropout=0.5, bidirectional=False):
        super(Encoder, self).__init__()
        self.model = eval(Encoder.unit_map[unit])(input_size, hidden_size, num_layers=num_layers, batch_first=False, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # retrurns both h vector over sequence and h_n (last)
        x = self.dropout(x)
        return self.model(x)


class Decoder(nn.Module):
    unit_map = {
        "gru": "nn.GRU",
        "lstm": "nn.LSTM",
        "rnn": "nn.RNN"
    }

    def __init__(self, input_size, output_size, unit="gru", num_layers=2, hidden_size=256, dropout=0.5, use_embedder=False):
        super(Decoder, self).__init__()

        self.output_size = output_size
        self.use_embedder = use_embedder
        self.unit_name = unit
        if self.use_embedder:
            emb_size = int(input_size * 0.75)
            self.embedder = nn.Embedding(input_size, emb_size)
            input_size = emb_size
        self.hidden_size = hidden_size
        self.model = eval(Decoder.unit_map[unit])(input_size, hidden_size, num_layers=num_layers, batch_first=False)
        self.classifier = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, d_in, prev_h, h_encoder=None):
        # needed also h_enc for compatibility with attention
        if self.use_embedder: d_in = self.embedder(d_in)
        d_in = self.dropout(d_in)
        outs, h = self.model(d_in, prev_h)
        outs = self.dropout(outs)
        res = self.classifier(outs.squeeze(0))
        return res.unsqueeze(0), h
    

class Seq2SeqModel(nn.Module):

    def __init__(self, dataset_path='iob_atis/atis.train.pkl', 
        embedding_method='glove', unit_name='gru',  
        num_layers=2, hidden_size=256, 
        intermediate_dropout=0.0, internal_dropout=0.0,
        device='cuda', decoder_input_mode='label_nograd', bidirectional=False,
        embedding_size=300, **kwargs):

        super(Seq2SeqModel, self).__init__()
        self.unit_name = unit_name
        self.device = device
        self.bidirectional = bidirectional

        self.idx2tag = self.get_entities(dataset_path)
        self.tag2idx = {k: i for i, k in enumerate(self.idx2tag)}

        self.embedder = get_embedder(embedding_method, vec_size=embedding_size, dataset_path=dataset_path)

        self.input_size = self.embedder.get_vec_size()
        self.output_size = len(self.tag2idx)
        self.num_layers = num_layers
        
        if self.bidirectional: self.encoder_hidden_size = hidden_size // 2
        else: self.encoder_hidden_size = hidden_size
        self.decoder_hidden_size = hidden_size

        self.encoder = Encoder(self.input_size, hidden_size=self.encoder_hidden_size, num_layers=num_layers, dropout=internal_dropout, unit=unit_name, bidirectional=bidirectional).to(device)
        self.hidden_dropout = nn.Dropout(p=intermediate_dropout)

        self.teacher_forcing_rate = 0.5

        assert decoder_input_mode in ['sentence', 'label', 'label_nograd', 'label_embed']
        self.decoder_input_mode = decoder_input_mode

        if self.decoder_input_mode == 'sentence':
            self.decoder = Decoder(self.input_size, self.output_size, hidden_size=self.decoder_hidden_size, dropout=internal_dropout, num_layers=num_layers, unit=unit_name).to(device)
        elif self.decoder_input_mode == 'label_embed':
            # add one embedding vector for start of sentence 
            self.decoder = Decoder(self.output_size + 1, self.output_size, hidden_size=self.decoder_hidden_size, dropout=internal_dropout, num_layers=num_layers, unit=unit_name, use_embedder=True).to(device)
        else:
            self.decoder = Decoder(self.output_size, self.output_size, hidden_size=self.decoder_hidden_size, dropout=internal_dropout, num_layers=num_layers, unit=unit_name).to(device)
    
    
    def get_entities(self, path):
        with open(path, 'rb') as f:
            entities = list(pickle.load(f)[1]["slot_ids"].keys())
        return entities
    
    
    def decoder_forward_onestep(self, curr_x, curr_y, curr_gt, prev_h, h_enc):
        
        batch_size = curr_x.shape[0]

        # defining decoder modality
        if self.decoder_input_mode == 'sentence':
            d_in = curr_x.unsqueeze(0)
        else:
            if self.training and torch.rand(1).item() < self.teacher_forcing_rate:
                if curr_y is not None:
                    # excluding beginning of sequence and transforming in 1-hot if needed
                    curr_y = torch.zeros(1, batch_size, self.output_size, device=self.device).index_fill(-1, curr_gt.squeeze(), 1)
            
            if self.decoder_input_mode == 'label': 
                #directly feed previous output
                if curr_y is None: d_in = torch.zeros(1, batch_size, self.output_size, device=self.device)
                else: d_in = curr_y.softmax(dim=-1)
            elif self.decoder_input_mode == 'label_nograd': 
                #creates a new label as 1-hot vector
                if curr_y is None: idx = torch.tensor([self.tag2idx['O'] for b in range(batch_size)]).to(self.device)
                else: idx = curr_y.argmax(-1).squeeze()
                d_in = torch.zeros(1, batch_size, self.output_size, device=self.device).index_fill(-1, idx, 1)
            elif self.decoder_input_mode == 'label_embed': 
                #uses an embedding layer also for labels
                if curr_y is None: d_in = self.output_size + torch.zeros(1, batch_size, device=self.device, dtype=torch.long) #custom index for beginning of sentence
                else: d_in = torch.argmax(curr_y, dim=-1)

        out, prev_dec_hidden = self.decoder(d_in, prev_h, h_enc)
        return out, prev_dec_hidden
    

    def forward(self, x, y=None):
        
        x = x.permute(1,0,2) #needed to change shape to (sequence, batch, embedding)
        if y is not None: 
            y = y.permute(1,0)
        h_enc, h = self.encoder(x) 

        if self.unit_name == 'lstm':
            # need to manage cell-state (c_t) vector
            c = h[1]
            h = h[0]
            if self.bidirectional:
                h = torch.stack([torch.cat((h[i, :, :], h[i+1, :, :]), dim=-1) for i in range(0, 2*self.num_layers, 2)], dim=0)
                c = torch.stack([torch.cat((c[i, :, :], c[i+1, :, :]), dim=-1) for i in range(0, 2*self.num_layers, 2)], dim=0)
            c = self.hidden_dropout(c)
            h = self.hidden_dropout(h)
            h = (h, c) 
        else:
            if self.bidirectional:
                h = torch.stack([torch.cat((h[i, :, :], h[i+1, :, :]), dim=-1) for i in range(0, 2*self.num_layers, 2)], dim=0)
            h = self.hidden_dropout(h) 

        out_probs = []
        prev_dec_hidden = h
        out = None
        curr_gt = None

        # looping over words (but keeping batch dimension)
        for word in range(x.shape[0]):
            if self.training and y is not None: 
                curr_gt = y[word].unsqueeze(0)
            out, prev_dec_hidden = self.decoder_forward_onestep(x[word], out, curr_gt, prev_dec_hidden, h_enc)
            out_probs.append(out.squeeze().unsqueeze(0))

        out_probs = torch.cat(out_probs, dim=0) 
        if len(out_probs.shape) > 2: 
            out_probs = out_probs.permute(1,0,2) #putting again to (batch, sequence, embedding) when in batch mode

        return out_probs

    
    def process_batch(self, data, loss_fn):
        sent, lab = data
        x, pad_mask, yt = self._embed_batch(sent, lab)

        x = x.to(self.device)
        yt = yt.to(self.device)
        
        logits = self(x, yt)

        if len(logits.shape) < 3: logits.unsqueeze_(0) #case batch_size == 1

        # storing results before gradienting
        yp = logits.argmax(dim=-1)
        res_yp = yp[pad_mask].flatten().tolist()
        res_yt = yt[pad_mask].flatten().tolist()
        
        # Zero-ing padding tokens
        logits = logits * pad_mask.int().unsqueeze(-1)
        yt = yt * pad_mask.int()

        # loss calculation per each item in the sequence (over batch size)
        loss = [(loss_fn(logits[:, i], yt[:, i])) for i in range(logits.shape[1])]
        loss = torch.stack(loss)
        loss = torch.mean(loss) #TODO: use mean instead of sum?

        return loss, res_yp, res_yt
    

    def get_label_idx(self, sent_probs):
        """
        Computes label indices from raw model outputs (tag probabilities for each token).
        """
        return sent_probs.argmax(dim=-1).tolist()
    

    def convert_label(self, v):
        """
        Converts a sequence of indexes (ordered as the sample sentence) into readable labels.
        """
        return [self.idx2tag[i] for i in v]


    def run_inference(self, sent, lab):
        padded_s, x, yt = self._embed_single(sent, lab)
        probs = self(x.unsqueeze(0).to(self.device)).softmax(dim=-1)
        res_yp = self.convert_label(self.get_label_idx(probs))

        assert len(res_yp) == len(yt)

        return padded_s, res_yp, yt
    

    def beam_inference(self, batch_sent, batch_label, beam_width=5):

        x = x.permute(1,0,2) #needed to change shape to (sequence, batch, embedding)
        if y is not None: y = y.permute(1,0)
        
        h_enc, h = self.encoder(x)
        if self.unit_name == 'lstm':
            # need to manage cell-state (c_t) vector
            c = h[1]
            h = h[0]
            if self.bidirectional:
                h = torch.stack([torch.cat((h[i, :, :], h[i+1, :, :]), dim=-1) for i in range(0, 2*self.num_layers, 2)], dim=0)
                c = torch.stack([torch.cat((c[i, :, :], c[i+1, :, :]), dim=-1) for i in range(0, 2*self.num_layers, 2)], dim=0)
            c = self.hidden_dropout(c)
            h = self.hidden_dropout(h)
            h = (h, c) 
        else:
            if self.bidirectional:
                h = torch.stack([torch.cat((h[i, :, :], h[i+1, :, :]), dim=-1) for i in range(0, 2*self.num_layers, 2)], dim=0)
            h = self.hidden_dropout(h) 
        
        prev_dec_hidden = h
        out = None

        beam_seq = [[] for b in range(beam_width)]
        beam_scores = torch.zeros(beam_width).to(self.device)
        for word in range(x.shape[0]):
            out, prev_dec_hidden = self.decoder_forward_onestep(x[word], out, None, prev_dec_hidden, h_enc)
            out = out.softmax(dim=-1)
            
            # CANDIDATES GENERATION
            new_beam_scores, new_beam_indices = out.topk(beam_width, dim=-1)
            new_beam_scores.log_().squeeze_()
            new_beam_indices.squeeze_()

            # BEAM UPDATE
            if len(beam_seq[0]) == 0:
                # handle this case separately to avoid choosing same index multiple times (because of the top scores)
                for b in range(beam_width):
                    beam_seq[b].append(new_beam_indices[b].item())
                    beam_scores[b] = new_beam_scores[b]
            else:
                # 1. generating combination of current beam scores + candidate scores (since we use log scores)
                combination_matrix = beam_scores.view(-1, 1) + new_beam_scores
                # 2. retrieving linearized indices of best product values
                best_combinations = combination_matrix.flatten().argsort(descending=True)[:beam_width]
                # 3. retrieving (row, col) indices for the best scores
                row, col = torch.div(best_combinations, beam_width, rounding_mode='trunc').tolist(), (best_combinations % beam_width).tolist()
                # 4. actual beam update
                new_beam_seq = []
                for b in range(beam_width):
                    new_beam_seq.append(beam_seq[row[b]] + [new_beam_indices[col[b]].item()])
                    beam_scores[b] = combination_matrix[row[b], col[b]]
                beam_seq = new_beam_seq

        return padded_s, [self.convert_label(seq) for seq in beam_seq], beam_scores.tolist(), yt
    

    # Private methods

    def _embed_batch(self, b_sent, b_lab):
        batch_max_length = max([len(sample) for sample in b_sent])
        embedded = [apply_padding(b_x, b_y, padded_size=batch_max_length) for b_x, b_y in zip(b_sent, b_lab)]
        x = [sent for sent, lab in embedded]
        pad_mask = torch.tensor([[w != '<PAD>' for w in sent] for sent in x]).cuda()
        y = [lab for sent, lab in embedded]
        x = torch.stack([self.embedder.get_sent(b) for b in x], dim=0)
        y = torch.stack([torch.tensor([self.tag2idx[tag] for tag in b], dtype=torch.long) for b in y], dim=0)
        return x, pad_mask, y
    

    def _embed_single(self, sent, lab=None):
        x, y = apply_padding(sent, lab)
        x_vec = self.embedder.get_sent(x).squeeze()
        return x, x_vec, y

