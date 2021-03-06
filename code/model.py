
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
        

    def forward(self, d_in, prev_h, h_encoder=None, sequence_lengths=None):
        # needed also h_enc and pad_mask for compatibility with attention
        if self.use_embedder: d_in = self.embedder(d_in)
        d_in = self.dropout(d_in)
        outs, h = self.model(d_in, prev_h)
        outs = self.dropout(outs)
        res = self.classifier(outs.squeeze(0))
        return res.unsqueeze(0), h
    

class Seq2SeqModel(nn.Module):

    def __init__(self, dataset_path='data/atis.train.pkl', 
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

        assert decoder_input_mode in ['sentence', 'label', 'label_nograd', 'label_embed', 'word+lab']
        self.decoder_input_mode = decoder_input_mode

        if self.decoder_input_mode == 'sentence':
            self.decoder = Decoder(self.input_size, self.output_size, hidden_size=self.decoder_hidden_size, dropout=internal_dropout, num_layers=num_layers, unit=unit_name).to(device)
        elif self.decoder_input_mode == 'label_embed':
            # add one embedding vector for start of sentence 
            self.decoder = Decoder(self.output_size + 1, self.output_size, hidden_size=self.decoder_hidden_size, dropout=internal_dropout, num_layers=num_layers, unit=unit_name, use_embedder=True).to(device)
        else:
            self.decoder = Decoder(self.output_size, self.output_size, hidden_size=self.decoder_hidden_size, dropout=internal_dropout, num_layers=num_layers, unit=unit_name).to(device)

        # combination layer required for 'word+lab' input mode
        self.wl_concat = nn.Sequential(*[
            nn.Linear(self.output_size + self.embedder.get_vec_size(), self.output_size, bias=False),
            nn.Tanh()
        ])
    
    def get_entities(self, path):
        with open(path, 'rb') as f:
            entities = list(pickle.load(f)[1]["slot_ids"].keys())
        return entities
    
    
    def decoder_forward_onestep(self, curr_x, curr_y, curr_gt, prev_h, h_enc, sequence_lengths=None):
        # in order to use this function also for attention, there are also arguments needed in that case

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
                # embedding needs indices
                if curr_y is None: d_in = self.output_size + torch.zeros(1, batch_size, device=self.device, dtype=torch.long) #custom index for beginning of sentence
                else: d_in = torch.argmax(curr_y, dim=-1)
            elif self.decoder_input_mode == 'word+lab':
                if curr_y is None: curr_y = self.output_size + torch.zeros(1, batch_size, self.output_size, device=self.device, dtype=torch.long)
                d_in = self.wl_concat(torch.cat((curr_y, curr_x.unsqueeze(0)), dim=-1))

        out, prev_dec_hidden = self.decoder(d_in, prev_h, h_enc, sequence_lengths)
        return out, prev_dec_hidden
    

    def forward(self, x, y=None, pad_mask=None):
        
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
        seq_len = pad_mask.shape[1]
        sequence_lengths = torch.arange(seq_len, dtype=torch.float32, device=self.device) * pad_mask.int() # obtain indices that are non-padding tokens 
        
        # looping over words (but keeping batch dimension)
        for word in range(x.shape[0]):
            if self.training and y is not None: 
                # initializes ground truth during training when not at start of sentence
                curr_gt = y[word].unsqueeze(0)
            out, prev_dec_hidden = self.decoder_forward_onestep(x[word], out, curr_gt, prev_dec_hidden, h_enc, sequence_lengths)
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

        sentence_lengths = pad_mask.int()
        
        logits = self(x, yt, pad_mask)

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
    


    def beam_inference(self, data, beam_width=5):

        sent, lab = data
        x, pad_mask, yt = self._embed_batch(sent, lab)

        batch_size = pad_mask.shape[0]
        seq_len = pad_mask.shape[1]

        x = x.permute(1,0,2) #needed to change shape to (sequence, batch, embedding)
        
        h_enc, h = self.encoder(x)
        if self.unit_name == 'lstm':
            # need to manage cell-state (c_t) vector
            c = h[1]
            h = h[0]
            if self.bidirectional:
                # concatenates forward and backward hidden states
                h = torch.stack([torch.cat((h[i, :, :], h[i+1, :, :]), dim=-1) for i in range(0, 2*self.num_layers, 2)], dim=0)
                c = torch.stack([torch.cat((c[i, :, :], c[i+1, :, :]), dim=-1) for i in range(0, 2*self.num_layers, 2)], dim=0)
            c = self.hidden_dropout(c)
            h = self.hidden_dropout(h)
            h = (h, c) 
        else:
            if self.bidirectional:
                # concatenates forward and backward hidden states
                h = torch.stack([torch.cat((h[i, :, :], h[i+1, :, :]), dim=-1) for i in range(0, 2*self.num_layers, 2)], dim=0)
            h = self.hidden_dropout(h) 

        x = x.permute(1,0,2)

        prev_dec_hidden = h
        batch_preds = []
        batch_scores = []

        for batch_index in range(batch_size):
            
            if self.unit_name == 'lstm': 
                prev_dec_hidden = (h[0][:, batch_index].unsqueeze(1).contiguous(), h[1][:, batch_index].unsqueeze(1).contiguous()) 
            else: 
                prev_dec_hidden = h[:, batch_index].unsqueeze(1).contiguous()
            batch_h_enc = h_enc[:, batch_index].unsqueeze(1).contiguous()
            sequence_lengths = (torch.arange(seq_len, dtype=float, device=self.device) * pad_mask.int()[batch_index]).unsqueeze(0) # obtain indices that are non-padding tokens 
            b_x = x[batch_index].unsqueeze(0).permute(1,0,2)

            beam_seq = [[] for b in range(beam_width)]
            beam_scores = torch.zeros(beam_width).to(self.device)
            best_beam_outs = tmp_beam_outs = [None for b in range(beam_width)]
            best_beam_hiddens = tmp_beam_hiddens = [prev_dec_hidden for b in range(beam_width)]
            
            for word in range(seq_len):
                new_beam_scores, new_beam_indices = [None for b in range(beam_width)], [None for b in range(beam_width)]
                for b in range(beam_width): 
                    tmp_beam_outs[b], tmp_beam_hiddens[b] = self.decoder_forward_onestep(b_x[word].unsqueeze(0), tmp_beam_outs[b], None, tmp_beam_hiddens[b], batch_h_enc, sequence_lengths)
                    new_beam_scores[b], new_beam_indices[b] = tmp_beam_outs[b].topk(beam_width, dim=-1)
                    
                    new_beam_scores[b].log_().squeeze_()
                    new_beam_indices[b].squeeze_()
                
                # BEAM UPDATE
                if len(beam_seq[0]) == 0:
                    # handle this case separately to avoid choosing same index multiple times (because of the top scores)
                    for b in range(beam_width):
                        beam_seq[b].append(new_beam_indices[b][b].item())
                        beam_scores[b] = new_beam_scores[b][b]
                else:
                    # 1. generating combination of current beam scores + candidate scores (since we use log scores)
                    combination_matrix = torch.stack(new_beam_scores) + beam_scores.view(-1, 1)
                    # 2. retrieving linearized indices of best product values
                    best_combinations = combination_matrix.flatten().argsort(descending=True)[:beam_width]
                    # 3. retrieving (row, col) indices for the best scores as combination of row-oldbeam / col-newindex
                    row, col = torch.div(best_combinations, beam_width, rounding_mode='trunc').tolist(), (best_combinations % beam_width).tolist()
                    # 4. actual beam update
                    new_beam_seq = []
                    for b in range(beam_width):
                        # updates predictions
                        new_beam_seq.append(beam_seq[row[b]] + [new_beam_indices[row[b]][col[b]].item()])
                        # updates scores
                        beam_scores[b] = combination_matrix[row[b], col[b]]
                        # updates outputs bookeep
                        best_beam_outs[b] = tmp_beam_outs[col[b]]
                        # updates hidden states bookeep
                        best_beam_hiddens[b] = tmp_beam_hiddens[col[b]]
                    beam_seq = new_beam_seq
            
            beam_preds = [self.convert_label(seq) for seq in beam_seq]
            for b in range(beam_width):
                # performs unpadding for current sentence in batch
                beam_preds[b] = [beam_preds[b][j] for j in range(seq_len) if pad_mask[batch_index][j].item()]
            
            batch_preds.append(beam_preds)
            batch_scores.append(beam_scores.tolist())
        
        
        unpadded_sent = [[sent[i][j] for j in range(seq_len) if pad_mask[i][j].item()] for i in range(batch_size)]
        unpadded_lab = [[lab[i][j] for j in range(seq_len) if pad_mask[i][j].item()] for i in range(batch_size)]

        return unpadded_sent, batch_preds, batch_scores, unpadded_lab
    

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

    def wrong_beam_inference(self, data, beam_width):
        sent, lab = data
        x, pad_mask, yt = self._embed_batch(sent, lab)

        batch_size = pad_mask.shape[0]
        seq_len = pad_mask.shape[1]

        x = x.permute(1,0,2) #needed to change shape to (sequence, batch, embedding)
        
        h_enc, h = self.encoder(x)
        if self.unit_name == 'lstm':
            # need to manage cell-state (c_t) vector
            c = h[1]
            h = h[0]
            if self.bidirectional:
                # concatenates forward and backward hidden states
                h = torch.stack([torch.cat((h[i, :, :], h[i+1, :, :]), dim=-1) for i in range(0, 2*self.num_layers, 2)], dim=0)
                c = torch.stack([torch.cat((c[i, :, :], c[i+1, :, :]), dim=-1) for i in range(0, 2*self.num_layers, 2)], dim=0)
            c = self.hidden_dropout(c)
            h = self.hidden_dropout(h)
            h = (h, c) 
        else:
            if self.bidirectional:
                # concatenates forward and backward hidden states
                h = torch.stack([torch.cat((h[i, :, :], h[i+1, :, :]), dim=-1) for i in range(0, 2*self.num_layers, 2)], dim=0)
            h = self.hidden_dropout(h) 

        prev_dec_hidden = h
        sequence_lengths = torch.arange(seq_len, dtype=float, device=self.device) * pad_mask.int()

        # performs 1st step of decoding to init beams
        out, prev_dec_hidden = self.decoder_forward_onestep(x[0], None, None, prev_dec_hidden, h_enc, sequence_lengths)
        beam_scores, beam_seq = out.squeeze().topk(beam_width, dim=-1)
        best_beam_outs = [torch.stack([torch.zeros(1,self.output_size, device=self.device).index_fill_(-1, beam_seq[_batch, b], torch.tensor(1)) for _batch in range(batch_size)], dim=1) for b in range(beam_width)]
        best_beam_hiddens = tmp_beam_hiddens = [prev_dec_hidden for b in range(beam_width)]

        for word in range(1, seq_len):
            tmp_beam_outs = [None for b in range(beam_width)]
            tmp_beam_hiddens = [None for b in range(beam_width)]
            new_beam_scores, new_beam_indices = [None for b in range(beam_width)], [None for b in range(beam_width)]
            
            for b in range(beam_width): 
                tmp_beam_outs[b], tmp_beam_hiddens[b] = self.decoder_forward_onestep(x[word], best_beam_outs[b], None, best_beam_hiddens[b], h_enc, sequence_lengths)
                new_beam_scores[b], new_beam_indices[b] = tmp_beam_outs[b].topk(beam_width, dim=-1)
                
                new_beam_scores[b].log_().squeeze_()
                new_beam_indices[b].squeeze_()

            # row: old beam | col: new value for each old beam 
            new_beam_scores = torch.stack(new_beam_scores, dim=1)
            new_beam_indices = torch.stack(new_beam_indices, dim=1)

            # 1. generating combination of current beam scores + candidate scores (since we use log scores)
            combination_matrix = beam_scores.view(batch_size, beam_width, 1) + new_beam_scores
            # 2. retrieving linearized indices of best product values
            best_combinations = combination_matrix.view(batch_size, -1).argsort(dim=-1, descending=True)[:, :beam_width] # --> (batch_size, beam_width)
            # 3. retrieving (row, col) indices for the best scores as combination of row-oldbeam / col-newindex
            row, col = torch.div(best_combinations, beam_width, rounding_mode='trunc'), (best_combinations % beam_width)
            # 4. beam update
            beam_scores = torch.tensor([[ new_beam_scores[_batch, row[_batch, b], col[_batch, b]].item() for b in range(beam_width)] for _batch in range(batch_size)], device=self.device)
            beam_seq = beam_seq.view(batch_size, beam_width, -1)
            # composes back as (batch_size, beam_seq, current length)
            beam_seq = torch.stack([
                    torch.stack([
                        torch.cat((
                            beam_seq[_batch, row[_batch, b]], 
                            new_beam_indices[_batch, row[_batch, b], col[_batch, b]].view(1)
                            ), dim=-1) 
                            for b in range(beam_width)
                    ], dim=0) 
                for _batch in range(batch_size)
            ], dim=0).to(self.device)

            tmp_beam_outs = torch.stack(tmp_beam_outs, dim=1).squeeze().view(batch_size, beam_width, -1)
            if self.unit_name == 'lstm':
                lstm_h = torch.stack([el[0] for el in tmp_beam_hiddens], dim=1).squeeze().view(batch_size, beam_width, -1)
                lstm_c = torch.stack([el[1] for el in tmp_beam_hiddens], dim=1).squeeze().view(batch_size, beam_width, -1)
                best_lstm_h = [torch.stack([lstm_h[_batch, col[_batch, b]] for _batch in range(batch_size)], dim=0).unsqueeze(0) for b in range(beam_width)]
                best_lstm_c = [torch.stack([lstm_c[_batch, col[_batch, b]] for _batch in range(batch_size)], dim=0).unsqueeze(0) for b in range(beam_width)]
                best_beam_hiddens = list(zip(best_lstm_h, best_lstm_c))
            else:
                tmp_beam_hiddens = torch.stack(tmp_beam_hiddens, dim=1).squeeze().view(batch_size, beam_width, -1)
                best_beam_hiddens = [torch.stack([tmp_beam_hiddens[_batch, col[_batch, b]] for _batch in range(batch_size)], dim=0).unsqueeze(0) for b in range(beam_width)]
            best_beam_outs = [torch.stack([tmp_beam_outs[_batch, col[_batch, b]] for _batch in range(batch_size)], dim=0).unsqueeze(0) for b in range(beam_width)]

        unpadded_sent = [[sent[i][j] for j in range(seq_len) if pad_mask[i][j].item()] for i in range(batch_size)]
        unpadded_lab = [[lab[i][j] for j in range(seq_len) if pad_mask[i][j].item()] for i in range(batch_size)]

        return unpadded_sent, beam_seq.tolist(), beam_scores.tolist(), unpadded_lab
