import torch
import torch.nn as nn
from torch.nn.functional import softmax

"""# Sequence Model

## Encoder & Decoder
"""

class Encoder(nn.Module):

    unit_map = {
        "gru": "nn.GRU",
        "lstm": "nn.LSTM",
        "rnn": "nn.RNN"
    }

    def __init__(self, input_size, unit="gru", num_layers=2, hidden_size=256):
        super(Encoder, self).__init__()
        self.model = eval(Encoder.unit_map[unit])(input_size, hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        return self.model(x)

"""Notes for Decoder:
* In reference code this is not initialized to W2V and learned through the process
"""

class Decoder(nn.Module):
    unit_map = {
        "gru": "nn.GRU",
        "lstm": "nn.LSTM",
        "rnn": "nn.RNN"
    }

    def __init__(self, input_size, output_size, unit="gru", num_layers=2, hidden_size=256):
        super(Decoder, self).__init__()

        self.output_size = output_size
        self.model = eval(Decoder.unit_map[unit])(input_size, hidden_size, num_layers=num_layers, batch_first=True)

        self.classifier = nn.Sequential(*[
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        ])
        

    def forward(self, x, h_encoder):
        h = self.model(x, h_encoder)
        outs = h[0] #excluding decoder final hidden state

        return self.classifier(outs)

"""## Seq2Seq

TODO:
- Appears that GRU returns all layers' final encoding; just taking the last one for now, but could have a projection head that mixes them
"""

class Seq2SeqModel(nn.Module):

    def __init__(self, embedder, unit_name='gru', num_layers=2, hidden_size=256, dropout_p=0.0, device='cuda'):
        super(Seq2SeqModel, self).__init__()
        
        self.embedder = embedder # needed for converting words to vectors during inference
        self.device = device
        self.input_size = self.embedder.get_embedding_size()
        self.output_size = len(self.embedder.get_entities())

        self.encoder = Encoder(self.input_size, hidden_size=hidden_size, num_layers=num_layers, unit=unit_name).to(device)
        self.hidden_dropout = nn.Dropout(p=dropout_p)
        self.decoder = Decoder(self.input_size, self.output_size, hidden_size=hidden_size, num_layers=num_layers, unit=unit_name).to(device)


    # Forward methods
    
    def forward(self, x):
        h = self.encoder(x)

        #TODO: understand which output do we need to use
        h = self.hidden_dropout(h[-1]) 

        out_probs = self.decoder(x, h)
        return out_probs
    
    
    def process_batch(self, data, loss_fn):
        x, yt = data
        x, yt = self.embedder(x, yt, _batch=True)

        x = x.to(self.device)
        yt = yt.to(self.device)
        
        logits = self(x)

        # computing mask for excluding padded tokens from accuracy computation
        pad_mask = torch.all(x != -1, dim=-1)

        # storing results before gradienting
        yp = logits.argmax(dim=-1)
        yp_no_padding = yp[pad_mask].flatten().tolist()
        yt_no_padding = yt[pad_mask].flatten().tolist()
        acc_labels_list = [yp_no_padding, yt_no_padding]
        
        # loss calculation per each item in the sequence (over batch size)
        loss = [loss_fn(logits[:, i], yt[:, i]) for i in range(yt.shape[1])] #needed <listcomp> because loss does not understand sequence data
        loss = torch.stack(loss)
        loss = torch.sum(loss) #TODO: use mean instead of sum?

        return loss, acc_labels_list


    # Testing methods
    
    def get_label_idx(self, sent_probs):
        """
        Computes label indices from raw model outputs (tag probabilities for each token).
        """
        return sent_probs.argmax(dim=-1).tolist()
    

    def convert_label(self, v):
        """
        Converts a sequence of indexes (ordered as the sample sentence) into readable labels.
        """
        return [self.embedder.idx2tag[i] for i in v]
    

    def run_inference(self, sentence):
        """
        Returns the sequence of predicted tags for a sentence. 
        """
        if self.embedder is None:
            raise RuntimeError("no embedder function has been set for testing this model. Please use the same of the training set.")
        else:
            x = self.embedder(sentence).unsqueeze(0).to(self.device)
            probs = self.forward(x).squeeze(0)
            # return self.convert_label(self.get_label_idx(probs))[1:len(sentence.split())+1] # excluding EOS, SOS and padding 
            return self.convert_label(self.get_label_idx(probs))


    def beam_inference(self, sentence, beam_width=5):
        if self.embedder is None:
            raise RuntimeError("no embedder function has been set for testing this model. Please use the same of the training set.")
        else:
            x = self.embedder(sentence).unsqueeze(0).to(self.device)
            probs = softmax(self.forward(x).squeeze(0), dim=-1)

            # initializes beam scores
            init_beam = probs[0].sort(descending=True)
            beam = {"score": init_beam[0][:beam_width], "seq": [[el] for el in init_beam[1][:beam_width].tolist()]}

            # looping along sequence length
            for i in range(1, probs.shape[0]):
                curr_scores = probs[i]
                new_indices = torch.arange(curr_scores.shape[0])
                
                combination_matrix = beam["score"].view(-1,1) * curr_scores

                best_combinations = combination_matrix.flatten().argsort(descending=True)[:beam_width]
                x,y = torch.div(best_combinations, new_indices.shape[0], rounding_mode='trunc'), best_combinations % new_indices.shape[0]

                beam["score"] = combination_matrix[x,y]
                beam["seq"] = [beam["seq"][xi] + [new_indices[yi].item()] for xi, yi in zip(x.tolist(),y.tolist())] #selecting only winning combinations
            
            return [self.convert_label(seq) for seq in beam["seq"]], beam["score"].tolist()



    # Others

    def set_embedder(self, embedder):
        """ 
        Sets the embedder **function** used by the model at inference time; this should be the same embedding function
        of the dataset used during training (to mantain consistent sizes and padding methods). 
        This is needed when testing / deploying the model. 
        
        :param embedder: a callable (str -> torch.Tensor), mapping a string into a tensor that can 
        be managed by a model.
        """
        self.embedder = embedder
