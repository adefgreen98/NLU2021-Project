import torch
import torch.nn as nn

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
        
        self.out = nn.Softmax(dim=-1) # TODO: tutorial uses LogSoftmax ??

    def forward(self, x, h_encoder):
        h = self.model(x, h_encoder)
        outs = h[0] #excluding decoder final hidden state

        o = self.out(self.classifier(outs))

        # for b in range(o.shape[0]): #looping over batch
        #     for i in range(o.shape[1]): #looping over sequence length
        #         o[b][i] = self.out(self.classifier(outs[b][i]))
        
        return o

"""## Seq2Seq

TODO:
- Appears that GRU returns all layers' final encoding; just taking the last one for now, but could have a projection head that mixes them
"""

class Seq2SeqModel(nn.Module):

    def __init__(self, unit_name, lab2idx, input_size, num_layers=2, hidden_size=256, device='cuda'):
        super(Seq2SeqModel, self).__init__()
        self.lab2idx = lab2idx
        self.idx2lab = {idx: lab for lab, idx in lab2idx.items()}
        self.device = device
        
        output_size = len(self.lab2idx)
        
        self.input_size = input_size

        self.encoder = Encoder(input_size, hidden_size=hidden_size, num_layers=num_layers, unit=unit_name).to(device)
        self.decoder = Decoder(input_size, output_size, hidden_size=hidden_size, num_layers=num_layers, unit=unit_name).to(device)

        self.embedder = None # needed for converting words to vectors during inference

    # Forward methods
    
    def forward(self, x):
        h = self.encoder(x)

        h = h[-1] #TODO: understand which output do we need to use

        out_probs = self.decoder(x, h)
        return out_probs
    
    
    def process_batch(self, data, loss_fn):
        x, yt = data
        x = x.to(self.device)
        yt = yt.to(self.device)
        yp = self(x)
        loss = [loss_fn(yp[:, i], yt[:, i]) for i in range(yp.shape[1])]
        loss = torch.stack(loss)
        loss = torch.sum(loss) #TODO: use mean instead of sum?

        acc_labels_list = [yp.argmax(dim=-1).flatten().tolist(), yt.flatten().tolist()]

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
        return [self.idx2lab[i] for i in v]
    
    def run_inference(self, sentence):
        """
        Returns the sequence of predicted tags for a sentence. Only works if an 
        embedder function was previously defined.
        """
        if self.embedder is None:
            print("Model Error: no embedder function has been set on this model")
            return None
        else:
            x = self.embedder(sentence).unsqueeze(0).to(self.device)
            probs = self.forward(x).squeeze(0)
            # return self.convert_label(self.get_label_idx(probs))[1:len(sentence.split())+1] # excluding EOS, SOS and padding 
            return self.convert_label(self.get_label_idx(probs))

    def set_embedder(self, embedder):
        """ 
        Sets the embedder **function** used by the model at inference time; this should be the same embedding function
        of the dataset used during training (to mantain consistent sizes and padding methods). 
        This is needed when testing / deploying the model. 
        
        :param embedder: a callable (str -> torch.Tensor), mapping a string into a tensor that can 
        be managed by a model.
        """
        self.embedder = embedder