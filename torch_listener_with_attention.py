import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch_model_base import TorchModelBase
from sklearn.model_selection import train_test_split
from torch_color_selector import (
    ColorizedNeuralListenerEncoder, ColorizedNeuralListener, ContextualColorDescriber, QuadraticForm, EncoderDecoder, Decoder, create_example_dataset)
import utils
from utils import START_SYMBOL, END_SYMBOL, UNK_SYMBOL
import time

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2020"


class AttentionalColorizedListenerEncoder(ColorizedNeuralListenerEncoder):
    '''
    Simple neural literal/pragmatic listener with dropout.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mew_layer = nn.Linear(#self.hidden_dim*2 + self.embed_dim, 
                                      self.hidden_dim,
                                      self.color_dim).to(self.device)
        self.sigma_layer = nn.Linear(#self.hidden_dim*2 + self.embed_dim, 
                                      self.hidden_dim,
                                      self.color_dim * self.color_dim).to(self.device)
        self.attention = nn.Linear(self.hidden_dim + self.embed_dim, 1)
        self.atten_softmax = nn.Softmax(dim=1)
        self.attention_dropout = nn.Dropout(self.dropout_prob)
        
    def forward(self, word_seqs, seq_lengths=None):

        embs = self.get_embeddings(word_seqs)

        # Packed sequence for performance:
        packed_embs = torch.nn.utils.rnn.pack_padded_sequence(
            embs, batch_first=True, lengths=seq_lengths, enforce_sorted=False)
        # RNN forward:
        output, hidden = self.rnn(packed_embs)
        # Unpack:
        output, seq_lengths = torch.nn.utils.rnn.pad_packed_sequence(
            output, batch_first=True)
        
        target_output = output[[i for i in range(output.shape[0])],seq_lengths-1]
        
        # Apply attention
        attn_scores = self.attention(torch.cat((embs, output), 2))
        attn_weights = self.atten_softmax(attn_scores).squeeze(2).unsqueeze(1)
        attn_output = torch.bmm(attn_weights, output).squeeze(1)
        attn_scores = self.attention_dropout(attn_output)
        #print(attn_output.shape, target_output.shape)
        
        # Combine all hidden states and feed into linear layer
        #hidden = [hidden[0]]
        #hidden_state = torch.cat(hidden, dim=2).squeeze(0)
        hidden_state = torch.cat([attn_output], dim=1)
        #hidden_state = output
        #print(hidden_state.shape)
        
        
        mew = self.mew_layer(hidden_state)
        #mew = self.mew_dropout(self.mew_hidden)
        
        self.sigma_hidden = self.sigma_layer(hidden_state)
        #sigma = self.sigma_dropout(self.sigma_hidden)
        sigma = self.sigma_hidden.view(-1, self.color_dim, self.color_dim)
        
        return target_output, mew, sigma
        
class AttentionalColorizedListenerDecoder(nn.Module):
    '''
    Simple decoder model for the neural literal/pragmatic listener.
    This model takes in two statistical params, mew and sigma, and returns a vector containing the normalized scores
    of each color in the context.
    '''
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.transform_func = QuadraticForm.apply
        self.hidden_activation = nn.Softmax(dim=1)
        
    def forward(self, color_seqs, mew, sigma):
        '''
        color_seqs : FloatTensor
            A m x k x n tensor where m is the number of examples, k is the number of colors in the context, and
            n is the size of the color dimension after transform
        '''
        color_scores = self.transform_func(mew, sigma, color_seqs)
        output = self.hidden_activation(color_scores)
        return output

class AttentionalColorizedListenerEncoderDecoder(EncoderDecoder):
    
    def forward(self, 
            color_seqs, 
            word_seqs, 
            seq_lengths=None, 
            mew=None, 
            sigma=None):
        if mew is None or sigma is None:
            _, mew, sigma = self.encoder(word_seqs, seq_lengths)
            
        output = self.decoder(
            color_seqs, mew=mew, sigma=sigma)
        
        return output

class AttentionalColorizedNeuralListener(ContextualColorDescriber):
    
    def __init__(self, *args, lr_rate=1., dropout_prob=0., **kwargs):
        super().__init__(*args, **kwargs)
        self.lr_rate = lr_rate
        self.cur_epoch = 0
        self.dropout_prob = dropout_prob

        self.transform_func = QuadraticForm.apply
        self.hidden_activation = nn.Softmax(dim=1)
        
    def build_graph(self):
        encoder = AttentionalColorizedListenerEncoder(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            embedding=self.embedding,
            hidden_dim=self.hidden_dim,
            color_dim=self.color_dim,
            dropout_prob=self.dropout_prob,
            device=self.device)

        decoder = AttentionalColorizedListenerDecoder(
            device=self.device)

        return AttentionalColorizedListenerEncoderDecoder(encoder, decoder, self.device)
    
    def fit(self, color_seqs, word_seqs):
        """Standard `fit` method where `word_seqs` are the inputs and
        `color_seqs` are the sequences to predict.

        Parameters
        ----------
        color_seqs : list of lists of lists of floats, or np.array
            Dimension (m, n, p) where m is the number of examples, n is
            the number of colors in each context, and p is the length
            of the color representations.
        word_seqs : list of list of int
            Dimension m, the number of examples. The length of each
            sequence can vary.

        Returns
        -------
        self

        """
        color_seqs_train, color_seqs_validate, word_seqs_train, word_seqs_validate = \
            train_test_split(color_seqs, word_seqs)
        
        self.color_dim = len(color_seqs[0][0])
        
        if not self.warm_start or not hasattr(self, "model"):
            self.model = self.build_graph()
            self.opt = self.optimizer(
                self.model.parameters(),
                lr=self.eta,
                weight_decay=self.l2_strength)
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opt, gamma=self.lr_rate)
            self.cur_epoch=0

        # Make sure that these attributes are aligned -- important
        # where a supplied pretrained embedding has determined
        # a `embed_dim` that might be different from the user's
        # argument.
        self.embed_dim = self.model.encoder.embed_dim

        self.model.to(self.device)

        self.model.train()

        dataset = self.build_dataset(color_seqs, word_seqs)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            collate_fn=dataset.collate_fn)

        loss = nn.CrossEntropyLoss()

        for iteration in range(self.cur_epoch + 1, self.cur_epoch + self.max_iter+1):
            epoch_error = 0.0
            start = time.time()
            for batch_colors, batch_words, batch_lens, _ in dataloader:
                
                batch_colors = batch_colors.to(self.device, non_blocking=True)
                
                batch_words = torch.nn.utils.rnn.pack_padded_sequence(
                    batch_words, batch_first=True, lengths=batch_lens, enforce_sorted=False)
                batch_words = batch_words.to(self.device, non_blocking=True)
                
                batch_lens = batch_lens.to(self.device, non_blocking=True)

                output = self.model(
                    color_seqs=batch_colors,
                    word_seqs=batch_words,
                    seq_lengths=batch_lens)
                
                color_targets = torch.ones(output.shape[0], dtype=torch.long) * 2
                color_targets = color_targets.to(self.device)
                err = loss(output, color_targets)
                
                epoch_error += err.item()
                self.opt.zero_grad()
                #self.model.encoder.rnn.weight_ih_l0.register_hook(lambda grad: print(grad))
                
                err.backward()
                self.opt.step()
            
            if iteration % 15 == 0:
                self.lr_scheduler.step()
                for param_group in self.opt.param_groups:
                    print(param_group["lr"])
                print(output.mean(dim=0), err.item())
                #print("output:", output.argmax(1).float())
                    
            print("Train: Epoch {}; err = {}; time = {}".format(iteration, epoch_error, time.time() - start))
            self.cur_epoch = self.cur_epoch + 1
        
        return self
    
    def predict(self, color_seqs, word_seqs, probabilities=False, verbose=False, max_length=20):
        
        self.model.to(self.device)

        self.model.eval()

        dataset = self.build_dataset(color_seqs, word_seqs)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=len(color_seqs),
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            collate_fn=dataset.collate_fn)

        loss = nn.CrossEntropyLoss()

        preds = []
        start = time.time()
        with torch.no_grad():
            for batch_colors, batch_words, batch_lens, _ in dataloader:

                batch_colors = batch_colors.to(self.device, non_blocking=True)

                batch_words = torch.nn.utils.rnn.pack_padded_sequence(
                    batch_words, batch_first=True, lengths=batch_lens, enforce_sorted=False)
                batch_words = batch_words.to(self.device, non_blocking=True)

                batch_lens = batch_lens.to(self.device, non_blocking=True)

                output = self.model(
                    color_seqs=batch_colors,
                    word_seqs=batch_words,
                    seq_lengths=batch_lens)

                color_targets = torch.ones(output.shape[0], dtype=torch.long) * 2
                color_targets = color_targets.to(self.device)
                err = loss(output, color_targets)
                #print(output.mean(dim=0), err.item())
        
        if verbose:
            print("Testing err = {}; time = {}".format(err.item(), time.time() - start))
        if not probabilities:
            output = output.argmax(1)
        p = output.cpu().detach().numpy()
        preds.extend(list(p))
        #print("new preds:", preds)

        return preds
    def save_model(self, path, inference_only=False):
        if inference_only:
            torch.save(self.model.state_dict(), path)
        else:
            torch.save(self.model, path)
        
    def load_model(self, path, inference_only=False, color_dim=None, **kwargs):
        if inference_only:
            if color_dim is None:
                raise AttributeError('When loading a state_dict, the color_dim must be passed')
            if self.model is None:
                self.model = literal_listener.build_graph()
                
            self.model.load_state_dict(torch.load(path, **kwargs))
            self.model.eval()
        else:
            self.model = torch.load(path, **kwargs)
            self.model.eval()

def create_example_dataset(group_size=100, vec_dim=2):
    """Creates simple datasets in which the inputs are three-vector
    sequences and the outputs are simple character sequences, with
    the range of values in the final vector in the input determining
    the output sequence. For example, a single input/output pair
    will look like this:

    [[0.44, 0.51], [0.87, 0.89], [0.1, 0.2]],  ['<s>', 'A', '</s>']

    The sequences are meaningless, as are their lengths (which were
    chosen only to be different from each other).

    """
    import random

    groups = ((0.0, 0.2), (0.4, 0.6), (0.8, 1.0))
    vocab = ['<s>', '</s>', 'A', 'B', '$UNK']
    seqs = [
        ['<s>', 'A', '</s>'],
        ['<s>', 'A', 'B', '</s>'],
        ['<s>', 'B', 'A', 'B', 'A', '</s>']]

    color_seqs = []
    word_seqs = []
    for i, ((l, u), seq) in enumerate(zip(groups, seqs)):

        dis_indices = list(range(len(groups)))
        dis_indices.remove(i)
        random.shuffle(dis_indices)
        disl1, disu1 = groups[dis_indices[0]]
        dis2 = disl2, disu2 = groups[dis_indices[1]]

        for _ in  range(group_size):
            target = utils.randvec(vec_dim, l, u)
            dis1 = utils.randvec(vec_dim, disl1, disu1)
            dis2 = utils.randvec(vec_dim, disl2, disu2)
            context = [dis1, dis2, target]
            color_seqs.append(context)

        word_seqs += [seq for _ in range(group_size)]

    return color_seqs, word_seqs, vocab

def simple_neural_listener_example(group_size=100, vec_dim=2, initial_embedding=False):
    from sklearn.model_selection import train_test_split

    color_seqs, word_seqs, vocab = create_example_dataset(
        group_size=group_size, vec_dim=vec_dim)

    if initial_embedding:
        import numpy as np
        embedding = np.random.normal(
            loc=0, scale=0.01, size=(len(vocab), 11))
    else:
        embedding = None

    X_train, X_test, y_train, y_test = train_test_split(
        color_seqs, word_seqs)

    mod = AttentionalColorizedNeuralListener(
        vocab,
        embed_dim=100,
        hidden_dim=101,
        max_iter=50,
        embedding=embedding)

    mod.fit(X_train, y_train)

    pred_indices = mod.predict(X_train, y_train)
    
    correct = 0
    for color_seq, pred_index in zip(y_train, pred_indices):
        target_index = len(color_seq[0]) - 1
        correct += int(target_index == pred_index)
    acc = correct / len(pred_indices)
    
    print("\nExact sequence: {} of {} correct. Accuracy: {}".format(correct, len(pred_indices), acc))

    return correct

if __name__ == '__main__':
    simple_neural_listener_example()