#%%

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

from utils import get_item_to_ix, items_to_ix, mini_batch_sort_split_prepare
import random

#%%

data_test = [[str(int(i)) for i in range(4, random.randint(6, 25))] for j in range(3, random.randint(40, 50))]
vocab = get_item_to_ix(data_test)
data_test_ix = items_to_ix(data_test, vocab)

#%%

def get_loss(Y_hat, Y, X_lengths):
    # TRICK 3 ********************************
    # before we calculate the negative log likelihood, we need to mask out the activations
    # this means we don't want to take into account padded items in the output vector
    # simplest way to think about this is to flatten ALL sequences into a REALLY long sequence
    # and calculate the loss on that.

    # flatten all the labels
    #Y = Y.view(-1)

    # flatten all predictions
    Y_hat = Y_hat.view(-1, len(vocab)-1)

    print(Y_hat.size())

    # create a mask by filtering out all tokens that ARE NOT the padding token
    tag_pad_token = 0
    mask = (Y > tag_pad_token).float()

    # count how many tokens we have
    nb_tokens = int(torch.sum(mask).data[0])

    # pick the values for the label and zero out the rest with the mask
    Y_hat = Y_hat[range(Y_hat.shape[0]), Y] * mask

    # compute cross entropy loss which ignores all <PAD> tokens
    ce_loss = -torch.sum(Y_hat) / nb_tokens

    return ce_loss


#%%

LAYERS_DIM = 2
HIDDEN_DIM = 10
EMBEDDING_DIM = 6
MINI_BATCH_SIZE = 8


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, layers_dim, mini_batch_size, vocab_size):

        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.layers_dim = layers_dim
        self.mini_batch_size = mini_batch_size

        self.word_embeddings = nn.Embedding(vocab_size,
                                            embedding_dim,
                                            padding_idx=0)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, layers_dim)

        # The linear layer that maps from hidden state space to tag space
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(self.layers_dim, self.mini_batch_size, self.hidden_dim),
                torch.zeros(self.layers_dim, self.mini_batch_size, self.hidden_dim))

    def forward(self, x_input, x_lengths):

        print('x_input      : {}'.format(x_input.size()))

        x_input = self.word_embeddings(x_input)
        print('embeds       : {}'.format(x_input.size()))

        x_input = torch.transpose(x_input, 0, 1)
        print('embeds_T     : {}'.format(x_input.size()))

        print('h_t          : {}'.format(self.hidden[0].size()))
        print('h_c          : {}'.format(self.hidden[1].size()))

        x_input = torch.nn.utils.rnn.pack_padded_sequence(x_input, X_lengths)

        lstm_out, self.hidden = self.lstm(x_input, self.hidden)
            #embeds_T.view(len(sentence), 1, -1), self.hidden)

        #print('lstm_out     : {}'.format(lstm_out.size()))

        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)


        print('lstm_out     : {}'.format(lstm_out.size()))

        # X = X.contiguous()
        # X = X.view(-1, X.shape[2])
        # print('X.contiguous : {}'.format(X.size()))


        #tag_space = self.fc(lstm_out.view(len(sentence), -1))

        tag_space = self.fc(lstm_out) #self.fc(lstm_out)
        print('tag_space    : {}'.format(tag_space.size()))

        tag_scores = F.log_softmax(tag_space, dim=1)
        print('tag_scores   : {}'.format(tag_scores.size()))

        return tag_scores

#%%

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, LAYERS_DIM, MINI_BATCH_SIZE, len(vocab)-1)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for i in range(0, len(data_test), MINI_BATCH_SIZE):

    data_batch = data_test_ix[i:i+MINI_BATCH_SIZE]

    X_train_mb, Y_train_mb, X_lengths = mini_batch_sort_split_prepare(data_batch)

    # Step 1. Remember that Pytorch accumulates gradients.
    # We need to clear them out before each instance
    model.zero_grad()

    # Also, we need to clear out the hidden state of the LSTM,
    # detaching it from its history on the last instance.
    model.hidden = model.init_hidden()

    # Step 2. Get our inputs ready for the network, that is, turn them into
    # Tensors of word indices.
    X_train_mb = torch.LongTensor(X_train_mb)
    Y_train_mb = torch.LongTensor(Y_train_mb)
    Y_train_mb = torch.transpose(Y_train_mb, 0, 1)


    print('X_train_mb   : {}'.format(X_train_mb.size()))
    print('Y_train_mb   : {}'.format(Y_train_mb.size()))

    # Step 3. Run our forward pass.
    Y_hat = model(X_train_mb, X_lengths)

    # Step 4. Compute the loss, gradients, and update the parameters by
    #  calling optimizer.step()

    #loss = get_loss(Y_hat, Y_train_mb, X_lengths)
    loss = loss_function(Y_hat, Y_train_mb)
    # loss.backward()
    # optimizer.step()

    break








