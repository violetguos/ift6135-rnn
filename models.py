import torch 
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt

# NOTE ==============================================
#
# Fill in code for every method which has a TODO
#
# Your implementation should use the contract (inputs
# and outputs) given for each model, because that is 
# what the main script expects. If you modify the contract, 
# you must justify that choice, note it in your report, and notify the TAs 
# so that we run the correct code.
#
# You may modify the internals of the RNN and GRU classes
# as much as you like, except you must keep the methods
# in each (init_weights_uniform, init_hidden, and forward)
# Using nn.Module and "forward" tells torch which 
# parameters are involved in the forward pass, so that it
# can correctly (automatically) set up the backward pass.
#
# You should not modify the interals of the Transformer
# except where indicated to implement the multi-head
# attention. 


def clones(module, N):
    '''
    A helper function for producing N identical layers (each with their own parameters).
    
    inputs: 
        module: a pytorch nn.module
        N (int): the number of copies of that module to return

    returns:
        a ModuleList with the copies of the module (the ModuleList is itself also a module)
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# Problem 1
class RNN(nn.Module): # Implement a stacked vanilla RNN with Tanh nonlinearities.
  def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
    """
    emb_size:     The number of units in the input embeddings
    hidden_size:  The number of hidden units per layer
    seq_len:      The length of the input sequences
    vocab_size:   The number of tokens in the vocabulary (10,000 for Penn TreeBank)
    num_layers:   The depth of the stack (i.e. the number of hidden layers at 
                  each time-step)
    dp_keep_prob: The probability of *not* dropping out units in the 
                  non-recurrent connections.
                  Do not apply dropout on recurrent connections.
    """
    super(RNN, self).__init__()
    self.emb_size = emb_size
    self.hidden_size = hidden_size
    self.vocab_size = vocab_size
    self.seq_len = seq_len
    self.num_layers = num_layers
    self.batch_size = batch_size

    # Use the GPU if you have one
    if torch.cuda.is_available():
        self.device = torch.device("cuda") 
    else:
        self.device = torch.device("cpu")

    # Embedding layer, input to hidden, output, and dropout (same everywhere)
    self.em = WordEmbedding(emb_size, vocab_size)
    self.drop = nn.Dropout(p=(1-dp_keep_prob))
    self.inp_hid = nn.Linear(emb_size, hidden_size, bias=False)
    self.out = nn.Linear(hidden_size, vocab_size, bias=True)

    # Account for arbitrary number of hidden layers/rnn connections
    self.hiddens = nn.ModuleList([self.inp_hid]+list(clones(nn.Linear(hidden_size, hidden_size, bias=False), num_layers-1)))
    self.rnns = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=True) for i in range(num_layers)])

    # Explicitly cast hiddens and rnns to use GPU when available (yes, a hack, but necessary)
    hiddens2 = nn.ModuleList([hid.to(self.device) for hid in self.hiddens])
    self.hiddens = hiddens2
    rnns2 = nn.ModuleList([rnn.to(self.device) for rnn in self.rnns])
    self.rnns = rnns2

    # Initialize the weights
    self.init_weights()

  def init_weights(self):
    # TODO ========================
    # Initialize the embedding and output weights uniformly in the range [-0.1, 0.1]
    # and output biases to 0 (in place). The embeddings should not use a bias vector.
    # Initialize all other (i.e. recurrent and linear) weights AND biases uniformly 
    # in the range [-k, k] where k is the square root of 1/hidden_size

    # Initialize the embedding and output weights uniformly
    self.em.lut.weight.data.uniform_(-0.1, 0.1) 
    self.out.weight.data.uniform_(-0.1, 0.1)

    # Initialize output biases to 0
    self.out.bias.data.fill_(0.0)

    # Initialize all other weights and biases uniformly, over sqrt(1/hidden_size)

    for i, hid in enumerate(self.hiddens):
      hid.weight.data.uniform_(-math.sqrt(1 / self.hidden_size), math.sqrt(1 / self.hidden_size))
      self.rnns[i].weight.data.uniform_(-math.sqrt(1 / self.hidden_size), math.sqrt(1 / self.hidden_size))
      self.rnns[i].bias.data.uniform_(-math.sqrt(1 / self.hidden_size), math.sqrt(1 / self.hidden_size))


  def init_hidden(self):
    # TODO ========================
    # initialize the hidden states to zero
    """
    This is used for the first mini-batch in an epoch, only.
    """
    states = Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
    return states # a parameter tensor of shape (self.num_layers, self.batch_size, self.hidden_size)

  def forward(self, inputs, hidden):
    # TODO ========================
    # Compute the forward pass, using nested python for loops.
    # The outer for loop should iterate over timesteps, and the 
    # inner for loop should iterate over hidden layers of the stack. 
    # 
    # Within these for loops, use the parameter tensors and/or nn.modules you 
    # created in __init__ to compute the recurrent updates according to the 
    # equations provided in the .tex of the assignment.
    #
    # Note that those equations are for a single hidden-layer RNN, not a stacked
    # RNN. For a stacked RNN, the hidden states of the l-th layer are used as 
    # inputs to to the {l+1}-st layer (taking the place of the input sequence).

    """
    Arguments:
        - inputs: A mini-batch of input sequences, composed of integers that 
                    represent the index of the current token(s) in the vocabulary.
                        shape: (seq_len, batch_size)
        - hidden: The initial hidden states for every layer of the stacked RNN.
                        shape: (num_layers, batch_size, hidden_size)
    
    Returns:
        - Logits for the softmax over output tokens at every time-step.
              **Do NOT apply softmax to the outputs!**
              Pytorch's CrossEntropyLoss function (applied in ptb-lm.py) does 
              this computation implicitly.
                    shape: (seq_len, batch_size, vocab_size)
        - The final hidden states for every layer of the stacked RNN.
              These will be used as the initial hidden states for all the 
              mini-batches in an epoch, except for the first, where the return 
              value of self.init_hidden will be used.
              See the repackage_hiddens function in ptb-lm.py for more details, 
              if you are curious.
                    shape: (num_layers, batch_size, hidden_size)
    """

    for t in range(inputs.size()[0]):  # For each timestep

      one_input = inputs[t]
      x = self.em(one_input)  # Through embedding
      x = self.drop(x)        # Initial dropout on input
      
      # If not in first timestep
      if t != 0:
        new_prevs = []
        final_hidden_states = []
        # Pass through stack
        for i, hid in enumerate(self.hiddens):
          x = hid(x)
          x_act = torch.tanh(x + self.rnns[i](prevs[i]))  # Recurrent
          new_prevs.append(x_act)   # No dropout on recurrent connections
          x_drop = self.drop(x_act)
          x = x_drop                # Pass along the dropped version up the stack
          final_hidden_states.append(x_drop)
          # At last step, get the output logit
          if i == len(self.hiddens)-1:
            x_out = self.out(x_drop)
            logits.append(x_out)
        # Set current step to previous step for next loop
        prevs = new_prevs

      # If in first timestep
      else:
        prevs = []
        logits = []
        # Pass through stack
        for i, hid in enumerate(self.hiddens):
          x = hid(x)
          x_act = torch.tanh(x + self.rnns[i](hidden[i]))  # Where hidden is from input 
          prevs.append(x_act)   # No dropout on recurrent connections
          x_drop = self.drop(x_act)
          x = x_drop            # Pass along the dropped version up the stack
          # At last step, get the output logit
          if i == len(self.hiddens)-1:
            x_out = self.out(x_drop)
            logits.append(x_out)

      # If in last timestep
      if t == inputs.size()[0]-1:
        hidden = torch.cat(final_hidden_states)
        logits = torch.cat(logits)

    return logits.view(self.seq_len, self.batch_size, self.vocab_size), hidden

  def generate(self, input, hidden, generated_seq_len):
    # TODO ========================
    # Compute the forward pass, as in the self.forward method (above).
    # You'll probably want to copy substantial portions of that code here.
    # 
    # We "seed" the generation by providing the first inputs.
    # Subsequent inputs are generated by sampling from the output distribution, 
    # as described in the tex (Problem 5.3)
    # Unlike for self.forward, you WILL need to apply the softmax activation 
    # function here in order to compute the parameters of the categorical 
    # distributions to be sampled from at each time-step.

    """
    Arguments:
        - input: A mini-batch of input tokens (NOT sequences!)
                        shape: (batch_size)
        - hidden: The initial hidden states for every layer of the stacked RNN.
                        shape: (num_layers, batch_size, hidden_size)
        - generated_seq_len: The length of the sequence to generate.
                       Note that this can be different than the length used 
                       for training (self.seq_len)
    Returns:
        - Sampled sequences of tokens
                    shape: (generated_seq_len, batch_size)
    """

    return samples


# Problem 2
class GRU(nn.Module): # Implement a stacked GRU RNN
  """
  Follow the same instructions as for RNN (above), but use the equations for 
  GRU, not Vanilla RNN.
  """
  def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
    super(GRU, self).__init__()

    # given params, save them
    self.emb_size = emb_size
    self.hidden_size = hidden_size
    self.seq_len = seq_len
    self.batch_size = batch_size
    self.vocab_size = vocab_size
    self.num_layers = num_layers
    self.dp_keep_prob = dp_keep_prob

    # different actication functions for differernt gates
    # sigmoid r
    self.act_reset = nn.Sigmoid()
    # sigmoid Z
    self.act_forget = nn.Sigmoid()
    # hidden tanh
    self.act_hidden = nn.Tanh()

    # Use the GPU if you have one
    if torch.cuda.is_available():
        self.device = torch.device("cuda")
    else:
        self.device = torch.device("cpu")

    # Embedding layer, input to hidden, output, and dropout (same everywhere)
    # self.em = nn.Embedding(vocab_size, emb_size)
    self.em = WordEmbedding(emb_size, vocab_size)
    self.drop = nn.Dropout(p=(1-dp_keep_prob))
    # transform the input
    # self.inp = nn.Linear(emb_size, hidden_size)
    self.out = nn.Linear(hidden_size, vocab_size, bias=True)

    # TAs changed stuff LOL
    # self.inp_hid = nn.Linear(emb_size, hidden_size)
    self.inp_rnn = nn.Linear(emb_size, hidden_size, bias=False)

    # Account for arbitrary number of hidden layers/rnn connections
    # W_h . h_t
    if num_layers == 1:
      self.hiddens_u_reset = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=True)])
      self.rnns_w_reset = nn.ModuleList([self.inp_rnn]+[nn.Linear(hidden_size, hidden_size, bias=False)])
      self.hiddens_u_forget = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=True)])
      self.rnns_w_forget = nn.ModuleList([self.inp_rnn]+[nn.Linear(hidden_size, hidden_size, bias=False)])
      self.hiddens_u = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=True)])
      self.rnns_w = nn.ModuleList([self.inp_rnn]+[nn.Linear(hidden_size, hidden_size, bias=False)])

    else:
      self.hiddens_u_reset = nn.ModuleList(list(clones(nn.Linear(hidden_size, hidden_size, bias=True), num_layers)))
      self.rnns_w_reset =  nn.ModuleList([self.inp_rnn] + list(clones(nn.Linear(hidden_size, hidden_size, bias=False), num_layers-1)))
      self.hiddens_u_forget = nn.ModuleList( list(clones(nn.Linear(hidden_size, hidden_size, bias=True), num_layers)))
      self.rnns_w_forget = nn.ModuleList([self.inp_rnn] + list(clones(nn.Linear(hidden_size, hidden_size, bias=False), num_layers-1)))
      self.hiddens_u = nn.ModuleList(list(clones(nn.Linear(hidden_size, hidden_size, bias=True), num_layers)))
      self.rnns_w = nn.ModuleList([self.inp_rnn] + list(clones(nn.Linear(hidden_size, hidden_size, bias=False), num_layers-1)))

    # Explicitly cast hiddens and rnns to use GPU when available (yes, a hack, but necessary)
    hiddens_u_reset2 = nn.ModuleList([hid.to(self.device) for hid in self.hiddens_u_reset])
    self.hiddens_u_reset = hiddens_u_reset2

    rnns_w_reset2 = nn.ModuleList([rnn.to(self.device) for rnn in self.rnns_w_reset])
    self.rnns_w_reset = rnns_w_reset2

    hiddens_u_forget2 = nn.ModuleList([hid.to(self.device) for hid in self.hiddens_u_forget])
    self.hiddens_u_forget = hiddens_u_forget2

    hiddens_u2 = nn.ModuleList([hid.to(self.device) for hid in self.hiddens_u])
    self.hiddens_u = hiddens_u2

    rnns_w_forget2 = nn.ModuleList([rnn.to(self.device) for rnn in self.rnns_w_forget])
    self.rnns_w_forget = rnns_w_forget2

    rnns_w2 = nn.ModuleList([rnn.to(self.device) for rnn in self.rnns_w])
    self.rnns_w = rnns_w2

    # Initialize the weights
    self.init_weights_uniform()

  def init_weights_uniform(self):
    # TODO ========================
    # Initialize the embedding and output weights uniformly
    # nn.init.uniform_(self.em.weight, -0.1, 0.1)
    self.em.lut.weight.data.uniform_(-0.1, 0.1)
    # nn.init.uniform_(self.out.weight, -0.1, 0.1)
    self.out.weight.data.uniform_(-0.1, 0.1)

    # Initialize output biases to 0
    # nn.init.zeros_(self.out.bias)
    self.out.bias.data.fill_(0.0)

    # Initialize all other weights and biases uniformly, over sqrt(1/hidden_size)
    for i in range(self.num_layers):
      # reset gates weights and bias, U_r
      self.hiddens_u_reset[i].weight.data.uniform_(-math.sqrt(1 / self.hidden_size), math.sqrt(1 / self.hidden_size))

      # reset gates W_r
      self.rnns_w_reset[i].weight.data.uniform_(-math.sqrt(1 / self.hidden_size), math.sqrt(1 / self.hidden_size))

      # forget gate weights and bias, U_z
      self.hiddens_u_forget[i].weight.data.uniform_(-math.sqrt(1 / self.hidden_size), math.sqrt(1 / self.hidden_size))
      self.hiddens_u_forget[i].bias.data.uniform_(-math.sqrt(1 / self.hidden_size), math.sqrt(1 / self.hidden_size))

      # forget gate, W_z
      self.rnns_w_forget[i].weight.data.uniform_(-math.sqrt(1 / self.hidden_size), math.sqrt(1 / self.hidden_size))

      # hiddens_u, U_h
      self.hiddens_u[i].weight.data.uniform_(-math.sqrt(1 / self.hidden_size), math.sqrt(1 / self.hidden_size))
      self.hiddens_u[i].bias.data.uniform_(-math.sqrt(1 / self.hidden_size), math.sqrt(1 / self.hidden_size))

      # rnn_u, W_h
      self.rnns_w[i].weight.data.uniform_(-math.sqrt(1 / self.hidden_size), math.sqrt(1 / self.hidden_size))


  def init_hidden(self):
    # TODO ========================
    states = Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))

    return states # a parameter tensor of shape (self.num_layers, self.batch_size, self.hidden_size)

  def forward(self, inputs, hidden):
    logits_list = []
    for t in range(inputs.size()[0]):  # For each timestep
      one_input = inputs[t]
      x = self.em(one_input)
      x = self.drop(x)
      reset = x
      forget = x
      h_tilde_t = x
      final_hidden_states = []
      new_prevs = []
      if t == 0:  # If in first timestep
        prevs = hidden

      for i in range(self.num_layers):
        # print("prevs[i]", prevs[i].size()) # no need to transform input in hiddens????
        reset = self.rnns_w_reset[i](reset) + self.hiddens_u_reset[i](prevs[i])
        reset_act = self.act_reset(reset)

        forget = self.rnns_w_forget[i](forget) + self.hiddens_u_forget[i](prevs[i])
        forget_act = self.act_forget(forget)

        h_tilde_t = self.rnns_w[i](h_tilde_t) + self.hiddens_u[i](torch.mul(reset_act, prevs[i]))
        h_tilde_t_act = self.act_hidden(h_tilde_t)
        # print("forget_act", forget_act.size())
        h_t = torch.mul((torch.ones(self.batch_size, self.hidden_size, device=self.device) - forget_act), prevs[i]) \
                + torch.mul(forget_act, h_tilde_t_act)
        # print("h_t", h_t.size())
        new_prevs.append(h_t)

        h_t_drop = self.drop(h_t)
        final_hidden_states.append(h_t_drop)

        x_out = self.out(h_t_drop)

      prevs = new_prevs
      logits_list.append(x_out)

    # if t == inputs.size()[0] - 1:  # If in last timestep
    hidden = torch.cat(final_hidden_states)
    logits = torch.cat(logits_list)

    return logits.view(self.seq_len, self.batch_size, self.vocab_size), hidden

  def generate(self, input, hidden, generated_seq_len):
    # TODO ========================
    """
    Arguments:
        - input: A mini-batch of input tokens (NOT sequences!)
                        shape: (batch_size)
        - hidden: The initial hidden states for every layer of the stacked RNN.
                        shape: (num_layers, batch_size, hidden_size)
        - generated_seq_len: The length of the sequence to generate.
                       Note that this can be different than the length used
                       for training (self.seq_len)
    Returns:
        - Sampled sequences of tokens
                    shape: (generated_seq_len, batch_size)
    """
    logits_list = []
    for t in range(generated_seq_len):  # For each timestep
      one_input = input
      x = self.em(one_input)
      x = self.drop(x)

      reset = x
      forget = x
      h_tilde_t = x
      final_hidden_states = []
      new_prevs = []

      if t ==0:  # If in first timestep
        prevs = hidden


      for i in range(self.num_layers):
        # print("prevs[i]", prevs[i].size()) # no need to transform input in hiddens????
        reset = self.rnns_w_reset[i](reset) + self.hiddens_u_reset[i](prevs[i])
        reset_act = self.act_reset(reset)

        forget = self.rnns_w_forget[i](forget) + self.hiddens_u_forget[i](prevs[i])
        forget_act = self.act_forget(forget)

        h_tilde_t = self.rnns_w[i](h_tilde_t) + self.hiddens_u[i](torch.mul(reset_act, prevs[i]))
        h_tilde_t_act = self.act_hidden(h_tilde_t)
        # print("forget_act", forget_act.size())
        h_t = torch.mul((torch.ones(self.batch_size, self.hidden_size, device=self.device) - forget_act), prevs[i]) \
                + torch.mul(forget_act, h_tilde_t_act)
        new_prevs.append(h_t)

        h_t_drop = self.drop(h_t)
        final_hidden_states.append(h_t_drop)

        x_out = self.out(h_t_drop)

      prevs = new_prevs
      logits_list.append(x_out)

    # If in last timestep
    logits = torch.cat(logits_list)
    # end of copying code from forward
    logits = logits.view(generated_seq_len, self.batch_size, self.vocab_size)
    # print("logits", logits.size())
    samples = torch.distributions.Categorical(logits=logits).sample()
    # print("in models generate samples", samples.size())

    return samples


# Problem 3
##############################################################################
#
# Code for the Transformer model
#
##############################################################################

"""
Implement the MultiHeadedAttention module of the transformer architecture.
All other necessary modules have already been implemented for you.
We're building a transfomer architecture for next-step prediction tasks, and 
applying it to sequential language modelling. We use a binary "mask" to specify 
which time-steps the model can use for the current prediction.
This ensures that the model only attends to previous time-steps.
The model first encodes inputs using the concatenation of a learned WordEmbedding 
and a (in our case, hard-coded) PositionalEncoding.
The word embedding maps a word's one-hot encoding into a dense real vector.
The positional encoding 'tags' each element of an input sequence with a code that 
identifies it's position (i.e. time-step).
These encodings of the inputs are then transformed repeatedly using multiple
copies of a TransformerBlock.
This block consists of an application of MultiHeadedAttention, followed by a 
standard MLP; the MLP applies *the same* mapping at every position.
Both the attention and the MLP are applied with Resnet-style skip connections, 
and layer normalization.
The complete model consists of the embeddings, the stacked transformer blocks, 
and a linear layer followed by a softmax.
"""

#This code has been modified from an open-source project, by David Krueger.
#The original license is included below:
#MIT License
#
#Copyright (c) 2018 Alexander Rush
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.



#----------------------------------------------------------------------------------

# TODO: implement this class
class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, n_units, dropout=0.1):
        """
        n_heads: the number of attention heads
        n_units: the number of input and output units
        dropout: probability of DROPPING units
        """
        super(MultiHeadedAttention, self).__init__()
        # This sets the size of the keys, values, and queries (self.d_k) to all 
        # be equal to the number of output units divided by the number of heads.
        self.d_k = n_units // n_heads
        # This requires the number of n_heads to evenly divide n_units.
        assert n_units % n_heads == 0
        self.n_units = n_units 

        # TODO: create/initialize any necessary parameters or layers
        # Initialize all weights and biases uniformly in the range [-k, k],
        # where k is the square root of 1/n_units.
        # Note: the only Pytorch modules you are allowed to use are nn.Linear 
        # and nn.Dropout
        # ETA: you can also use softmax
        # ETA: you can use the "clones" function we provide.
        # ETA: you can use masked_fill
        
    def forward(self, query, key, value, mask=None):
        # TODO: implement the masked multi-head attention.
        # query, key, and value correspond to Q, K, and V in the latex, and 
        # they all have size: (batch_size, seq_len, self.n_units)
        # mask has size: (batch_size, seq_len, seq_len)
        # As described in the .tex, apply input masking to the softmax 
        # generating the "attention values" (i.e. A_i in the .tex)
        # Also apply dropout to the attention values.

        return # size: (batch_size, seq_len, self.n_units)






#----------------------------------------------------------------------------------
# The encodings of elements of the input sequence

class WordEmbedding(nn.Module):
    def __init__(self, n_units, vocab):
        super(WordEmbedding, self).__init__()
        self.lut = nn.Embedding(vocab, n_units)
        self.n_units = n_units

    def forward(self, x):
        #print (x)
        return self.lut(x) * math.sqrt(self.n_units)


class PositionalEncoding(nn.Module):
    def __init__(self, n_units, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_units)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, n_units, 2).float() *
                             -(math.log(10000.0) / n_units))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)



#----------------------------------------------------------------------------------
# The TransformerBlock and the full Transformer


class TransformerBlock(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(TransformerBlock, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ResidualSkipConnectionWithLayerNorm(size, dropout), 2)
 
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)) # apply the self-attention
        return self.sublayer[1](x, self.feed_forward) # apply the position-wise MLP


class TransformerStack(nn.Module):
    """
    This will be called on the TransformerBlock (above) to create a stack.
    """
    def __init__(self, layer, n_blocks): # layer will be TransformerBlock (below)
        super(TransformerStack, self).__init__()
        self.layers = clones(layer, n_blocks)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class FullTransformer(nn.Module):
    def __init__(self, transformer_stack, embedding, n_units, vocab_size):
        super(FullTransformer, self).__init__()
        self.transformer_stack = transformer_stack
        self.embedding = embedding
        self.output_layer = nn.Linear(n_units, vocab_size)
        
    def forward(self, input_sequence, mask):
        embeddings = self.embedding(input_sequence)
        return F.log_softmax(self.output_layer(self.transformer_stack(embeddings, mask)), dim=-1)


def make_model(vocab_size, n_blocks=6, 
               n_units=512, n_heads=16, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(n_heads, n_units)
    ff = MLP(n_units, dropout)
    position = PositionalEncoding(n_units, dropout)
    model = FullTransformer(
        transformer_stack=TransformerStack(TransformerBlock(n_units, c(attn), c(ff), dropout), n_blocks),
        embedding=nn.Sequential(WordEmbedding(n_units, vocab_size), c(position)),
        n_units=n_units,
        vocab_size=vocab_size
        )
    
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


#----------------------------------------------------------------------------------
# Data processing

def subsequent_mask(size):
    """ helper function for creating the masks. """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, x, pad=0):
        self.data = x
        self.mask = self.make_mask(self.data, pad)
    
    @staticmethod
    def make_mask(data, pad):
        "Create a mask to hide future words."
        mask = (data != pad).unsqueeze(-2)
        mask = mask & Variable(
            subsequent_mask(data.size(-1)).type_as(mask.data))
        return mask


#----------------------------------------------------------------------------------
# Some standard modules

class LayerNorm(nn.Module):
    "layer normalization, as in: https://arxiv.org/abs/1607.06450"
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ResidualSkipConnectionWithLayerNorm(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(ResidualSkipConnectionWithLayerNorm, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class MLP(nn.Module):
    """
    This is just an MLP with 1 hidden layer
    """
    def __init__(self, n_units, dropout=0.1):
        super(MLP, self).__init__()
        self.w_1 = nn.Linear(n_units, 2048)
        self.w_2 = nn.Linear(2048, n_units)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
