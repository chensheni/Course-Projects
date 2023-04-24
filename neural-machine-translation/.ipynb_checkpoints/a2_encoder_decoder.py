'''
This code is provided solely for the personal and private use of students
taking the CSC401H/2511H course at the University of Toronto. Copying for
purposes other than this use is expressly prohibited. All forms of
distribution of this code, including but not limited to public repositories on
GitHub, GitLab, Bitbucket, or any other online platform, whether as given or
with any changes, are expressly prohibited.

Authors: Sean Robertson, Jingcheng Niu, Zining Zhu, and Mohamed Abdall
Updated by: Raeid Saqur <raeidsaqur@cs.toronto.edu>

All of the files in this directory and all subdirectories are:
Copyright (c) 2023 University of Toronto
'''

'''Concrete implementations of abstract base classes.

You don't need anything more than what's been imported here
'''

import torch
from typing import Optional, Union, Tuple, Type, Set

from a2_abcs import EncoderBase, DecoderBase, EncoderDecoderBase

# All docstrings are omitted in this file for simplicity. So please read
# a2_abcs.py carefully so that you can have a solid understanding of the
# structure of the assignment.

class Encoder(EncoderBase):


    def init_submodules(self):
        # Hints:
        # 1. You must initialize the following submodules:
        #   self.rnn, self.embedding

        # 2. You will need the following object attributes:
        #   self.source_vocab_size, self.word_embedding_size,
        #   self.pad_id, self.dropout, self.cell_type,
        #   self.hidden_state_size, self.num_hidden_layers.

        # 3. cell_type will be one of: ['lstm', 'rnn']
        # 4. Relevant pytorch modules: torch.nn.{LSTM, RNN, Embedding}

        # cell_type options
        models = {'lstm': torch.nn.LSTM, 'rnn': torch.nn.RNN}

        # embedding
        self.embedding = torch.nn.Embedding(num_embeddings = self.source_vocab_size,
                                            embedding_dim  = self.word_embedding_size,
                                            padding_idx    = self.pad_id)
        
        # model
        self.rnn = models[self.cell_type](input_size    = self.word_embedding_size,
                                          hidden_size   = self.hidden_state_size,
                                          num_layers    = self.num_hidden_layers,
                                          dropout       = self.dropout,
                                          bidirectional = True)

    def forward_pass(
            self,
            source_x: torch.LongTensor,
            source_x_lens: torch.LongTensor,
            h_pad: float = 0.) -> torch.FloatTensor:
        # Recall:
        #   source_x is shape (S, B)
        #   source_x_lens is of shape (B,)
        #   h_pad is a float
        #
        # Hints:
        # 1. The structure of the encoder should be:
        #   input seq -> |embedding| -> embedded seq -> |rnn| -> seq hidden
        # 2. You will need to use the following methods:
        #   self.get_all_rnn_inputs, self.get_all_hidden_states

        emb = self.get_all_rnn_inputs(source_x)
        out = self.get_all_hidden_states(emb, source_x_lens, h_pad)

        #print(f'source_x.shape: {source_x.shape}')
        #print(f'source_x_lens: {source_x_lens}')
        #print(f'h_pad: {h_pad}')
        #print(f'emb.shape: {emb.shape}')
        #print(f'out.shape: {out.shape}')
        return out
        

    def get_all_rnn_inputs(self, source_x: torch.LongTensor) -> torch.FloatTensor:
        # Recall:
        #   source_x is shape (S, B)
        #   x (output) is shape (S, B, I)
        x = self.embedding(source_x)
        return x

    def get_all_hidden_states(
            self, 
            x: torch.FloatTensor,
            source_x_lens: torch.LongTensor,
            h_pad: float) -> torch.FloatTensor:
        # Recall:
        #   x is of shape (S, B, I)
        #   source_x_lens is of shape (B,)
        #   h_pad is a float
        #   h (output) is of shape (S, B, 2 * H)
        #
        # Hint:
        #   relevant pytorch modules:
        #   torch.nn.utils.rnn.{pad_packed,pack_padded}_sequence

        # pack tensors before feeding in the model
        # print(f'x:{x}')
        x_pack = torch.nn.utils.rnn.pack_padded_sequence(input = x, 
                                                         lengths = source_x_lens, 
                                                         enforce_sorted = False)
        
        # packed output result
        if self.cell_type    == 'rnn':
            output, hn       = self.rnn(x_pack)
        elif self.cell_type  == 'lstm':
            output, (hn, cn) = self.rnn(x_pack)

        # unpack output
        h, lens_unpacked = torch.nn.utils.rnn.pad_packed_sequence(sequence = output, 
                                                                  padding_value = h_pad)
        #print(f'output of encoder:{h.shape}')
        return h


class DecoderWithoutAttention(DecoderBase):
    '''A recurrent decoder without attention'''

    def init_submodules(self):
        # Hints:
        # 1. You must initialize the following submodules:
        #   self.embedding, self.cell, self.output_layer
        # 2. You will need the following object attributes:
        #   self.target_vocab_size, self.word_embedding_size, self.pad_id
        #   self.hidden_state_size, self.cell_type.
        # 3. cell_type will be one of: ['lstm', 'rnn']
        # 4. Relevant pytorch modules:
        #   torch.nn.{Embedding, Linear, LSTMCell, RNNCell}

        # cell_type options
        models = {'lstm': torch.nn.LSTM, 'rnn': torch.nn.RNN}

        # embedding
        self.embedding = torch.nn.Embedding(num_embeddings = self.target_vocab_size,
                                            embedding_dim  = self.word_embedding_size,
                                            padding_idx    = self.pad_id)
        #print(f'self.target_vocab_size:{self.target_vocab_size}')
        #print(f'self.word_embedding_size:{self.word_embedding_size}')
        
        # cell
        self.cell = models[self.cell_type](input_size  = self.word_embedding_size,
                                           hidden_size = self.hidden_state_size)
        
        # output_layer
        self.output_layer = torch.nn.Linear(in_features  = self.hidden_state_size,
                                            out_features = self.target_vocab_size)
        

    def forward_pass(
            self,
            target_y_tm1: torch.LongTensor,
            htilde_tm1: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor,
            source_x_lens: torch.LongTensor) -> Tuple[
                torch.FloatTensor, Union[
                    torch.FloatTensor,
                    Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # Recall:
        #   target_y_tm1 is of shape (B,)
        #   htilde_tm1 is of shape (B, 2 * H)
        #   h is of shape (S, B, 2 * H)
        #   source_x_lens is of shape (B,)
        #   logits_t (output) is of shape (B, V)
        #   htilde_t (output) is of same shape as htilde_tm1
        #
        # Hints:
        # 1. The structure of the encoder should be:
        #   encoded hidden -> |embedding| -> embedded hidden -> |rnn| ->
        #   decoded hidden -> |output layer| -> output logits
        # 2. You will need to use the following methods:
        #   self.get_current_rnn_input, self.get_current_hidden_state,
        #   self.get_current_logits
        # 3. You can assume that htilde_tm1 is not empty. I.e., the hidden state
        #   is either initialized, or t > 1.
        # 4. The output of an LSTM cell is a tuple (h, c), but a GRU cell or an
        #   RNN cell will only output h.

        # encoder RNN input
        xtilde_t = self.get_current_rnn_input(target_y_tm1, htilde_tm1, h, source_x_lens)
        #print(f'xtilde_t11.shape:{xtilde_t.shape}')
        #print(f'htilde_tm1.shape:{htilde_tm1[0].shape}')
        #print(f'htilde_tm2.shape:{htilde_tm1[1].shape}')

        # decoder hidden states
        htilde_t = self.get_current_hidden_state(xtilde_t, htilde_tm1)
        #print(f'htilde_t.shape:{htilde_t.shape}')

        # get logits
        if   self.cell_type == 'lstm':
            logits_t = self.get_current_logits(htilde_t[0])
        elif self.cell_type == 'rnn':
            logits_t = self.get_current_logits(htilde_t)

        return logits_t, htilde_t


    def get_first_hidden_state(
            self,
            h: torch.FloatTensor,
            source_x_lens: torch.LongTensor) -> torch.FloatTensor:
        # Recall:
        #   h is of shape (S, B, 2 * H)
        #   source_x_lens is of shape (B,)
        #   htilde_tm1 (output) is of shape (B, 2 * H)
        #
        # Hint:
        # 1. Ensure it is derived from encoder hidden state that has processed
        # the entire sequence in each direction. You will need to:
        # - Populate indices [0: self.hidden_state_size // 2] with the hidden
        #   states of the encoder's forward direction at the highest index in
        #   time *before padding*
        # - Populate indices [self.hidden_state_size//2:self.hidden_state_size]
        #   with the hidden states of the encoder's backward direction at time
        #   t=0
        # 2. Relevant pytorch function: torch.cat
        
        # forward & backward
        forward  = h[source_x_lens - 1, :, 0 : self.hidden_state_size//2]
        forward_1  = h[source_x_lens - 1, torch.arange(source_x_lens.shape[0]), 0 : self.hidden_state_size//2]
        backward = h[0                , :                                   , self.hidden_state_size//2 : self.hidden_state_size]
        print(f'forward:{forward.shape}')
        print(f'forward1:{forward_1.shape}')
        #print(f'backward:{backward.shape}')
        # concatenate
        htilde_tm1 = torch.cat([forward, backward], dim = 1).to(h.device)
        #print(f'htilde_tm1xx:{htilde_tm1.shape}')
        return htilde_tm1

    def get_current_rnn_input(
            self,
            target_y_tm1: torch.LongTensor,
            htilde_tm1: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor,
            source_x_lens: torch.LongTensor) -> torch.FloatTensor:
        # Recall:
        #   target_y_tm1 is of shape (B,)
        #   htilde_tm1 is of shape (B, 2 * H) or a tuple of two of those (LSTM)
        #   h is of shape (S, B, 2 * H)
        #   source_x_lens is of shape (B,)
        #   xtilde_t (output) is of shape (B, Itilde)
        xtilde_t = self.embedding(target_y_tm1)
        return xtilde_t.to(h.device)

    def get_current_hidden_state(
            self,
            xtilde_t: torch.FloatTensor,
            htilde_tm1: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]]) -> Union[
                    torch.FloatTensor,
                    Tuple[torch.FloatTensor, torch.FloatTensor]]:
        # Recall:
        #   xtilde_t is of shape (B, Itilde)
        #   htilde_tm1 is of shape (B, 2 * H) or a tuple of two of those (LSTM)
        #   htilde_t (output) is of same shape as htilde_tm1
        #print(f'xtilde_t:{xtilde_t.shape}')
        #print(f'htilde_tm1:{htilde_tm1.shape}')
        xtilde_t   = torch.unsqueeze(xtilde_t, 0)
        #htilde_tm1 = torch.unsqueeze(htilde_tm1, 0)
        #output, hn = self.cell(xtilde_t, htilde_tm1)
        #print(f'output:{output.shape}')
        #print(f'hn:{hn.shape}')
        #htilde_t   = torch.squeeze(output, 0)
        if self.cell_type == 'lstm':
            htilde_tm1 = [torch.unsqueeze(htilde_tm1[0], 0), torch.unsqueeze(htilde_tm1[1], 0)]
        else:
            htilde_tm1 = torch.unsqueeze(htilde_tm1, 0)

        output, hn = self.cell(xtilde_t, htilde_tm1)

        if self.cell_type == 'lstm':
            htilde_t = [torch.squeeze(hn[0], 0), torch.squeeze(hn[1], 0)]
        else:
            htilde_t = torch.squeeze(hn, 0)
        
        return htilde_t

    def get_current_logits(
            self,
            htilde_t: torch.FloatTensor) -> torch.FloatTensor:
        # Recall:
        #   htilde_t is of shape (B, 2 * H), even for LSTM (cell state discarded)
        #   logits_t (output) is of shape (B, V)
        logits_t = self.output_layer(htilde_t)
        #print(f'htilde_t:{htilde_t.shape}')
        #print(f'logits_t:{logits_t.shape}')
        return logits_t


class DecoderWithAttention(DecoderWithoutAttention):
    '''A decoder, this time with attention

    Inherits from DecoderWithoutAttention to avoid repeated code.
    '''

    def init_submodules(self):
        # Hints:
        # 1. Same as the case without attention, you must initialize the
        #   following submodules: self.embedding, self.cell, self.output_layer
        # 2. You will need the following object attributes:
        #   self.target_vocab_size, self.word_embedding_size, self.pad_id
        #   self.hidden_state_size, self.cell_type.
        # 3. cell_type will be one of: ['lstm', 'rnn']
        # 4. Relevant pytorch modules:
        #   torch.nn.{Embedding, Linear, LSTMCell, RNNCell, GRUCell}
        # 5. The implementation of this function should be different from
        #   DecoderWithoutAttention.init_submodules.
        models = {'lstm': torch.nn.LSTM, 'rnn': torch.nn.RNN}

        # embedding
        self.embedding = torch.nn.Embedding(num_embeddings = self.target_vocab_size,
                                            embedding_dim  = self.word_embedding_size,
                                            padding_idx    = self.pad_id)
        
        # cell (only a change here)
        self.cell = models[self.cell_type](input_size  = self.word_embedding_size + self.hidden_state_size,
                                           hidden_size = self.hidden_state_size)
        
        # output_layer
        self.output_layer = torch.nn.Linear(in_features  = self.hidden_state_size,
                                            out_features = self.target_vocab_size)

    def get_first_hidden_state(
            self,
            h: torch.FloatTensor,
            source_x_lens: torch.LongTensor) -> torch.FloatTensor:
        # Hint: For this time, the hidden states should be initialized to zeros.
        return torch.zeros_like(h[0]).to(h.device)

    def get_current_rnn_input(
            self,
            target_y_tm1: torch.LongTensor,
            htilde_tm1: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor,
            source_x_lens: torch.LongTensor) -> torch.FloatTensor:
        # Hint: Use attend() for c_t
        xtilde_t = self.embedding(target_y_tm1)
        c_tm1 = self.attend(htilde_tm1, h, source_x_lens)

        # xtilde_t: concatenate both (order does not matter)
        xtilde_t = torch.cat((xtilde_t, c_tm1), dim = 1).to(h.device)
        return xtilde_t

    def attend(
            self,
            htilde_t: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor,
            source_x_lens: torch.LongTensor) -> torch.FloatTensor:
        '''The attention mechanism. Calculate the context vector c_t.

        Parameters
        ----------
        htilde_t : torch.FloatTensor or tuple
            Like `htilde_tm1` (either a float tensor or a pair of float
            tensors), but matching the current hidden state.
        h : torch.FloatTensor
            A float tensor of shape ``(S, B, self.hidden_state_size)`` of
            hidden states of the encoder. ``h[s, b, i]`` is the
            ``i``-th index of the encoder RNN's last hidden state at time ``s``
            of the ``b``-th sequence in the batch. The states of the
            encoder have been right-padded such that ``h[source_x_lens[b]:, b]``
            should all be ignored.
        source_x_lens : torch.LongTensor
            An integer tensor of shape ``(B,)`` corresponding to the lengths
            of the encoded source sentences.

        Returns
        -------
        c_t : torch.FloatTensor
            A float tensor of shape ``(B, self.hidden_state_size)``. The
            context vector c_t is the product of weights alpha_t and h.

        Hint: Use get_attention_weights() to calculate alpha_t.
        '''
        # attention weights
        alpha_t = self.get_attention_weights(htilde_t, h, source_x_lens).unsqueeze(-1)

        # combination: multiply + sum
        c_t = torch.mul(alpha_t, h)
        c_t = torch.sum(c_t, dim = 0)
        return c_t

    def get_attention_weights(
            self,
            htilde_t: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor,
            source_x_lens: torch.LongTensor) -> torch.FloatTensor:
        # DO NOT MODIFY! Calculates attention weights, ensuring padded terms
        # in h have weight 0 and no gradient. You have to implement
        # get_attention_scores()
        # alpha_t (output) is of shape (S, B)
        a_t = self.get_attention_scores(htilde_t, h)
        pad_mask = torch.arange(h.shape[0], device=h.device)
        pad_mask = pad_mask.unsqueeze(-1) >= source_x_lens.to(h.device)  # (S, B)
        a_t = a_t.masked_fill(pad_mask, -float('inf'))
        return torch.nn.functional.softmax(a_t, 0)

    def get_attention_scores(
            self,
            htilde_t: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor) -> torch.FloatTensor:
        # Recall:
        #   htilde_t is of shape (B, 2 * H)
        #   h is of shape (S, B, 2 * H)
        #   a_t (output) is of shape (S, B)
        #
        # Hint:
        # Relevant pytorch function: torch.nn.functional.cosine_similarity
        if   self.cell_type == "lstm":
            a_t = torch.nn.functional.cosine_similarity(htilde_t[0], h, dim = -1)
        elif self.cell_type == "rnn":
            a_t = torch.nn.functional.cosine_similarity(htilde_t,    h, dim = -1)
        return a_t

class DecoderWithMultiHeadAttention(DecoderWithAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.W is not None, 'initialize W!'
        assert self.Wtilde is not None, 'initialize Wtilde!'
        assert self.Q is not None, 'initialize Q!'

    def init_submodules(self):
        super().init_submodules()  # Do not change this line

        # Hints:
        # 1. The above line should ensure self.output_layer, self.embedding, self.cell are
        #    initialized
        # 2. You need to initialize the following submodules:
        #       self.W, self.Wtilde, self.Q
        # 3. You will need the following object attributes:
        #       self.hidden_state_size
        # 4. self.W, self.Wtilde, and self.Q should process all heads at once. They
        #    should not be lists!
        # 5. You do *NOT* need self.heads at this point
        # 6. Relevant pytorch module: torch.nn.Linear (note: set bias=False!)
        self.W      = torch.nn.Linear(in_features = self.hidden_state_size, out_features = self.hidden_state_size, bias = False)
        self.Wtilde = torch.nn.Linear(in_features = self.hidden_state_size, out_features = self.hidden_state_size, bias = False)
        self.Q      = torch.nn.Linear(in_features = self.hidden_state_size, out_features = self.hidden_state_size, bias = False)

    def attend(
            self,
            htilde_t: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor,
            source_x_lens: torch.LongTensor) -> torch.FloatTensor:
        # Hints:
        # 1. You can use super().attend to call for the regular attention
        #   function.
        # 2. Relevant pytorch function:
        #   tensor().view, tensor().repeat_interleave
        # 3. Fun fact:
        #   tensor([1,2,3,4]).repeat(2) will output tensor([1,2,3,4,1,2,3,4]).
        #   tensor([1,2,3,4]).repeat_interleave(2) will output
        #   tensor([1,1,2,2,3,3,4,4]), just like numpy.repeat.
        # 4. You *WILL* need self.heads at this point
        # count
        #print(f'self.hidden_state_size:{self.hidden_state_size}')
        #print(f'self.heads:{self.heads}')
        number_of_parts = self.hidden_state_size // self.heads
        #print(f'number_of_parts:{number_of_parts}')
        if   self.cell_type == "lstm":
            htilde_t = (self.Wtilde(htilde_t[0]), htilde_t[1])
        elif self.cell_type == "rnn":
            htilde_t = self.Wtilde(htilde_t)

        h = self.W(h)
        #print(f'h:{h.shape}')

        # reshape
        # S * cnt per head
        h = h.view(h.shape[0], -1, number_of_parts)
        #print(f'htilde_t:{htilde_t.shape}')
        if   self.cell_type == "lstm":
            htilde_t = (htilde_t[0].view(-1, number_of_parts), 
                        htilde_t[1].view(-1, number_of_parts))
        elif self.cell_type == "rnn":
            htilde_t = htilde_t.view(-1, number_of_parts)

        #print(f'htilde_t:{htilde_t.shape}')

        # repeat for multiheads
        source_x_lens_multi = source_x_lens.repeat_interleave(repeats = self.heads)

        ctn = super().attend(htilde_t, h, source_x_lens_multi)
        #print(f'ctn:{ctn.shape}')
        # reshape back
        ctn = ctn.view(-1, self.hidden_state_size)
        #print(f'ctn:{ctn.shape}\n')

        return self.Q(ctn)

class EncoderDecoder(EncoderDecoderBase):

    def init_submodules(
            self,
            encoder_class: Type[EncoderBase],
            decoder_class: Type[DecoderBase]):
        # Hints:
        # 1. You must initialize the following submodules:
        #   self.encoder, self.decoder
        # 2. encoder_class and decoder_class inherit from EncoderBase and
        #   DecoderBase, respectively.
        # 3. You will need the following object attributes:
        #   self.source_vocab_size, self.source_pad_id,
        #   self.word_embedding_size, self.encoder_num_hidden_layers,
        #   self.encoder_hidden_size, self.encoder_dropout, self.cell_type,
        #   self.target_vocab_size, self.target_eos, self.heads
        # 4. Recall that self.target_eos doubles as the decoder pad id since we
        #   never need an embedding for it.
        self.encoder = encoder_class(source_vocab_size = self.source_vocab_size,
                                     pad_id = self.source_pad_id,
                                     word_embedding_size = self.word_embedding_size,
                                     num_hidden_layers = self.encoder_num_hidden_layers,
                                     hidden_state_size = self.encoder_hidden_size,
                                     dropout = self.encoder_dropout,
                                     cell_type = self.cell_type)

        self.decoder = decoder_class(target_vocab_size = self.target_vocab_size,
                                     pad_id = self.target_eos,
                                     word_embedding_size = self.word_embedding_size,
                                     hidden_state_size = self.encoder_hidden_size * 2,
                                     cell_type = self.cell_type,
                                     heads = self.heads)

    def translate(self, input_sentence):
        # This method translates the input sentence from the model's source
        # language to the target language.
        # 1. Tokenize the input sentence.
        # 2. Compute the length of the input sentence.
        # 3. Feed the tokenized sentence into the model.
        # 4. Decode the output of the sentence into a string.

        # Hints:
        # 1. You will need the following methods/attributs from the dataset.
        # Consult :class:`HansardEmptyDataset` for a description of parameters
        # and attributes.
        #   self.dataset.tokenize(input_sentence)
        #       This function tokenizes the input sentence.  For example:
        #       >>> self.dataset.tokenize('This is a sentence.')
        #       ['this', 'is', 'a', 'sentence']
        #   self.dataset.source_word2id
        #       A dictionary that maps tokens to ids for the source language.
        #       For example: `self.dataset.source_word2id['francophone'] -> 5127`
        #   self.dataset.source_unk
        #       The speical token for unknown input tokens.  Any token in the
        #       input sentence that isn't present in the source vocabulary should
        #       be converted to this special token.
        #   self.dataset.target_id2word
        #       A dictionary that maps ids to tokens for the target language.
        #       For example: `self.dataset.source_word2id[6123] -> 'anglophone'`
        # 
        # 2. Relevant pytorch function:
        #   tensor().view, tensor().repeat_interleave

        # 1. Tokenize the input sentence.
        token_list = self.dataset.tokenize(input_sentence)
        # print(f'token_list:{token_list}')

        # 2. Compute the length of the input sentence.
        token_cnt  = len(token_list)
        # since blank = 2
        token_id_list = [2] * (token_cnt)
        for i in range(token_cnt):
            # self.dataset.source_word2id is a dict
            if token_list[i] in self.dataset.source_word2id.keys():
                token_id_list[i] = self.dataset.source_word2id[token_list[i]]
            else:
                token_id_list[i] = self.dataset.source_unk

        # 3. Feed the tokenized sentence into the model.
        token_id_list = torch.tensor(token_id_list).unsqueeze(-1)
        source_x_lens = torch.tensor([token_cnt])

        logits = self.forward(token_id_list, source_x_lens)
        output = logits.view(logits.shape[0], -1)[:,0].tolist()

        # 4. Decode the output of the sentence into a string.
        output_list = []
        for i in range(len(output)):
            if output[i] in self.dataset.target_id2word.keys():
                output_list.append(self.dataset.target_id2word[output[i]])

        return " ".join(output_list)

    def get_logits_for_teacher_forcing(
            self,
            h: torch.FloatTensor,
            source_x_lens: torch.LongTensor,
            target_y: torch.LongTensor) -> torch.FloatTensor:
        # Recall:
        #   h is of shape (S, B, 2 * H)
        #   source_x_lens is of shape (B,)
        #   target_y is of shape (T, B)
        #   logits (output) is of shape (T - 1, B, Vo)
        #
        # Hints:
        # 1. Relevant pytorch modules: torch.{zero_like, stack}
        # 2. Recall an LSTM's cell state is always initialized to zero.
        # 3. Note logits sequence dimension is one shorter than target_y (why?)

        # first decoder hidden state
        htilde_t = None
        #print(f'h.shape:{h.shape}')
        #print(f'source_x_lens.shape:{source_x_lens.shape}')
        #print(f'target_y.shape:{target_y.shape}')

        # logits size (one shorter than target_y)
        logits = torch.zeros(target_y.shape[0] - 1, h.shape[1], self.target_vocab_size).to(h.device)
        #print(f'logits.shape:{logits.shape}')

        # iterate over tokens
        for t in range(1, target_y.shape[0]):
            htilde_tm1 = htilde_t

            logits_t, htilde_t = self.decoder.forward(target_y[t-1, :], htilde_tm1, h, source_x_lens)
            logits[t-1, :, :] = logits_t

        return logits.to(h.device)

    def update_beam(
            self,
            htilde_t: torch.FloatTensor,
            b_tm1_1: torch.LongTensor,
            logpb_tm1: torch.FloatTensor,
            logpy_t: torch.FloatTensor) -> Tuple[
                torch.FloatTensor, torch.LongTensor, torch.FloatTensor]:
        # perform the operations within the psuedo-code's loop in the
        # assignment.
        # You do not need to worry about which paths have finished, but DO NOT
        # re-normalize logpy_t.
        #
        # Recall
        #   htilde_t is of shape (B, K, 2 * H) or a tuple of two of those (LSTM)
        #   logpb_tm1 is of shape (B, K)
        #   b_tm1_1 is of shape (t, B, K)

        ## Output order:
        #   logpb_t (first output) is of shape (B, K)
        #   b_t_0 (second output) is of shape (B, K, 2 * H) or a tuple of two of
        #      those (LSTM)
        #   b_t_1 (third output) is of shape (t + 1, B, K)
        # Hints:
        # 1. Relevant pytorch modules:
        #   torch.{flatten, topk, unsqueeze, expand_as, gather, cat}
        # 2. If you flatten a two-dimensional array of shape z of (X, Y),
        #   then the element z[a, b] maps to z'[a*Y + b]
        
        # logpb_t
        extended_path = logpb_tm1.unsqueeze(-1).repeat(1,1,logpy_t.shape[-1]) + logpy_t
        extended_path = extended_path.reshape(logpy_t.shape[0], logpy_t.shape[1] * logpy_t.shape[2])

        # select path
        logpb_t, top_index = torch.topk(extended_path, k = self.beam_width, dim = 1)
        paths = torch.div(top_index, logpy_t.shape[-1],  rounding_mode='floor')
        v_ind = torch.remainder(top_index, logpy_t.shape[-1])

        # b_t_0
        if self.decoder.cell_type == "lstm":
            b_t_0 = (htilde_t[0].gather(dim = 1, index = paths.unsqueeze(dim = 2).expand_as(htilde_t[0])),
                     htilde_t[1].gather(dim = 1, index = paths.unsqueeze(dim = 2).expand_as(htilde_t[1])))
        else:
            b_t_0 = htilde_t.gather(dim = 1, index = paths.unsqueeze(dim = 2).expand_as(htilde_t))

        # keep the valid paths and concatenate
        b_tm1_1 = b_tm1_1.gather(2, paths.expand_as(b_tm1_1))
        b_t_1 = torch.cat([b_tm1_1, v_ind.unsqueeze(0)], dim=0)

        return logpb_t, b_t_0, b_t_1
