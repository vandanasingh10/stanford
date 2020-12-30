#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn


# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(f)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        #pad_token_idx = vocab.src['<pad>']
        #self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        self.char_embedding_size = 50 # this is the embedding size of each character
        self.word_embedding_size = embed_size # this is the embedding size of the word
        self.embed_size = embed_size # nmt_model need this attribute

        # we will get the embedding for the characters
        self.char_embedding = nn.Embedding(len(vocab.char2id), self.char_embedding_size, padding_idx=0)

        self.cnn = CNN(self.char_embedding_size, self.word_embedding_size, kernel_size=5)
        self.highway = Highway(self.word_embedding_size)
        self.dropout = nn.Dropout(0.3)

        ### YOUR CODE HERE for part 1f

        ### END YOUR CODE

    def forward(self, input_tensor):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input_tensor: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        #output = self.embeddings(input_tensor)
        #return output
        ## End A4 code

        # the output shape will be (sentence_length, batch_size, max_word_length, char_embedding_size)
        x_char_embed = self.char_embedding(input_tensor)  

        # we are bringing the input channels(the char_embed_dims) of each word to -2 postion
        # the new shape will be (sentence_length, batch_size, char_embedding_size, max_word_length)
        x_reshaped = x_char_embed.permute(0, 1, 3, 2)  

        # conv1d only takes 3 dimension object. So, combine first two dimension
        # the new shape is (seq_len*batch_size, char_embedding_size, max_word_length)
        x_reshaped_reduced = x_reshaped.view(-1, x_reshaped.shape[-2], x_reshaped.shape[-1])  

        # the output shape will be (seq_len*batch_size, word_embedding_size)
        x_conv = self.cnn(x_reshaped_reduced) 

        # no change in output shape
        x_highway = self.highway(x_conv)  

        ## dropout layer and reshaping to sent_length, batch, word_embedding
        x_word_embed = self.dropout(x_highway.view(x_reshaped.shape[0], 
                                                   x_reshaped.shape[1], 
                                                   self.word_embedding_size))

        return x_word_embed
        ### YOUR CODE HERE for part 1f

        ### END YOUR CODE
