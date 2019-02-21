#!/usr/bin/env python
# -*- coding: utf-8 -*-

import keras.backend as K
from keras.layers import Input, Lambda, LSTM, Activation, concatenate, Dropout, warnings, Flatten, Dense, TimeDistributed, Multiply, Masking, Embedding
from keras.models import Model, Sequential
from keras.engine.topology import get_source_inputs
from keras.utils import get_file, plot_model, layer_utils
from keras.activations import softmax


class WordNet():
	def __init__(self):
		self.model = None
		self.model = self._audio_submodel()
		self.model.summary()
        

	def __call__(self, model_option):
		if model_option == "train":
        		return self.model
		else:
        		return None
    

	def _audio_submodel(self):
        
		hidden_size = 128
		input_size = 80		# change depending on feature size
		output_classes = 662

		inp = Input((None, input_size))
		inp2 = Masking(mask_value=0.0)(inp)
		lstm = LSTM(hidden_size, return_sequences=False)(inp2)

		# Attention Part (uncomment)

		lstm = LSTM(hidden_size, return_sequences=True)(lstm)
		attention = TimeDistributed(Dense(1))(lstm)		# attention weights!!
		attention = Lambda(lambda x: softmax(x, axis=1))(attention)
		context = Multiply()([attention,lstm])
		out = Lambda(lambda x: K.sum(x,axis=1))(context)		
		out = Dense(4096, activation='relu')(lstm)
		out = Dense(output_classes, activation='softmax')(out)
		model = Model(inputs=inp,outputs=[out])			
		
        
		return model
    

