## Sequence Classification with Self Attention in Keras

#### Implementation of a sequence classification system in Keras. The input consists of a sequence of features passed to two layers of LSTM. Self attention is applied on the second LSTM and the final weighted context is supplied to two dense layers with softmax classification.
#### NOTE: The code supports different sequence lengths. Sequences are padded to the max sequence length in the batch. To avoid loss on padded timesteps, masking is applied.
#### </br>

### Requirements
* Python 3.6
* Keras
* Numpy
* Pickle
* Sklearn
* Matplotlib
#### </br>

### Running the Code


