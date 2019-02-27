## Sequence Classification with Self Attention in Keras

#### Implementation of a sequence classification system in Keras. The input consists of a sequence of features passed to two layers of LSTM. Self attention is applied on the second LSTM and the final weighted context is supplied to two dense layers with softmax classification.
#### NOTE: The code supports different sequence lengths. Sequences are padded to the max sequence length in the batch. To avoid loss computation on padded timesteps, masking is applied.
#### </br>

### Requirements
* Python 3.6
* Keras
* Numpy
* Pickle
* Sklearn
* Matplotlib
#### </br>

### Dataset

#### Store your dataset as a dictionary in a pickle file. The dictionary should be in the form of \<filename\>: \<datapoint\>. Enter the path for the dataset in the data_path variable in 'train_dataset.py'. Also modify the input_size and output_classes variables in 'model_attention.py'.
#### </br>

### Running the Code

#### To train the model, run:
#### ```python train_dataset.py```
#### </br>

#### NOTE: Model and history is saved in the path specified by the save_path variable in train_dataset.py

