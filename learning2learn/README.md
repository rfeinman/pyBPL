## Learning hyper-parameters for pyBPL code
To begin, make sure that pyBPL is in your `PYTHONPATH`.

### Learning sub-stroke sequence RNN
In this section you will train an RNN to model sequences of sub-stroke IDs. 

1. Create spline data set

First, you'll need to create the pre-processed omniglot dataset. 
You can do this with the following steps. 
First, make sure you have the Omniglot background data set file in this folder, called `data_background.mat`. 
Also, make sure that the BPL and BPL_fit_hyperparameters Matlab repositories are in your Matlab path. 
Call the Matlab function `omniglot_extract_splines.m` with parameter "train", which will extract spline sequences for each stroke in the background Omniglot dataset. 
This will create a file called `data_background_splines.mat`. 

2. Create sub-stroke ID data set

Next, run the Python script `make_subid_sequences.py --mode=train` to build the dataset of sub-stroke ID sequences, with one sequence per stroke in the Omniglot background set. 
This will create a data file called `subid_sequences_background.p`.

3. Train sub-stroke ID LSTM

Finally, run the script `train_rnn_subids.py`. 
This will train an RNN to model sub-stroke ID sequences. 
The model will be saved to a file called `rnn_subids.h5`

### Evaluating sub-stroke RNN against the HMM
Now you will evaluate the performance of the RNN on the omniglot evaluation set and compare it to the performance of the bigram HMM.

1. Create dataset

The process mirrors 1-2. from above.
Make sure you have the Omniglot evaluation set file in this folder, called `data_evaluation.mat`.
Call the Matlab function `omniglot_extract_splines.m` with parameter "test".
Then, run the Python function `make_subid_sequences.py --mode=test`.

2. Run the evaluation script

Run the Python script `compare_hmm_rnn.py` to compare the performance of the RNN vs the HMM on the evaluation set. Results will be printed.