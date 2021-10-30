# rnn_with_deep_set_initialization
This repository contains two experimental models: <br /> <br />

1. RNNWithDeepSetInitialization and RNNWithSetTransformerInitialization: RNN model with adaptive inital hidden state represented by Deep Set or Set Transformer of input sequence. Hypothesis is that order-independent information (which is hard to extract for RNN) may improve the performance. Set Transformers usually beat Deep Sets on various benchmarks, hence one can hope that it will be more efficient in this model as well.  <br />

2. ConditionalRNNProcess: it is RNN model with Auxiliary Set Input (ASI) (in addition to usual vector sequence input). Via Set Transformer set input is mapped to a vector, from which adaptive inital hidden state of RNN is constructed. <br /> <br />

---
Theoretically RNNs are universal (in Turing sense) models of computation. Parameters of RNN is like a "program", which allows to map input sequences to output sequences in the desired way. Can RNN be "operating system", which can run different "programs" using the same weights, when those "programs" are represented as Auxiliary Set Input (like in ConditionalRNNProcess)?





