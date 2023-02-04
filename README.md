# echogen
Auto-regressive generator which takes a text file as input, where each line is assumed to be a training example, and generates / echos new examples of similar type. Under the hood it's a auto-regressive character-level language model, with a wide choice of models. For example, one could feed the model a list of baby names and it would generate ideas for new ones. This project is purely educational and was originally designed to give me a foundational understanding of the history of language and auto-regressive models. As a side quest, I implemented the MLP model without the use of loss.backwards() as a healthy reminder of how gradients flow under all the layers of abstractions in modern ML libraries. Fundamental understanding of the internals in deep neural networks is easily forgotten, especially with all the tools we have available at our fingertips today. 

Currently supported models:

bigram, following Daniel Jurafsky & James H. Martin 2021. <br /> 
MLP, following Bengio et al. 2003  <br /> 
CNN, following DeepMind WaveNet 2016. <br /> 
Transformer, following Vaswani et al. 2017. <br /> 

Future models:
RNN, following Mikolov et al. 2010  <br /> 
LSTM, following Graves et al. 2014. <br /> 
GRU, following Kyunghyun Cho et al. 2014. <br /> 
