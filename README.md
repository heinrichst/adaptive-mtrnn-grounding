# Adaptive MTRNN Grounding
Computational model as used in the paper:

Crossmodal Language Grounding in an Embodied Neurocognitive Model
by Stefan Heinrich, Yuan Yao, Tobias Hinz, Zhiyuan Liu, Thomas Hummel, Matthias Kerzel, Cornelius Weber, and Stefan Wermter

Contact: heinrich@informatik.uni-hamburg.de

Reference:  
@article{Heinrich2020AdaptiveMTRNNGrounding,
	author       = {Stefan Heinrich and Yuan Yao and Tobias Hinz and Zhiyuan Liu and Thomas Hummel and Matthias Kerzel and Cornelius Weber and Stefan Wermter},
	title        = {Crossmodal Language Grounding in an Embodied Neurocognitive Model},
	journal      = {Frontiers in Neurorobotics},
	number       = {14},
	volume       = {52},
	publisher    = {Frontiers Media S.A.},
	year         = {2020},
	doi          = {10.3389/fnbot.2020.00052}
}

## Model
The adaptive MTRNN grounding model is an end-to-end associator model that maps temporally dynamic multimodal perception to a temporally dynamic verbal description using adaptive MTRNNs with context abstraction for input and an adaptive MTRNN with context bias, where the abstracted context and context biases are connected to form cell assemblies.

####Usage:  
Use the MultiMTRNN models in a **tensorflow** version 1.15 environment with tensorflow-**keras**. For documentation on variable connotation see MultyMTRNN.py and mtrnn.py/xmtrnn.py. 

MultiMTRNN.MultiMTRNN() builds the keras model with default parameters from the paper.

##Data
The model was tested for embodied language grounding on the Embodied Multi-modal Interaction in Language learning (EMIL) data collection. The data can get downloaded via: https://www.inf.uni-hamburg.de/en/inst/ab/wtm/research/corpora.html

##Funding
Partial support from the German Research Foundation (DFG) and the National Science Foundation of China (NSFC) under project Crossmodal Learning (TRR-169).

##Achknowledgements
Thanks to Shreyans Bhansali and Erik Strahl for support in the data collection as well as the NVIDIA Corporation for support with a GPU grant.