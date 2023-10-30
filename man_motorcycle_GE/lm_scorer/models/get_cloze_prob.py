
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import re

def Sort_Tuple(tup):  
  
	# (Sorts in descending order)  
	tup.sort(key = lambda x: x[1])  
	return tup[::-1]


def softmax(x):
	exps = np.exp(x)
	return np.divide(exps, np.sum(exps))
	
# Load pre-trained model 
model = GPT2LMHeadModel.from_pretrained('distilgpt2', output_hidden_states = True, output_attentions = True)
model.eval()
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')


def cloze_prob(text):

	whole_text_encoding = tokenizer.encode(text)
	# Parse out the stem of the whole sentence (i.e., the part leading up to but not including the critical word)
	text_list = text.split()
	stem = ' '.join(text_list[:-1])
	stem_encoding = tokenizer.encode(stem)
	# cw_encoding is just the difference between whole_text_encoding and stem_encoding
	# note: this might not correspond exactly to the word itself
	cw_encoding = whole_text_encoding[len(stem_encoding):]
	# Run the entire sentence through the model. Then go "back in time" to look at what the model predicted for each token, starting at the stem.
	# Put the whole text encoding into a tensor, and get the model's comprehensive output
	tokens_tensor = torch.tensor([whole_text_encoding])
	
	with torch.no_grad():
		outputs = model(tokens_tensor)
		predictions = outputs[0]   

	logprobs = []
	# start at the stem and get downstream probabilities incrementally from the model(see above)
	start = -1-len(cw_encoding)
	for j in range(start,-1,1):
			raw_output = []
			for i in predictions[-1][j]:
					raw_output.append(i.item())
	
			logprobs.append(np.log(softmax(raw_output)))
			
	# if the critical word is three tokens long, the raw_probabilities should look something like this:
	# [ [0.412, 0.001, ... ] ,[0.213, 0.004, ...], [0.002,0.001, 0.93 ...]]
	# Then for the i'th token we want to find its associated probability
	# this is just: raw_probabilities[i][token_index]
	conditional_probs = []
	for cw,prob in zip(cw_encoding,logprobs):
			conditional_probs.append(prob[cw])
	# now that you have all the relevant probabilities, return their product.
	# This is the probability of the critical word given the context before it.

	return np.exp(np.sum(conditional_probs))


	
context_exp_1 = 'do you believe in time travel' 
context_exp_2 ='nothing more important to me than my family'
context_1 = cloze_prob(context_exp_1)
context_2= cloze_prob(context_exp_2)

print(context_exp_1, context_1)
print(context_exp_2, context_2)
