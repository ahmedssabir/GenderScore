#!/usr/bin/env python3
import sys
import argparse
import torch
import numpy as np
import re
import os
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from lm_scorer.models.auto import AutoLMScorer as LMScorer
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2Tokenizer, GPT2LMHeadModel


parser=argparse.ArgumentParser(description='call all scores and  predict (the bias gender) via Belief-revision')
#parser.add_argument('--vis', default='visual-context_label.txt',help='class-label from the classifier (CLIP)', type=str, required=True)  
parser.add_argument('--context_prob', default='context_prob.txt', help='from the keyword extracter', type=str, required=True) 
parser.add_argument('--sent',  default='tweet.txt', help='sentence tweet', type=str, required=True) 
parser.add_argument('--GPT2model', default="gpt2", help='gpt2, gpt2-medium, gpt2-large, gpt2-xl, distilgpt2', type=str, required=False)  
parser.add_argument('--BERTmodel', default='roberta-large-nli-stsb-mean-tokens', help='all-mpnet-base-v2, multi-qa-mpnet-base-dot-v1, all-distilroberta-v1', type=str, required=False) 
args = parser.parse_args()



model = GPT2LMHeadModel.from_pretrained(args.GPT2model, output_hidden_states = True, output_attentions = True)

model.eval()
#tokenizer =  gr.Interface.load('huggingface/distilgpt2')

#tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
tokenizer = GPT2Tokenizer.from_pretrained(args.GPT2model)


device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_sts = SentenceTransformer(args.BERTmodel, device=device)
# batch_size = 1
# scorer = LMScorer.from_pretrained(args.GPT2model , device=device, batch_size=batch_size)


def cos_sim(a, b):
    return np.inner(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))

def Sort_Tuple(tup):  
  
	# (Sorts in descending order)  
	tup.sort(key = lambda x: x[1])  
	return tup[::-1]

def softmax(x):
	exps = np.exp(x)
	return np.divide(exps, np.sum(exps))


def get_lines(file_path):
    with open(file_path) as f:
            return f.read().strip().split('\n')


def cloze_prob(text):

	whole_text_encoding = tokenizer.encode(text)
	text_list = text.split()
	stem = ' '.join(text_list[:-1])
	stem_encoding = tokenizer.encode(stem)
	cw_encoding = whole_text_encoding[len(stem_encoding):]
	# Run the entire sentence through the model. Then go "back in time" to look at what the model predicted for each token, starting at the stem.
	# Put the whole text encoding into a tensor, and get the model's comprehensive output
	tokens_tensor = torch.tensor([whole_text_encoding])
	
	with torch.no_grad():
		outputs = model(tokens_tensor)
		predictions = outputs[0]   

	logprobs = []
	start = -1-len(cw_encoding)
	for j in range(start,-1,1):
			raw_output = []
			for i in predictions[-1][j]:
					raw_output.append(i.item())
	
			logprobs.append(np.log(softmax(raw_output)))
			
	conditional_probs = []
	for cw,prob in zip(cw_encoding,logprobs):
			conditional_probs.append(prob[cw])
	# This is the probability of the critical word given the context before it.
	return np.exp(np.sum(conditional_probs))


def Gender_score(sentence, cloze_sentence , context_prob):
    
    sentence_emb = model_sts.encode(sentence, convert_to_tensor=True)
    cloze_sentence_emb = model_sts.encode(cloze_sentence, convert_to_tensor=True)
    LM = cloze_prob(sentence)
    #LM  = scorer.sentence_score(caption, reduce="mean")
    #print("LM:", LM)
    sim = util.pytorch_cos_sim(sentence_emb, cloze_sentence_emb)    
    sim = sim.cpu().numpy()
    sim = sim.item()
    #print("sim:", sim)
    score = pow(float(LM),pow((1-float(sim))/(1+ float(sim)),1-float(context_prob)))
     
    return score

                             
output_path = 'GS-sentence-to-context_output.txt'
# compute visual context
f=open(output_path, "w")
for i in range(len(get_lines(args.sent))):
    temp =[]

    #sentence_context = get_lines(args.vis)[i]
    context_prob = get_lines(args.context_prob)[i]
    sentence = get_lines(args.sent)[i]
   
   
    # Cloze sentence 
    man_sentence = 'this tweet was written by a man'
    woman_sentence ='this tweet was written by a woman'

    
    score_m  =  Gender_score(sentence , man_sentence, context_prob)
    score_w  =  Gender_score(sentence , woman_sentence, context_prob)
    both_score = score_m + score_w
    #print("gender score m:", score_m) 
    #print("gender score w:", score_w)                 
    temp.append(both_score)
  
  
    result = ','.join((str(sentence), 'man_score: '+str(score_m), 'woman_score: '+str(score_w)))         
    result = re.sub(r'\s*,\s*', ',', result) 

    f.write(result)
    f.write('\n')
f.close()


if __name__ == "__main__":
    if score_m > score_w:
        print("sentence:", sentence, 'predicted bias-gender: man')
    else:
        print("sentence:", sentence, 'predicted bias-gender: woman')  
   #print("Done!")
      
