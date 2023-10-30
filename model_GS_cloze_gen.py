#!/usr/bin/env python3
import sys
import argparse
import torch
import re
import os
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from lm_scorer.models.auto import AutoLMScorer as LMScorer
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity


parser=argparse.ArgumentParser(description='call all scores and  fill the mask (gender) via Belief-revision')
parser.add_argument('--vis', default='visual-context_label.txt',help='class-label from the classifier (CLIP)', type=str, required=True)  
parser.add_argument('--vis_prob', default='visual-context.txt', help='prob from the classifier (Resent152/CLIP)', type=str, required=True) 
parser.add_argument('--c',  default='caption.txt', help='caption from the baseline (any)', type=str, required=True) 
parser.add_argument('--GPT2model', default="gpt2", help='gpt2, gpt2-medium, gpt2-large, gpt2-xl, distilgpt2', type=str, required=False)  
parser.add_argument('--BERTmodel', default='roberta-large-nli-stsb-mean-tokens', help='all-mpnet-base-v2, multi-qa-mpnet-base-dot-v1, all-distilroberta-v1', type=str, required=False) 
args = parser.parse_args()


device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(args.BERTmodel, device=device)
batch_size = 1
scorer = LMScorer.from_pretrained(args.GPT2model , device=device, batch_size=batch_size)




def cos_sim(a, b):
    return np.inner(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))


def get_lines(file_path):
    with open(file_path) as f:
            return f.read().strip().split('\n')

def Gender_score(caption, visual_context_label, visual_context_prob):
    
    caption_emb = model.encode(caption, convert_to_tensor=True)
    visual_context_label_emb = model.encode(visual_context_label, convert_to_tensor=True)
    LM  = scorer.sentence_score(caption, reduce="mean")
    #print("LM:", LM)
    sim = util.pytorch_cos_sim(caption_emb, visual_context_label_emb)    
    sim = sim.cpu().numpy()
    sim = sim.item()
    #print("sim:", sim)
    score = pow(float(LM),pow((1-float(sim))/(1+ float(sim)),1-float(visual_context_prob)))
     
    return score

                             
output_path = 'gender-caption_score_output.txt'
# compute visual context
f=open(output_path, "w")
for i in range(len(get_lines(args.c))):
    temp =[]

    visual_context_label = get_lines(args.vis)[i]
    visual_context_prob = get_lines(args.vis_prob)[i]
    caption = get_lines(args.c)[i]
   
   
    # Cloze sentence 
    man_sentence = "there is a man"
    woman_sentence = "there is a woman"
    
    score_m  =  Gender_score(caption, man_sentence, visual_context_prob)
    score_w  =  Gender_score(caption, woman_sentence, visual_context_prob)
    both_score = score_m + score_w
    #print("gender score m:", score_m) 
    #print("gender score w:", score_w)                 
    temp.append(both_score)
  
  
    result = ','.join((str(caption), 'man_score: '+str(score_m), 'woman_score: '+str(score_w)))         
    result = re.sub(r'\s*,\s*', ',', result) 

    f.write(result)
    f.write('\n')
f.close()


if __name__ == "__main__":
   # Gender_score(caption, visual_context_label, visual_context_prob)
    if score_m > score_w:
        print("caption:", caption, 'predicted bias-gender: man')
    else:
        print("caption:", caption, 'predicted bias-gender: woman')  
        
 
      
