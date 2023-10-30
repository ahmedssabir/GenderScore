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


def get_lines(file_path):
    with open(file_path) as f:
            return f.read().strip().split('\n')


def fill_mask(sentence, score_dict):
    words = sentence.split()
    max_score = -1
    max_word = None
    for word in score_dict:
        if score_dict[word] > max_score:
            max_score = score_dict[word]
            max_word = word
    filled_sentence = sentence.replace("[MASK]", max_word)
    return filled_sentence


# Bias Ratio to man  (paper)  
def Ratio_m(man, woman):
    man_score = float(man)
    woman_score= float(woman)
    
    return (man_score)/(man_score + woman_score) * 100 

# Bias Ratio to woman 
def Ratio_w(woman, man):
    man_score = float(man)
    woman_score= float(woman)
    
    return (woman_score)/(man_score + woman_score) * 100 

        
def Gender_score(caption, visual_context_label, visual_context_prob): 
    caption_emb = model.encode(caption, convert_to_tensor=True)
    visual_context_label_emb = model.encode(visual_context_label, convert_to_tensor=True)
    LM  = scorer.sentence_score(caption, reduce="mean")
    print("LM:", LM)
    sim = util.pytorch_cos_sim(caption_emb, visual_context_label_emb)    
    sim = sim.cpu().numpy()
    sim = sim.item()
    print("sim:", sim)
    score = pow(float(LM),pow((1-float(sim))/(1+ float(sim)),1-float(visual_context_prob)))
     
    return score

                             
output_path = 'gender_score_estimation_output.txt'

 
f=open(output_path, "w")
for i in range(len(get_lines(args.c))):
    temp =[]

    # visual context
    visual_context_label = get_lines(args.vis)[i]
    # visual_prob
    visual_context_prob = get_lines(args.vis_prob)[i]
    # gener MASk caption 
    caption_MASK = get_lines(args.c)[i]
    
    # fill with both gender 
    gender = 0 
    
    # man_score  
    caption_m = fill_mask(caption_MASK, {"man": gender })
    print('caption_m', caption_m)
    score_m =  Gender_score(caption_m, visual_context_label, visual_context_prob)
    print('score_m:', score_m)   
    
   
    # woman_score
    caption_w = fill_mask(caption_MASK, {"woman": gender })
    print('caption_w', caption_w)
    score_w =  Gender_score(caption_w, visual_context_label, visual_context_prob)
    print('score_w:', score_w)
    
    # most object-to-gender bias 
    object_gender_caption = fill_mask(caption_MASK, {"man": score_m, "woman": score_w})
    print('object_gender_caption:', object_gender_caption)
    gender_score = max(score_m, score_w)  
    
    ratio_to_m = Ratio_m(score_m, score_w)
    print('ratio_to_m:', ratio_to_m)
    
    ratio_to_w = Ratio_w(score_w, score_m)
    print('ratio_to_w:', ratio_to_w)
                         
    temp.append(object_gender_caption)
  
    
    result = ','.join((str(object_gender_caption), str(gender_score), 'ratio_to_m: '+str(ratio_to_m), 'ratio_to_w: '+str(ratio_to_w)))
    print('GE prediction:',result)  
    result = re.sub(r'\s*,\s*', ',', result) 

    f.write(result)
    f.write('\n')

        
f.close()

if __name__ == "__main__":
    pass 
    
    
