import torch
import sys
import argparse

# GPT score: https://github.com/simonepri/lm-scorer
from lm_scorer.models.auto import AutoLMScorer

parser = argparse.ArgumentParser(description='call all scores and compute the visual context based Belief-revision')
parser.add_argument('--c', default='c', help='language model score (GPT2)', type=str, required=True)
parser.add_argument('--output', default='', help='caption from the baseline (any)', type=str, required=True)
args = parser.parse_args()


scorer = AutoLMScorer.from_pretrained("gpt2-large")




# mean
def score(sentence):
    return scorer.sentence_score(sentence, reduce="mean")


file1 = []

with open(args.c, 'rU') as f1:
    for line1 in f1:
        file1.append(line1.rstrip())

result = []
# print caltion score to file
f = open(args.output, "w")
for i in range(len(file1)):
    temp = []
    messages = file1[i]
    w = score(messages)
    print(w)

    temp.append(w)

    result = file1[i] + ',' + str(w)

    f.write(result)
    f.write('\n')
    print(result)

f.close()

