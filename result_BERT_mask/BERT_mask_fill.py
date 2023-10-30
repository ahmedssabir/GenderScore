# source code 
#https://gist.github.com/yuchenlin/a2f42d3c4378ed7b83de65c7a2222eb2

import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import logging
import argparse
logging.basicConfig(level=logging.INFO)# OPTIONAL

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

parser=argparse.ArgumentParser()
parser.add_argument('--caption',  default='caption.txt', help='beam search k-1', type=str,required=True)
parser.add_argument('--output', default='', help='', type=str,required=True)
args = parser.parse_args()


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()
# model.to('cuda')  # if you have gpu


def predict_masked_sent(text, top_k=1):
    # Tokenize input
    text = "[CLS] %s [SEP]"%text
    tokenized_text = tokenizer.tokenize(text)
    masked_index = tokenized_text.index("[MASK]")
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    # tokens_tensor = tokens_tensor.to('cuda')    # if you have gpu

    # Predict all tokens
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]

    probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)
    top_k_weights, top_k_indices = torch.topk(probs, top_k, sorted=True)

    for i, pred_idx in enumerate(top_k_indices):
        predicted_token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
        token_weight = top_k_weights[i]
        print("[MASK]: '%s'"%predicted_token, " | weights:", float(token_weight))
        
        return predicted_token
        
output_path = args.output

file1 = []
with open(args.caption,'rU') as f1:
    for line1 in f1:
       file1.append(line1.rstrip())

result=[]
f=open(output_path, "w")
for i in range(len(file1)):
    temp =[]
    text  = file1[i]
    #predict_masked_sent("a [MASK]  in a black shirt and a green tie.", top_k=5)
    predicted_token_output = predict_masked_sent(text, top_k=1)
    temp.append(predicted_token_output)
    print(predicted_token_output)
 
    result= file1[i]+','+str(predicted_token_output)
    f.write(result)
    f.write('\n')
f.close()   
