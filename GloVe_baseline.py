import gensim  

#model= load_glove_model("glove.840B.300d.txt", 300)

import torch
import torchtext
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--gender',  default='gender', help='beam serach', type=str,required=True)
parser.add_argument('--vis', default='visual_context_label.txt', help='visual_context from ResNet', type=str,required=True)
parser.add_argument('--output', default='', help='', type=str,required=True)
args = parser.parse_args()


glove = torchtext.vocab.GloVe(name="840B", # trained on Wikipedia 2014 corpus of 6 billion words
                              dim=300)   # embedding size = 100

#x = glove['cat']
#y = glove['dog']
#torch.cosine_similarity(x.unsqueeze(0), y.unsqueeze(0))

#glove_file = datapath('GNGloVe-300d-0/1b-vectors300-0.8-0.8.txt')
#model_path = datapath('glove.840B.300d.txt')

#glove_file = datapath(model_path)

#tmp_file = get_tmpfile("test_word2vec.txt")

#_ = glove2word2vec(glove_file, tmp_file)

#model = KeyedVectors.load_word2vec_format(tmp_file)


#a = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
#print(a)

#model.similarity('woman', 'man')

#model.similarity('woman', 'snowboard')
#model.similarity('man', 'snowboard')


#model.similarity('woman', 'snowboard')
#model.similarity('man', 'snowboard')

file1 = []

file2 = []


#with open('woman.txt','rU') as f:
with open(args.gender,'rU') as f:

#with open('person.txt','rU') as f:     	
    for line in f:
    
       file1.append(line.rstrip())


with open(args.vis,'rU') as f1:
    for line1 in f1: 
       file2.append(line1.rstrip())
     

resutl=[]

f=open(args.output, "w")




from sklearn.metrics.pairwise import cosine_similarity

for i in range(len(file1)):
    temp =[]
    x = glove[file1[i]]
    y = glove[file2[i]]     

    try :
        #w = model.similarity(file1[i],file2[i])
        #w = cosine_similarity( glove[file1[i]],  glove[file2[i]])
        w = torch.cosine_similarity(x.unsqueeze(0), y.unsqueeze(0))
    except KeyError : 
        #print('out_of_dict')
        w = 0  #OVV to 0 

  
    temp.append(w)
    result= file1[i]+','+file2[i]+','+str(w)
    #result = file1[i]+',',+file2[i]+','+(w)
    f.write(result)
    f.write('\n')
    print (result)
f.close()
