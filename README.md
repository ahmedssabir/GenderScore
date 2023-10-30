# Women Wearing Lipstick: Measuring the Bias Between an Object and Its Related Gender


<!--
 <img src="overview_bias.png"> 
 -->
 
 <img src="overview_bias.png"> 

  



## Overview 
In this paper, we investigate the impact of objects on gender bias in image captioning systems. Our results show that only gender-specific objects have a strong gender bias (_e.g._ <em>woman-lipstick</em>). In addition, 
      we propose a visual semantic-based gender score that measures the degree of bias and can be used as a plug-in for any image captioning system. Our experiments demonstrate the utility of the gender score, since 
      we observe that our score can measure the bias relation between a caption and its related gender; therefore, our score can be used as an additional metric to the existing Object Gender Co-Occ approach. 
  

 
 
 This repository contains the  implementation of the paper  [Women Wearing Lipstick: Measuring the Bias Between an Object and Its Related Gender](). EMNLP Findings 2023
 
 [![arXiv](https://img.shields.io/badge/arXiv-2310.00843-b31b1b.svg)](https://arxiv.org/abs/2810.00843) [![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](https://ahmed.jp/project_page/gender_score_2023/index.html)
 
 



 ## Quick Start 
For a quick start please have a look at this [project page](https://ahmed.jp/project_page/gender_score_2023/index.html) 
and [demo](https://huggingface.co/spaces/AhmedSSabir/Demo-for-Gender-Score), 

<!-- [huggingface demo (EN)](https://huggingface.co) --> 
 <!-- , and [huggingface demo (JP)](https://huggingface.co) -->

[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://github.com/ahmedssabir/GenderScore/tree/main/paper_demo)



### Requirements
- Python 3.7
- sentence_transformers 2.2.2

```
conda create -n gender_score python=3.7 anaconda
conda activate gender_score
pip install -U sentence-transformers 
``` 


```
parser.add_argument('--vis', default='visual-context_label.txt',help='class-label from the classifier (CLIP)', type=str, required=True)  
parser.add_argument('--vis_prob', default='visual-context.txt', help='prob from the classifier (Resent152/CLIP)', type=str, required=True) 
parser.add_argument('--c',  default='caption.txt', help='caption from the baseline (any)', type=str, required=True) 
parser.add_argument('--GPT2model', default="gpt2", help='gpt2, gpt2-medium, gpt2-large, gpt2-xl, distilgpt2', type=str, required=False)  
parser.add_argument('--BERTmodel', default='roberta-large-nli-stsb-mean-tokens', help='all-mpnet-base-v2, multi-qa-mpnet-base-dot-v1, all-distilroberta-v1', type=str, required=False) 
```


<!--
```
python model_GS.py --GPT2model distilgpt2 --BERTmodel stsb-distilbert-base --vis man_motorcycle_visual_context.txt --vis_prob  man_motorcycle_visual_context_prob.txt --c man_motorcycle_caption.txt
```


 ```
 python model.py --lm LM.txt --vis visual_context_label.txt --vis_prob visual_context_prob.txt --c caption.txt
 
``` 
-->

 
 For the demo (man-motorcycle)
<!--
 python model_GS.py --GPT2model distilgpt2 --BERTmodel stsb-distilbert-base --vis man_motorcycle_visual_context.txt --vis_prob  man_motorcycle_visual_context_prob.txt --c man_motorcycle.txt
 -->
 ```
python model_GS.py --GPT2model gpt2  --BERTmodel roberta-large-nli-stsb-mean-tokens --vis  man_motorcycle_GS/man_motorcycle_visual_context.txt --vis_prob  man_motorcycle_GS/man_motorcycle_visual_context_prob.txt --c man_motorcycle_GS/man_motorcycle_caption.txt
``` 

<!--
 ```
 python model.py  --vis motorcycle_gender_demo/man_motorcycle/man_motorcycle_visual_context.txt --vis_prob motorcycle_gender_demo/man_motorcycle/man_motorcycle_visual_context_prob.txt --c motorcycle_gender_demo/man_motorcycle/man_motorcycle.txt --output result.txt
 ```
 -->
 
 For Gender Score Estimation  

 <!-- 
 
 python model_GE.py --GPT2model distilgpt2  --BERTmodel stsb-distilbert-base --vis  man_motorcycle_GE/visual_context_demo_motorcycle.txt --vis_prob  man_motorcycle_GE/visual_context_prob_demo_motorcycle.txt --c man_motorcycle_GE/caption_demo_motorcycle_MASK.txt
 -->
## Example

<img align="center" width="400" height="300" src="COCO_val2014_000000175024.jpg">

```
python model_GE.py --GPT2model gpt2  --BERTmodel roberta-large-nli-stsb-mean-tokens --vis  man_motorcycle_GE/visual_context_demo_motorcycle.txt --vis_prob  man_motorcycle_GE/visual_context_prob_demo_motorcycle.txt --c man_motorcycle_GE/caption_demo_motorcycle_MASK.txt
```
input
```
Caption: a [MASK] riding a motorcycle on a road
visual context: motor scooter
visual context prob: 0.2183
```
To extract the visual context, please refer to this [page](https://github.com/ahmedssabir/Belief-Revision-Score/tree/main/model/Resent-152)

Output
```
# object-to-m bias 
caption_m a man riding a motorcycle on a road
LM: 0.12759140133857727
cosine score (sim): 0.5452305674552917
score_m: 0.45320714150193153

# object-to-w bias 
caption_w a woman riding a motorcycle on a road
LM: 0.11249390989542007
cosine score (sim): 0.5037289261817932
score_w: 0.39912252800731546

# most object-to-gender bias 
object_gender_caption: a man riding a motorcycle on a road
ratio_to_m: 53.17275201306536
ratio_to_w: 46.82724798693463
```

For the full demo 

 ```
python model_GE.py --GPT2model gpt2  --BERTmodel roberta-large-nli-stsb-mean-tokens --vis  man_motorcycle_GE/man_motorcycle_visual_context.txt --vis_prob  man_motorcycle_GE/man_motorcycle_visual_context_prob.txt --c man_motorcycle_GE/man_motorcycle_caption_MASK.txt
For cloze sentence-gender and the caption 
```

##  Cloze gender to caption
We want to explore all the cases and let the proposed distance/score decide which gender (i.e. bias) is in the  image based on a visual bias. In particular, inspired by the cloze probability last word completion task, we generate two identical sentences but with different gender, and then we compute the likelihood revisions between the  sentence-gender and the caption using the object probability. 


## Example

<!-- <img align="center" width="400" height="200" src="COCO_val2014_000000000042.jpg"> -->

<img align="center" width="400" height="300" src="COCO_val2014_000000066568.jpg">




```
python model_GS_cloze_gen.py  --GPT2model gpt2  --BERTmodel roberta-large-nli-stsb-mean-tokens --vis  GS-cloze-gender_demo/visual_context_demo.txt --vis_prob  GS-cloze-gender_demo/visual_context_prob_demo.txt --c GS-cloze-gender_demo/caption_demo.txt
 ```

<!-- Gender-Bias output on this BLIP baseline without gender in the image ```COCO_val2014_000000000042.jpg: a dog laying on top of a pile of shoes``` --> 
Gender-Bias output on this BLIP-2 baseline without gender in the image ```COCO_val2014_000000066568.jpg: two teddy bears sitting next to each other on a desk```



```
gender score m: 0.31807795162729513
gender score w: 0.3182200321892468
```

## Gender Score for non-visual task: gender bias in twitter 
In this section, we apply our proposed gender score to a subset of the Twitter   user gender classification dataset. We use a BERT based keyword extractor  to extract the biased context from the sentence (_e.g._ travel-man, woman-family), and we then employ  the cloze probability (Gonzalez-Marquez, 2007)  to extract the probability of the context. We observe  that there are some keywords that have a strong  bias: women are associated with keywords such as  novel, beauty and hometown. Meanwhile, men are  more frequently related to words such as gaming, coffee and inspiration.   

First we extract the context from each sentence with associate probability (_i.e._ tweet). We use [KeyBERT](https://github.com/MaartenGr/KeyBERT).

```
from keybert import KeyBERT

sentence = “Blood makes you related. Loyalty makes you family”

kw_model = KeyBERT()
kw_model.extract_keywords(sentence, keyphrase_ngram_range=(1, 1), stop_words=None)

Keywords
[('loyalty', 0.5631),
 ('blood', 0.5612),
 ('family', 0.4476),..]

Top-3 average bias_context_prob = 0.5239 
``` 

```
python GS_cloze_prob.py --sent GS_twitter_demo/sentence.txt --context_prob GS_twitter_demo/bias_context_prob.txt
```
<!-- python GS_cloze_prob.py --sent GS_twitter_demo/sentence.txt --context_prob GS_twitter_demo/ave_bias_context_prob.txt -->

```
gender score m: 0.0024223817787602246
gender score w: 0.005365207900075213
sentence: blood makes you related. Loyalty makes you family. [predicted bias-gender: w]
GT: m
```

The GS can be adapted for another language (tweet or short text). Therefore, We also [adopt this score](GS_twitter_demo_jp) to the Japanese language. <!-- Please refer to this demo for fast start [huggingface demo (JP)](https://huggingface.co) -->



<!--
## Citation


The details of this repo are described in the following paper. If you find this repo useful, please kindly cite it:

```bibtex
@article{sabir2022belief,
  title={Belief Revision based Caption Re-ranker with Visual Semantic Information},
  author={Sabir, Ahmed and Moreno-Noguer, Francesc and Madhyastha, Pranava and Padr{\'o}, Llu{\'\i}s},
  journal={arXiv preprint arXiv:2209.08163},
  year={2022}
}
```
--->
 
 ### Acknowledgement
The implementation of the Belief Revision Score relies on resources from <a href="https://github.com/simonepri/lm-scorer">lm-score</a>, <a href="https://github.com/huggingface/transformers">Huggingface Transformers</a>, and <a href="https://www.sbert.net/">SBERT</a>. We thank the original authors for their well organized codebase.
