## Gender score  for twitter  (Japanese)


### Requirements
- Python 3.7
- sentence_transformers 2.2.2

```
conda create -n gender_score python=3.7 anaconda
conda activate gender_score_j
pip install -U sentence-transformers
pip install protobuf==3.20
``` 

```
sentence = “これこれ！！なっちょのインスタ開設はこれがあるから尚幸せなのよ！” # "so happy that Nacho is opening an Instagram"
Bias_context ='インスタ開設' # "opening an Instagram" 
```
```
python GS_cloze_prob_jp.py --sent sentence.txt --context_prob ave_bias_context_prob.txt
```
output
```
sentence: これこれ！！なっちょのインスタ開設はこれがあるから尚幸せなのよ！ predicted bias-gender: woman 
GT = woman 
```
