# DistilBERT Goodreads genre

## Model Description

The model can predict the genre of the book / text provided to it.
For the fine-tuning and classification, we have used distilBERT, as it is smaller in size compared to original BERT model and hence faster to train.
Also, BERT / distilBERT models give contextual emeddings.




## Uses

This finetuned model can be used for text classification. It has been trained to classify 8 different genres of text
The genres include:

- young adult
- comics_graphic
- history_biography
- mystery_thriller_crime
- fantasy_paranormal
- poetry
- romance
- children



### Data & Objective

For finetuning, I used good reads dataset https://mengtingwan.github.io/data/goodreads.html
The goal is to predict the genre of the book.



#### Hardware

- **Hardware Type:** GPU T4
- **Hours used:** ~ 15 Minutes
- **Cloud Provider:** Kaggle

#### Results
| Metric	| Score |
|-----------|-------|
| Accuracy  | 0.XX  |
| F1 Score  | 0.XX  |
| Eval Loss | 0.XX  |



## Important Links


- [HuggingFace](https://huggingface.co/computervisionpro/distilbert-goodreads-genres/tree/main)
- [WandB](https://wandb.ai/computervisionpro-na/mlops-assignment2)
- [Kaggle Notebook](https://www.kaggle.com/code/computervisionpro/mlops-assignment-2-g25ait2063-finetune)
