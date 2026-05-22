# DistilBERT Goodreads genre (Classification)

## Model Description

The model can predict the genre of the book / text provided to it.
For the fine-tuning and classification, we have used distilBERT, as it is smaller in size compared to original BERT model and hence faster to train.
Also, BERT / distilBERT models give contextual emeddings.




## Uses

This finetuned model can be used for text classification. It has been trained to classify 8 different genres of books / text
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

For finetuning, we used good reads dataset which can be found here: https://mengtingwan.github.io/data/goodreads.html

The goal is to predict the genre of the book and to demonstrate how to train & deploy a model.

## Setup

1. Clone this repo: git clone https://github.com/computervisionpro/distilbert-goodreads-genre
2. Install python libraries: `pip install -r requirements.txt`
3. Upload the jupyter notebook file to Kaggle
4. Set environment variables, on Kaggle you can go to Add-ons and select Secrets. After that, you can use those keys with below lines: 
  - `os.environ['WANDB_API_KEY']=your_key`
  - `os.environ['HF_TOKEN']=your_token`
4. Run the notebook on Kaggle with GPU-T4 enabled
5. One can run the inference file `distilbert_inference.py` for testing the trained custom model

### Hardware

- **Hardware Type:** GPU T4
- **Hours used:** ~ 15 Minutes
- **Cloud Provider:** Kaggle



#### Final Result

| Metric	| Score |
|-----------|-------|
| Accuracy  | 0.583  |
| F1 Score  | 0.584  |
| Eval Loss | 2.35  |



## Important Links


- [HuggingFace](https://huggingface.co/computervisionpro/distilbert-goodreads-genres/tree/main)
- [WandB](https://wandb.ai/computervisionpro-na/mlops-assignment2)
- [Kaggle Notebook](https://www.kaggle.com/code/computervisionpro/mlops-assignment-2-g25ait2063-finetune)
- [Github](https://github.com/computervisionpro/distilbert-goodreads-genre)
