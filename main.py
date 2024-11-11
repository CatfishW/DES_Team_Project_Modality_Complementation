
from tqdm import tqdm
from config.config import cfg
from colorama import Fore, Style, init
from engine import trainer
from dataset.dataset import tokenized_books_eval,tokenized_books_test
from model.model import tokenizer
if not cfg['model']['do_eval']:
    #run test dataset every 1000 steps
    trainer.train()
trainer.evaluate()
#Metrics Computations on Eval Dataset
results = trainer.predict(tokenized_books_eval)




