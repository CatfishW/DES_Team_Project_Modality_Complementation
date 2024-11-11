import torch
from model.model import tokenizer
from colorama import Fore, Style, init
from config.config import cfg
import numpy as np
import evaluate
from transformers import DefaultDataCollator

