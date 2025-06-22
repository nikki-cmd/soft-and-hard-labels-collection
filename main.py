from llama_cpp import Llama
from huggingface_hub import hf_hub_download

from configs import configs
import data_tools
from data_tools.dataloader import Loader 
from data_tools.uploader import Uploader
import tools.get_labels as get
from tools.get_labels import getHL 


import numpy as np

loader = Loader(path=configs.dataset_path)
questions = loader.get_questions()

answers = []
for q in questions:
    answers.append(getHL(q)) 
    
uploader = Uploader(path=configs.dataset_path, answers=answers)
uploader.upload_answers()

