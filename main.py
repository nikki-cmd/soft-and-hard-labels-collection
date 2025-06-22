from llama_cpp import Llama
from huggingface_hub import hf_hub_download

from configs import configs
from data_tools.dataloader import Loader 
from data_tools.uploader import Uploader
import tools.get_hard_labels as get
from tools.get_hard_labels import getHL 
from tools.get_soft_labels import getSL 

import csv

import numpy as np

loader = Loader(path=configs.dataset_path)
questions = loader.get_questions()

#---Hard Labels---
answers = []
for q in questions:
    answers.append(getHL(q)) 
    
uploader = Uploader(path=configs.dataset_path, answers=answers)
uploader.upload_answers()

#---Soft Labels---

for q in questions:
    answer, softlabels = getSL(q)
    print(f"\nFinal answer: {answer}\n{'-'*50}")
    
    if softlabels is not None:
        csv_filename = "softlabels_matrix.csv"
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['token_idx'] + list(range(32000)))
            
            for i, row in enumerate(softlabels):
                writer.writerow([i] + list(row))
        
        print(f"Softlabels matrix saved to {csv_filename}")