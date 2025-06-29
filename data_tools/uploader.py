import pandas as pd
import numpy as np
import configs.configs as cfg
from datetime import datetime
import os

class AnswersUploader():
    def __init__(self, path, answers):
        self.data_path = path
        self.answers = answers
        
    def upload_answers(self):
        df = pd.read_csv(self.data_path)
        df['llama_attempt'] = self.answers
        df.to_csv(cfg.answered_dataset_path, index=False)

    
class SoftsUploader():
    def __init__(self, distributions_matrix, question_id, current_time):
        self.distributions_matrix = distributions_matrix
        self.question_id = question_id
        self.current_time = current_time
            
    def upload_distributions(self):
        
        folder_path = f'softlabels/run{self.current_time}'
        os.makedirs(folder_path, exist_ok=True)
        
        np.savez(os.path.join(folder_path, f'question_{self.question_id}.npz'),*self.distributions_matrix)
    
        