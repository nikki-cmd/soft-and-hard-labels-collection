import pandas as pd
import numpy as np
import configs.configs as cfg

class AnswersUploader():
    def __init__(self, path, answers):
        self.data_path = path
        self.answers = answers
        
    def upload_answers(self):
        df = pd.read_csv(self.data_path)
        df['llama_attempt'] = self.answers
        df.to_csv(cfg.answered_dataset_path, index=False)

    
class SoftsUploader():
    def __init__(self, distributions_matrix, question_id):
        self.distributions_matrix = distributions_matrix
        self.question_id = question_id
            
    def upload_distributions(self):
        np.savez(f'softlabels/question_{self.question_id}.npz', *self.distributions_matrix)
    
        