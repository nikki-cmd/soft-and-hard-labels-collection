import pandas as pd
import numpy as np
import os

class AnswersUploader():
    def __init__(self, path, answers, questions, tokenized_answers, dataset_name):
        self.data_path = path
        self.answers = answers
        self.questions = questions
        self.dataset_name = dataset_name
        self.tokenized_answers = tokenized_answers
        
    def upload_answers(self):
        df = pd.DataFrame({'questions':self.questions, 'model_attempt':self.answers, 'tokenized_model_attempt':self.tokenized_answers})
        df.to_csv(f"runs/hardlabels/{self.dataset_name}.csv", index=False)

    
class SoftsUploader():
    def __init__(self, distributions_matrix, indeces_matrix, question_id, current_time):
        self.distributions_matrix = distributions_matrix
        self.question_id = question_id
        self.current_time = current_time
        self.indeces_matrix = indeces_matrix
            
    def upload_distributions(self):
        folder_path = f'runs/softlabels/run{self.current_time}'
        os.makedirs(folder_path, exist_ok=True)
        
        np.savez(os.path.join(folder_path, f'question_{self.question_id}.npz'), 
                logits=np.array(self.distributions_matrix, dtype=object), 
                indeces=np.array(self.indeces_matrix, dtype=object))
    
        