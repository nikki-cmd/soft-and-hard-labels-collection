import pandas as pd
import numpy as np

class Uploader():
    def __init__(self, path, answers, distributions_matrix):
        self.data_path = path
        self.answers = answers
        self.distributions_matrix = distributions_matrix
        
    def upload_answers(self):
        df = pd.read_csv(self.data_path)
        df['llama_attempt'] = self.answers
        df.to_csv("new_dataset.csv", index=False)
        
    def upload_distributions(self):
        print(len(self.distributions_matrix))
        np.savez("softlabels_numpy_zip.npz", *self.distributions_matrix)
        