import pandas as pd

class Uploader():
    def __init__(self, path, answers):
        self.data_path = path
        self.answers = answers
        
    def upload_answers(self):
        df = pd.read_csv(self.data_path)
        df['llama_attempt'] = self.answers
        df.to_csv("new_dataset.csv", index=False)
        