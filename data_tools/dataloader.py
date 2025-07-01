import pandas as pd

class Loader():
    def __init__(self, path, start_question=1):
        self.data_path = path
        self.start_question = start_question
    
    def load_data(self):
        dataset = pd.read_csv(self.data_path)
        return dataset
    
    def get_questions(self):
        dataset = self.load_data()
        data = dataset[dataset['id'] >= self.start_question]
        return data[['id', 'question']]
