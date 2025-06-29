import pandas as pd

class Loader():
    def __init__(self, path):
        self.data_path = path
    
    def load_data(self):
        dataset = pd.read_csv(self.data_path)
        print(dataset.columns)
        return dataset
    
    def get_questions(self):
        dataset = self.load_data()
        return dataset[['id', 'question']]
