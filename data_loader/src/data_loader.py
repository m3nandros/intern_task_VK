import pandas as pd

class DataLoader:
    def __init__(self):
        pass

    def load_data(self):
        data = pd.read_csv('../../intern_task.csv')
        return data