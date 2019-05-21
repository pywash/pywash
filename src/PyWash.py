# pip install PyWash
# import PyWash as pw
# data = pw.AutoCleaner( [file_path] )

from src.SharedDataFrame import SharedDataFrame


class AutoCleaner():
    """
    Main interface for the PyWash package

    All functions that users can use come here
    """
    def __init__(self, file_path: str):
        self.data = SharedDataFrame(file_path)

    def get_classification(self):
        return self.data.analyze_data()
