import configparser

class Predictor():

    def __init__(self) -> None:
        self.config = configparser.ConfigParser()
        self.config.read("config.ini")