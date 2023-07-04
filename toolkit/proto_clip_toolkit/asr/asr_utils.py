import json

class Config:
    def __init__(self, config_path):
        config = json.load(open(config_path))

        for (key , value) in config.items():
            self.__setattr__(key, value)