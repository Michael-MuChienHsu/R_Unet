import json
import argparse

class argrements():
    def __init__(self):
        self.videopath = ''
        self.step = ''
  
        self.parseJSON()
    
    def parseJSON(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('JSON', type= str, help='display an integer')
        args = parser.parse_args()

        with open(args.JSON+".json") as json_file:
            try:
                config = json.load(json_file)
                self.videopath = config["videopath"]
                self.step = config["step"]
            except Exception as e:
                print(str(e))
