import json
import argparse

class argrements():
    def __init__(self):
        self.videopath = ''
        self.step = ''
        self.epoch_num = ''
        self.output_path = ''
        self.gray_scale = ''
        self.sz_idx = ''
        self.lr = ''
        self.loss_func = ''
        self.predict_frame = ''
        self.parseJSON()
    
    def parseJSON(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('JSON', type= str, help='display an integer')
        args = parser.parse_args()

        with open(args.JSON+".json") as json_file:
            try:
                config = json.load(json_file)
                self.videopath = config["videopath"]
                self.output_path = config["output_path"]
                self.step = config["step"]        
                self.sz_idx = config["size_idx"]
                self.loss_func = config["loss_function"]
                self.epoch_num = config["epoch"]
                self.lr = config["learning_rate"]
                self.gray_scale = config["gray_scale"]
                self.predict_frame = config["predict_frame"]
                if self.gray_scale == 'True':
                    self.gray_scale = True
                else:
                    self.gray_scale = False
            except Exception as e:
                print(str(e))
