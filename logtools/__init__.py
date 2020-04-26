import pickle
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np

class Logger():
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)  # Make sure it exists

    def log_file(self, _file_):
        file_name = os.path.basename(_file_)
        shutil.copyfile(_file_, os.path.join(self.log_dir, file_name))

    def log_variable(self, var, name):
        with open(os.path.join(self.log_dir, name) + '.pickle', 'bw') as pickle_file:
            pickle.dump(var, pickle_file)

    def log_variables(self, *args):
        """
        Log several variables
        :param args: var1, name1, var2, name2, var3, name3, ...
        :return:
        """
        if len(args) != 0:
            self.log_variable(args[0], args[1])
            self.log_variables(*args[2:])

    def log_text(self, text, fname):
        with open(os.path.join(self.log_dir, fname), 'w') as file:
            file.write(text)

    def log_figure(self, fig, name):
        fig.savefig(os.path.join(self.log_dir, name))