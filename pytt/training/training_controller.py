# trains model in sections, copying the checkpoint folder for each
# loads raw dataset(s)
# creates batcher using custom function
# for each section
#   creates/loads model, optimizer, batch_iterator
#   calls train
#   copy checkpoint directory

import shutil
import os
from pytt.logger import logger

class AbstractTrainingController:
    def __init__(self, static_parameters, sections, folder):
        self.static_parameters = static_parameters
        self.sections = sections
        if not os.path.exists(folder):
            os.mkdir(folder)
        self.folder = folder
        if not os.path.exists(os.path.join(folder, 'checkpoint')):
            os.mkdir(os.path.join(folder, 'checkpoint'))
        self.current_section = 0

    def train(self):
        self.logger.log("Static Parameters: "+str(self.static_parameters))
        while self.current_section < len(sections):
            dynamic_parameters = self.sections[self.current_section]
            self.log_section(self.current_section, dynamic_parameters)
            self.train_section({**self.static_parameters, **dynamic_parameters})
            self.save_section(self.current_section)
            self.current_section += 1

    def log_section(self, i, dynamic_parameters):
        logger.log("Section %i" % i)
        logger.log("\tDynamic"+str(dynamic_parameters))

    def train_section(self, parameters):
        raise NotImplementedError

    def save_section(self, i):
        main_folder = os.path.join(self.folder, 'checkpoint')
        section_folder = os.path.join(self.folder, 'checkpoint%i' % (i+1))
        shutil.copytree(main_folder, section_folder)
