# trains model in sections, copying the checkpoint folder for each
# loads raw dataset(s)
# creates batcher using custom function
# for each section
#   creates/loads model, optimizer, batch_iterator
#   calls train
#   copy checkpoint directory

from logger import logger

class TrainingController:
    def __init__(self, static_parameters, sections, folder):
        self.static_parameters = static_parameters
        self.sections = sections
        self.folder = folder
        self.trainer = None

    def train(self):
        self.logger.log("Static Parameters: "+str(self.static_parameters))
        for i,dynamic_parameters in enumerate(self.sections):
            self.log_section(i, dynamic_parameters)
            self.trainer = self.setup({**self.static_parameters, **dynamic_parameters})
            self.trainer.train()
            self.save_section(i)

    def log_section(self, i, dynamic_parameters):
        logger.log("Section %i" % i)
        logger.log("\tDynamic"+str(dynamic_parameters))

    def setup(self, parameters):
        raise NotImplementedError

    def save_section(self):
        pass
