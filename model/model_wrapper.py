# Model
#   takes in inputs outputs a raw un-readable output to be post processed by the loss function, error function, or debatcher
#   takes in one parameters argument when created
#   can create a ModelInfo oject
# ModelInfo
#   contains parameters object and model state_dict
#   contains save function
#   contains load function
import torch
from torch import nn

class ModelWrapper:
	def __init__(self, model, devices=['cpu']):
		self.devices = devices
		self.device_models = [model.to(device) for device in devices]


	def __call__(self, batch):
		# TODO: implement this
		raise NotImplementedError
		n = len(self.device_models)
		if n > 1:
			batches = batch.split(n)
			outputs = [None]*n
			# TODO: do this in parallel
			for i,batch in enumerate(batches):
				batch = batch.to(self.devices[i])
				outputs[i] = self.device_models[i](batch)
				for n,p in self.device_models[i].named_parameters():
					if n not in grads.keys():
						grads[n] = 0
					grads[n] += p.grad
			# TODO: do this in parallel too?
			for model in self.device_models:
				for n,p in model.named_parameters():
					p.grad = grads[n]
			return outputs[0].__class__.combine(outputs)
		else:
			return self.device_models[0](batch.to(self.devices[0]))
