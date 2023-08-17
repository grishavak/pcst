import torch
import torch.nn as nn
import torch.nn.functional as F
from models import register_model


@register_model("custom")
class CustomModel(nn.Module):
	"""UNet as defined in https://arxiv.org/abs/1805.07709"""
	def __init__(self, *args):
		super(CustomModel, self).__init__()

		""" Add your code here (replace *args with your arguments) """
		

	@staticmethod
	def add_args(parser):
		"""Add model-specific arguments to the parser."""
		parser.add_argument("--bias", action='store_true', help="use residual bias")
		parser.add_argument("--residual", action='store_true', help="use residual connection")


	@classmethod
	def build_model(cls, args):
		return cls(args.bias, args.residual)


	def forward(self, x):

		""" Add your code here """

		return out
