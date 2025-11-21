# coding=utf-8
# import the core dragonfly modules
from dragonfly.model import Model

# import the extension functions
from .model import model_to_vis_set, model_comparison_to_vis_set

# inject the methods onto the classes
Model.to_vis_set = model_to_vis_set
Model.to_vis_set_comparison = model_comparison_to_vis_set
