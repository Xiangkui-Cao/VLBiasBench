import torch
from utils import get_config, merge_from_dict

class VLM_BaseModel(torch.nn.Module):
    def __init__(self, model_name, **kwargs):
        super().__init__()
        self.model_name = model_name
        self.config = get_config("./configs/model_config.yaml", model_name)
        if kwargs.get("cfg_options"):
            self.config = merge_from_dict(self.config, kwargs.get("cfg_options"))
        print(model_name, self.config)
        self.inst_pre = self.config.get("inst_pre", "")
        self.inst_suff = self.config.get("inst_suff", "")
        

    def forward(self, instruction, images, **kwargs):
        return self.generate(instruction, images, **kwargs)
    
    def generate(self, instruction, images, **kwargs):
        """
        instruction: (str) a string of instruction
        images: (list) a list of image urls
        Return: (str) a string of generated response
        """
        raise NotImplementedError


class T2I_BaseModel(torch.nn.Module):
    def __init__(self, model_name, **kwargs):
        super().__init__()
        self.config = get_config("./configs/model_config.yaml", model_name)
        if kwargs.get("cfg_options"):
            self.config = merge_from_dict(self.config, kwargs.get("cfg_options"))
        print(model_name, self.config)
        self.inst_pre = self.config.get("inst_pre", "")
        self.inst_suff = self.config.get("inst_suff", "")

    def forward(self, instruction, images, **kwargs):
        return self.generate(instruction, images, **kwargs)
    
    def generate(self, instruction, images, **kwargs):
        """
        instruction: (str) a string of instruction
        images: (list) a list of image urls
        Return: (str) a string of generated response
        """
        raise NotImplementedError