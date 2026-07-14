import torch
import torch.nn as nn

class Architecture(nn.Modules):
    
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList()
        
    def add(self, *modules):
        new_sequential = nn.Sequential(*modules)
        self.blocks.append(new_sequential)
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x