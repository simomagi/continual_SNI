
from torch import nn 
import torch 

 
from torch import nn 
import torch 
from copy import deepcopy
from torchvision.models import resnet18

class FusionFc(nn.Module):
    
    def __init__(self,data_type, n_classes=14) -> None:
        super(FusionFc, self).__init__()
        self.n_classes = n_classes
        self.net_1d =   nn.Sequential(nn.Linear(909, 512), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(512, 256), nn.ReLU(),  
                                      nn.Dropout(p=0.5), nn.Linear(256, 256), nn.ReLU())
        backbone  = resnet18()
        
        backbone.fc = nn.Identity()
        self.data_type = data_type
        if self.data_type != "rgb":
            backbone.conv1 =  nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
        self.cnn_2d = nn.Sequential(backbone, nn.Linear(512, 256), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(512,512), nn.ReLU())
        self.head_var = "fc"
        self.fc  =  nn.Linear(512, self.n_classes)
        

    
    def forward(self, hpf, dct_hist):
        out_hpf = self.cnn_2d(hpf)
        out_dct = self.net_1d(dct_hist)
        out = torch.cat([out_dct, out_hpf], dim=1)
        out = self.fusion(out)
        return out 
    