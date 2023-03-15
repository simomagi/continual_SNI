
from torch import nn 
import torch 
from copy import deepcopy
 
from torch import nn 
import torch 
from copy import deepcopy
from torchvision.models import resnet18
 
 



class IncrementalFusionNet(nn.Module):
    
    def __init__(self, n_classes=14) -> None:
        super(IncrementalFusionNet, self).__init__()
        self.n_classes = n_classes
        self.net_1d =   nn.Sequential(nn.Linear(909, 512), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(512, 256), nn.ReLU(),  
                                      nn.Dropout(p=0.5), nn.Linear(256, 256), nn.ReLU())
        backbone  = resnet18()
        backbone.fc = nn.Identity()
        backbone.conv1 =  nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn_2d = nn.Sequential(backbone, nn.Linear(512, 256), nn.ReLU())
        
        self.head = nn.Sequential(nn.Linear(512,512), nn.ReLU(), nn.Linear(512, self.n_classes))
    
    def get_copy(self):
        """Get weights from the model"""
        return deepcopy(self.state_dict())

    def set_state_dict(self, state_dict):
        """Load weights into the model"""
        self.load_state_dict(deepcopy(state_dict))
        return

    
    def forward(self, hpf, dct_hist):
        out_dct = self.net_1d(dct_hist)
        out_hpf = self.cnn_2d(hpf)
 
        out = torch.cat([out_dct, out_hpf], dim=1)
        out = self.head(out)
        return out 
    

if __name__ == "__main__":
    net = IncrementalFusionNet()
    
    hpf = torch.rand((1,1,256,256))
    dct_hist = torch.randn((1,1,909))
    print(net(hpf, dct_hist))
    print( sum(p.numel() for p in net.parameters() if p.requires_grad))
 
    
    
    