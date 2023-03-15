
from torch import nn 
from torchvision.models import resnet18

class Resnet18(nn.Module):
    
    def __init__(self, data_type, n_classes=14) -> None:
        super(Resnet18, self).__init__()
        self.n_classes = n_classes
        backbone  = resnet18()
        self.data_type = data_type
        if self.data_type != "rgb":
            backbone.conv1 =  nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head_var = "fc"
        self.fc = nn.Sequential(nn.Linear(512, self.n_classes))
    
    def forward(self, hpf):
        out = self.backbone(hpf)
        return out 
    