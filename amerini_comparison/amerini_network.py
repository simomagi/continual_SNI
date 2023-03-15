
from torch import nn 
import torch 
from copy import deepcopy


class CNN_1D(nn.Module):
    def __init__(self) -> None:
        super(CNN_1D, self).__init__()
        self.relu = nn.ReLU()
         
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=100, kernel_size=3, padding="same")
        self.maxpoo11 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=100, out_channels=100,  kernel_size=3, padding="same")
        self.maxpoo12 = nn.MaxPool1d(kernel_size=2)
        self.fc = nn.Linear(22700, 256)   
        self.dropout3 = nn.Dropout(p=0.5)
        
        
    def forward(self, dct_hist):
        out = self.conv1(dct_hist)
        out = self.relu(out)
        out = self.maxpoo11(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpoo12(out)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.dropout3(out)
        
        return out 
    
        
        
class CNN_2D(nn.Module):
    def __init__(self) -> None:
  
        super(CNN_2D, self).__init__()
        self.relu = nn.ReLU()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3), padding="same") 
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), padding="same")
    
        self.max_pool1 = nn.MaxPool2d(kernel_size=(2,2))
        self.drop_1 = nn.Dropout(p=0.5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3,3), padding="same")
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3,3), padding="same")
        self.max_pool2 = nn.MaxPool2d(kernel_size=(2,2))
        self.drop_2 = nn.Dropout(p=0.5)
        self.fc = nn.Linear(262144, 256)
        self.drop_3 = nn.Dropout(p=0.5)
        
    def forward(self,noise_res):
        out = self.conv1(noise_res)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.max_pool1(out)
        out = self.drop_1(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.relu(out)
        out = self.max_pool2(out)
        out = self.drop_2(out)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.drop_3(out)
        return out 
    
    
       

class FusionNet(nn.Module):
    
    def __init__(self, n_classes=14) -> None:
        super(FusionNet, self).__init__()
        self.n_classes = n_classes
        self.cnn_1d = CNN_1D()
        self.cnn_2d = CNN_2D()
        self.head = nn.Sequential(nn.Linear(512,512), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(512, self.n_classes))
    
    def get_copy(self):
        """Get weights from the model"""
        return deepcopy(self.state_dict())

    def set_state_dict(self, state_dict):
        """Load weights into the model"""
        self.load_state_dict(deepcopy(state_dict))
        return

    
    def forward(self, noise_res, dct_hist):
        out_dct = self.cnn_1d(dct_hist)
        out_noise = self.cnn_2d(noise_res)
        out = torch.cat([out_dct, out_noise], dim=1)
        out = self.head(out)
        return out 
    
    
    
    
    
if __name__ == "__main__":
    
    #cnn_2d = CNN_2D()
    #
    #out = cnn_2d(noise_res)
    
    dct_hist = torch.randn((1, 1, 909))
    noise_res = torch.randn((1,1, 256,256))
    #cnn_1d = CNN_1D()
    
    #out = cnn_1d(dct_hist)
    net = FusionNet()
    # out = net(noise_res, dct_hist)
    #net = resnet50()
    
    trainable_params = sum(
	p.numel() for p in net.parameters() if p.requires_grad )
    
    print(trainable_params)

        
        