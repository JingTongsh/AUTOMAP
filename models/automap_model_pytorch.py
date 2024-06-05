import torch
import torch.nn as nn
import torch.nn.functional as F

class AUTOMAP_Basic_Model(nn.Module):
    def __init__(self, config):
        super(AUTOMAP_Basic_Model, self).__init__()
        
        # Fully connected layers
        self.fc1 = nn.Linear(config.fc_input_dim, config.fc_hidden_dim)
        self.fc2 = nn.Linear(config.fc_hidden_dim, config.fc_output_dim)
        
        # Reshape and zero padding
        self.im_h = config.im_h
        self.im_w = config.im_w
        self.zero_padding = nn.ZeroPad2d(4)
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.deconv = nn.ConvTranspose2d(64, 1, kernel_size=7, stride=1, padding=3)
        
    def forward(self, x):
        # Forward pass through fully connected layers
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        
        # Reshape to (batch_size, 1, im_h, im_w)
        x = x.view(-1, 1, self.im_h, self.im_w)
        
        # Zero padding
        x = self.zero_padding(x)
        
        # Convolutional layers
        x = F.relu(self.conv1(x))
        c2 = F.relu(self.conv2(x))
        
        # Transpose convolutional layer
        x = self.deconv(c2)
        
        # Flatten the output
        output = x.view(-1, (self.im_h + 8) * (self.im_w + 8))
        
        return output

# Example configuration class
class Config:
    fc_input_dim = 1024
    fc_hidden_dim = 512
    fc_output_dim = 256
    im_h = 16
    im_w = 16


if __name__ == '__main__':
    # Initialize the model
    config = Config()
    model = AUTOMAP_Basic_Model(config)
    print(model)
    x = torch.randn(1, config.fc_input_dim)
    y = model(x)
    print(y.shape)

