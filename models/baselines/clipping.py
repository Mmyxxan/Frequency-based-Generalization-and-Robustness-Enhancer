import torch, clip

class LinearClassifier(torch.nn.Module):
    def __init__(self, dim, num_labels=2):
        super(LinearClassifier, self).__init__()
        # torch.set_default_dtype(torch.float16)
        self.num_labels = num_labels
        self.linear = torch.nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)
        # linear layer
        return self.linear(x)
        
class clipmodel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor, self.preprocess = clip.load("ViT-L/14", device="cpu") # self.preprecess will not be used during training, which is handled in Dataset class 
        # self.fc = nn.Linear(768, 2)
        self.fc = LinearClassifier(768, 2)

    def forward(self, x):
        # with torch.no_grad():
        intermediate_output = self.feature_extractor.encode_image(x)
        output = self.fc(intermediate_output)
        return output
