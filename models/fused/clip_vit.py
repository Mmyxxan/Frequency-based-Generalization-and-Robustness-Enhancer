from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from .backbone import Backbone
from torchvision import transforms

class CLIPViT(Backbone):

    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )

    @staticmethod
    def preprocess(image_tensor):
        return CLIPViT.normalize(image_tensor.clone())

    def __init__(self, model_name="openai/clip-vit-large-patch14", freeze=True, pretrained=True):
        super().__init__()
        if pretrained:
            self.model = CLIPVisionModel.from_pretrained(model_name)
        else:
            # Create a config object to initialize the model
            config = CLIPVisionConfig.from_pretrained(model_name)  # Load the config from Hugging Face
            self.model = CLIPVisionModel(config)  # Pass the config to the model constructor

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self._out_features = self.model.config.hidden_size # usually 1024 for large

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        return outputs.last_hidden_state[:, 0] # CLS token
