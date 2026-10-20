import torch
from torch import nn
import transformers


class ClipTokenizer:

    def __init__(self):
        super().__init__()
        self.tokenizer = transformers.CLIPTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

    @torch.inference_mode()
    def __call__(self, instructions):
        return self.tokenizer(
            instructions,
            padding="longest",
            return_tensors="pt"
        )["input_ids"]


class ClipTextEncoder(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            self.model = transformers.CLIPTextModel.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
        else:
            # Full actor checkpoints already contain the frozen text weights.
            self.model = transformers.CLIPTextModel(transformers.CLIPTextConfig())

    def forward(self, tokens):
        return self.model(tokens).last_hidden_state
