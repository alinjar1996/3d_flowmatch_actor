from .clip import ClipTextEncoder, ClipTokenizer


def fetch_text_encoders(model_name, pretrained=True):
    """Return encoder class and latent dimension."""
    if model_name == 'clip':
        return ClipTextEncoder(pretrained=pretrained), 512
    return None, None


def fetch_tokenizers(model_name):
    """Return tokenizer class."""
    if model_name == 'clip':
        return ClipTokenizer()
    return None
