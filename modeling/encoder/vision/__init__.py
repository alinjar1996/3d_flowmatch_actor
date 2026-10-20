from .clip import load_clip


def fetch_visual_encoders(model_name, pretrained=True):
    if model_name == "clip":
        return load_clip(pretrained=pretrained)
    return None, None
