import base64
import os

def get_logo_base64(filename):
    with open(filename, "rb") as f:
        return base64.b64encode(f.read()).decode()

IUB_LOGO = get_logo_base64("iub_logo_clean.png")
BZU_LOGO = get_logo_base64("bzu_logo_clean.png")
