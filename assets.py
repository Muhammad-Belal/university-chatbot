import base64
import os

def get_logo_base64(filename):
    with open(filename, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    ext = os.path.splitext(filename)[1][1:]  # png
    return f"data:image/{ext};base64,{data}"  # ← yeh zaroori hai!

IUB_LOGO = get_logo_base64("iub_logo_clean.png")
BZU_LOGO = get_logo_base64("bzu_logo_clean.png")
