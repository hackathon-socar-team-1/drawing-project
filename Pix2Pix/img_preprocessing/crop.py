import glob
from PIL import Image

files = glob.glob('./*.png')

i = 1041
for f in files:
    img = Image.open(f)

    img_resize_lanczos = img.resize((256, 256), Image.LANCZOS)
    img_resize_lanczos.save(f'./a_{i}.png')
    i += 1