# MacOS doesn't like this...
# import os
# os.environ["ARCADE_HEADLESS"] = "true"
import arcade
from arcade.experimental import Shadertoy
from PIL import Image, ImageOps
import numpy as np
import os
import glob

IMAGE = "./img_in/*"
IMG_SIZE = (460, 320)
OUT_PATH = "./img_out/"
NOISE_IMAGE = "./noise.png"
SHADER = "./shaders/rain.glsl"

def readImage(path):
    img = Image.open(path)
    img = img.resize(IMG_SIZE)
    img = img.rotate(180)
    img = ImageOps.mirror(img)
    img = np.asarray(img, dtype=np.uint8)
    return img

def toTexture(img, ctx):
    channels = img.shape[2]
    img_shape = img.shape[:2]
    img_shape = img_shape[::-1]
    tex = arcade.gl.Texture(
        ctx, 
        img_shape, 
        components=channels, 
        dtype="f1", 
        data=img.tobytes(order="C"), 
        samples=0
    )
    return tex

def main():
    # Setup Window
    window = arcade.open_window(*IMG_SIZE)

    # Setup Shader:
    shader = Shadertoy.create_from_file(window.get_size(), SHADER)
    shader.channel_1 = toTexture(readImage(NOISE_IMAGE), window.ctx)

    for fn in glob.glob(IMAGE):
        # Load Images
        img = readImage(fn)

        # Set shader data
        shader.channel_0 = toTexture(img, window.ctx)

        # Render image
        random_time = np.random.random() * 10
        shader.render(time=random_time)

        # Save image
        result = arcade.get_image(0, 0, IMG_SIZE[0]/2, IMG_SIZE[1]/2)
        filename = os.path.basename(fn).split(".")[0]
        out_path = f"{OUT_PATH}{filename}.png"
        result.save(out_path)

if __name__ == "__main__":
    main()