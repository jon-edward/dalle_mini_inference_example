
from dalle_mini_toy import infer_images

import os
from pathlib import Path

IMAGES_DIR = Path(__file__).parent.joinpath("images").absolute()


if __name__ == '__main__':
    prompts = ["dog on a skateboard", "melancholy clown on a bicycle"]
    images = infer_images(prompts)

    try:
        os.mkdir(IMAGES_DIR)
    except OSError:
        pass

    for idx, img in enumerate(images):
        img.save(str(IMAGES_DIR.joinpath(f"{idx:03d}.png")))
