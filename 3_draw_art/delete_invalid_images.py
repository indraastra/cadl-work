import os
import sys

from PIL import Image, ImageOps


def delete_invalid_images(image_dir):
    valid = 0
    invalid = 0
    for f in os.listdir(image_dir):
        is_invalid = False
        path = os.path.join(image_dir, f)
        try:
            img = ImageOps.fit(Image.open(path), (32, 32))
            w, h = img.size
            mode = img.mode
            if w == 0 or h == 0 or mode != "RGB":
                is_invalid = True
        except:
            is_invalid = True
        if is_invalid:
            print("Deleting:", path)
            os.remove(path)
            invalid += 1
        else:
            valid += 1
    print("{} valid files, {} invalid files".format(valid, invalid))


if __name__ == '__main__':
    image_dir = sys.argv[1]
    delete_invalid_images(image_dir)
