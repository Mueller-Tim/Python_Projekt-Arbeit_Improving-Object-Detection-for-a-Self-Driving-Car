import os
import random
from PIL import Image


def generate_collage(image_dir, images_per_row, images_per_col, output_file="collage.png"):
    # Get all image paths from the directory
    image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if
                   img.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Check if there are enough images
    required_images = images_per_row * images_per_col
    if len(image_paths) < required_images:
        print(f"Not enough images in the directory. Found {len(image_paths)}, but need {required_images}.")
        return

    # Randomly select the required number of images
    selected_images = random.sample(image_paths, required_images)

    # Load the images
    images = [Image.open(img) for img in selected_images]

    # Resize all images to the size of the first image
    image_size = images[0].size
    images = [img.resize(image_size) for img in images]

    # Calculate the collage size
    collage_width = image_size[0] * images_per_row
    collage_height = image_size[1] * images_per_col
    collage = Image.new('RGB', (collage_width, collage_height))

    # Paste the images into the collage
    for idx, img in enumerate(images):
        row = idx // images_per_row
        col = idx % images_per_row
        x = col * image_size[0]
        y = row * image_size[1]
        collage.paste(img, (x, y))

    # Save the collage
    collage.save(output_file)
    print(f"Collage saved as {output_file}.")


# Example usage
if __name__ == "__main__":
    image_dir = r"E:\PA\data\cone_dataset\img_40"
    images_per_row = 3
    images_per_col = 3
    generate_collage(image_dir, images_per_row, images_per_col)
