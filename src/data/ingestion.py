# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from PIL import Image
import torch
import torchvision.transforms as transforms
import random
import numpy as np

random.seed(42)

class Cutout(object):
    """Apply Cutout augmentation to a PIL Image by masking out a random square region."""
    def __init__(self, cutout_size=4):
        self.cutout_size = cutout_size

    def __call__(self, img):
        # Convert image to numpy array
        np_img = np.array(img)
        h, w, _ = np_img.shape

        # Choose random center for the cutout square
        center_x = random.randint(0, w)
        center_y = random.randint(0, h)

        # Compute the bounds of the square
        x1 = max(0, center_x - self.cutout_size // 2)
        y1 = max(0, center_y - self.cutout_size // 2)
        x2 = min(w, center_x + self.cutout_size // 2)
        y2 = min(h, center_y + self.cutout_size // 2)

        # Set the selected region to zero (black)
        np_img[y1:y2, x1:x2, :] = 0
        return Image.fromarray(np_img)

@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.option("--rotate", is_flag=True, help="Apply slight rotation to the image.")
@click.option(
    "--flip",
    type=click.Choice(["horizontal", "vertical"], case_sensitive=False),
    help="Flip the image.",
)
@click.option("--brightness", is_flag=True, help="Adjust brightness randomly.")
@click.option("--contrast", is_flag=True, help="Adjust contrast randomly.")
@click.option("--cutout", is_flag=True, help="Apply cutout augmentation to the image.")
def main(input_filepath, output_filepath, rotate, flip, brightness, contrast, cutout):
    """
    Runs data processing scripts to turn raw PNG images from input_filepath
    into transformed PNG images stored in output_filepath.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Processing image: {input_filepath}")

    
    input_path = Path(input_filepath)
    if input_path.suffix.lower() != ".png":
        logger.error("Only PNG images are supported!")
        return

    
    image = Image.open(input_filepath).convert("RGB")

   
    transform_list = []

    if rotate:
        degree = random.randint(-45, 45)
        transform_list.append(transforms.RandomRotation(degrees=15))  
    if flip == "horizontal":
        transform_list.append(
            transforms.RandomHorizontalFlip(p=1.0)
        )  
    elif flip == "vertical":
        transform_list.append(
            transforms.RandomVerticalFlip(p=1.0)
        )  
    if brightness:
        brightness = random.uniform(0.7, 1.3)
        transform_list.append(
            transforms.ColorJitter(brightness=brightness)
        )  
    if contrast:
        contrast = random.uniform(0.7, 1.3)
        transform_list.append(
            transforms.ColorJitter(contrast=contrast)
        ) 
    if cutout:
        transform_list.append(Cutout(cutout_size=6))
    
    if transform_list:
        transform_pipeline = transforms.Compose(transform_list)
        transformed_image = transform_pipeline(image)
    else:
        transformed_image = image  

    output_path = Path(output_filepath)
    transformed_image.save(output_path, format="PNG")
    logger.info(f"Saved transformed image: {output_filepath}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Find and load environment variables from .env
    load_dotenv(find_dotenv())

    main()
