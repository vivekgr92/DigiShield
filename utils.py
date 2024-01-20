import torch
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms import ToPILImage, ToTensor
totensor = ToTensor()
topil = ToPILImage()

import firebase_admin
from firebase_admin import credentials, firestore

def read_firebase_db(credentials_path):
    # Initialize Firebase Admin SDK
    cred = credentials.Certificate(credentials_path)
    firebase_admin.initialize_app(cred)

    try:
        # Initialize Firestore
        db = firestore.client()

        # Reference to the "stripe-webhooks" collection
        collection_ref = db.collection("stripe-webhooks")

        # Get all documents in the collection
        docs = collection_ref.stream()

        # Iterate over documents
        for doc in docs:
            # Access document data
            data = doc.to_dict()
            print(f"Document ID: {doc.id}, Data: {data}")

            # Check if type is "charge.succeeded"
            if data.get("type") == "charge.succeeded":
                # Don't forget to close the Firebase app when done
                firebase_admin.delete_app(firebase_admin.get_app())
                return True

    except Exception as e:
        print(f"An error occurred: {e}")

    # Don't forget to close the Firebase app when done
    firebase_admin.delete_app(firebase_admin.get_app())
    return False




def resize_and_crop(img, size, crop_type="center"):
    '''Resize and crop the image to the given size.'''
    if crop_type == "top":
        center = (0, 0)
    elif crop_type == "center":
        center = (0.5, 0.5)
    else:
        raise ValueError

    width, height = size

    # Check if width or height is None
    if width is None:
        width = img.size[0]
    if height is None:
        height = img.size[1]

    return ImageOps.fit(img, (width, height), centering=center)


def recover_image(image, init_image, mask, background=False):
    image = totensor(image)
    mask = totensor(mask)[0]
    init_image = totensor(init_image)
    
    if background:
        result = mask * init_image + (1 - mask) * image
    else:
        result = mask * image + (1 - mask) * init_image
    return topil(result)


def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def prepare_mask_and_masked_image(image, mask):
    
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    return mask, masked_image
