from tqdm import tqdm
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.requests import Request
from io import BytesIO
import requests
import torch
from PIL import Image, ImageOps
from diffusers import StableDiffusionInpaintPipeline
from torchvision.transforms import ToPILImage
from utils import preprocess, prepare_mask_and_masked_image, recover_image, resize_and_crop, read_firebase_db
import base64
import pdb
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1


app = FastAPI()

topil = ToPILImage()

pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    revision="fp16",
    torch_dtype=torch.float16,
    safety_checker=None,
)
pipe_inpaint = pipe_inpaint.to("cuda")

## Good params for editing that we used all over the paper --> decent quality and speed 
guidance_scale = 7.5
num_inference_steps = 100
seed = 329357

# If required, create a face detection pipeline using MTCNN:
mtcnn = MTCNN(select_largest=False, device='cuda')

# Create an inception resnet (in eval mode):
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def detect_faces(img):

    try:
        # Get cropped image tensor
        img_cropped = mtcnn(img)

        # Check if a face is detected
        if img_cropped is not None:
            # Convert the tensor to a PIL image
            img_cropped_np = img_cropped.squeeze().numpy().transpose(1, 2, 0)
            # Create a mask with the same shape as the original image
            mask = np.zeros_like(img_cropped_np, dtype=np.uint8)

            # Draw a rectangle (mask) around the detected face
            h, w, _ = img_cropped_np.shape
            mask[h//4:3*h//4, w//4:3*w//4, :] = 255  # Adjust the rectangle as needed

            # Apply the mask to the cropped image
            img_cropped_masked = np.where(mask != 0, img_cropped_np, 0)

            # Convert the masked image to a PIL image
            img_cropped_masked_pil = Image.fromarray(img_cropped_masked.astype('uint8'))
            # img_cropped_np = (img_cropped_np - img_cropped_np.min()) / (img_cropped_np.max() - img_cropped_np.min())

            # # Scale the values to [0, 255] and convert to uint8
            # img_cropped_np = (img_cropped_np * 255).astype('uint8')

            # img_cropped_pil = Image.fromarray(img_cropped_np)

            return img_cropped_masked_pil
        else:
            return None

    except Exception as e:
        print(f"Error: {e}")
        return None

def pgd(X, targets, model, criterion, eps=0.1, step_size=0.015, iters=40, clamp_min=0, clamp_max=1, mask=None):
    X_adv = X.clone().detach() + (torch.rand(*X.shape)*2*eps-eps).cuda()
    pbar = tqdm(range(iters))
    for i in pbar:
        actual_step_size = step_size - (step_size - step_size / 100) / iters * i  
        X_adv.requires_grad_(True)

        loss = (model(X_adv).latent_dist.mean - targets).norm()
        pbar.set_description(f"Loss {loss.item():.5f} | step size: {actual_step_size:.4}")

        grad, = torch.autograd.grad(loss, [X_adv])
        
        X_adv = X_adv - grad.detach().sign() * actual_step_size
        X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        X_adv.grad = None    
        
        if mask is not None:
            X_adv.data *= mask
            
    return X_adv

def get_target():
    target_url = 'https://www.rtings.com/images/test-materials/2015/204_Gray_Uniformity.png'
    response = requests.get(target_url)
    target_image = Image.open(BytesIO(response.content)).convert("RGB")
    target_image = target_image.resize((512, 512))
    return target_image


def immunize_fn(init_image, mask_image):
    with torch.autocast('cuda'):
        mask, X = prepare_mask_and_masked_image(init_image, mask_image)
        X = X.half().cuda()
        mask = mask.half().cuda()

        targets = pipe_inpaint.vae.encode(preprocess(get_target()).half().cuda()).latent_dist.mean

        adv_X = pgd(X, 
                    targets = targets,
                    model=pipe_inpaint.vae.encode, 
                    criterion=torch.nn.MSELoss(), 
                    clamp_min=-1, 
                    clamp_max=1,
                    eps=0.12, 
                    step_size=0.01, 
                    iters=200,
                    mask=1-mask
                   )

        adv_X = (adv_X / 2 + 0.5).clamp(0, 1)
        
        adv_image = topil(adv_X[0]).convert("RGB")
        adv_image = recover_image(adv_image, init_image, mask_image, background=True)
        return adv_image        

def generate_mask(image_size, initial_value=0):
    # Create a NumPy array for the mask
    mask = np.full(image_size, initial_value, dtype=np.uint8)
    mask = ImageOps.invert(Image.fromarray(mask).convert('RGB'))
    return mask

# Function to convert image to base64
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Function to convert PIL image to bytes
def image_to_bytes(image):
    with BytesIO() as byte_io:
        image.save(byte_io, format="JPEG")
        return byte_io.getvalue()

def process_image(image, prompt, seed, guidance_scale, num_inference_steps, immunize=False):

    torch.manual_seed(seed)
    init_image =  Image.fromarray(np.array(image))
    detected_face_img = detect_faces(init_image)
    init_image = resize_and_crop(init_image, (512,512))

    mask_image = generate_mask(init_image.size)
    
    mask_image = ImageOps.invert(detected_face_img).convert('RGB')
    mask_image = resize_and_crop(mask_image, (512,512))
    
  
    
    if immunize:
        immunized_image = immunize_fn(init_image, mask_image)
        
    image_edited = pipe_inpaint(prompt=prompt, 
                        image=init_image if not immunize else immunized_image, 
                        mask_image=mask_image, 
                        height = init_image.size[0],
                        width = init_image.size[1],
                        eta=1,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        ).images[0]
        
    image_edited = recover_image(image_edited, init_image, mask_image)

    if immunize:
        # Return the immunized image
        return [image_to_bytes(immunized_image), image_to_bytes(image_edited)]
    else:
        return [image_to_bytes(init_image), image_to_bytes(image_edited)]
    
### Routes ####

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/upload")
def upload_file(request: Request, payload: dict):
    try:
        global immunized_img_base64, edited_img_base64

        immunized_img_base64, edited_img_base64 = None, None

        # Extract image and prompt from the payload
        image_url = payload.get("img")
        prompt = payload.get("prompt")
        immunize = payload.get("immunize")

        if immunize == "yes":
            immunize=True
        else:
            immunize=False


        # Download the image from the URL
        response = requests.get(image_url)
        response.raise_for_status()  # Check if the request was successful

        # Process the image using PIL (you can replace this with your own processing logic)
        image = Image.open(BytesIO(response.content))

        # Process image and store base64 in variable

        immunized_img_base64, edited_img_base64 = process_image(image, prompt, seed, guidance_scale, num_inference_steps, immunize)

        return {"message": "Image processed successfully."}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Route to get the processed image
@app.get("/get_immunized_image")
def get_processed_image():
    try:
        global immunized_img_base64

        if immunized_img_base64:
            return StreamingResponse(BytesIO(immunized_img_base64), media_type="image/jpeg")
        else:
            return JSONResponse(content={"message": "No processed image available."}, status_code=404)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/get_edited_image")
def get_edited_image():
    try:
        global edited_img_base64

        if edited_img_base64:
            return StreamingResponse(BytesIO(edited_img_base64), media_type="image/jpeg")
        else:
            return JSONResponse(content={"message": "No processed image available."}, status_code=404)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.put("/reset_variables")
def reset_variables():
    try:
        global immunized_img_base64, edited_img_base64
        immunized_img_base64, edited_img_base64 = None, None
        return {"message": "Varaibles reset successfully"}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    

if __name__ == "__main__":
    

    uvicorn.run(app, host="0.0.0.0", port=7860)

