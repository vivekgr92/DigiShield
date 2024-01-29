import base64
from io import BytesIO
import requests
import gradio as gr
import requests
import torch
from tqdm import tqdm
from PIL import Image, ImageOps
from diffusers import StableDiffusionInpaintPipeline
from torchvision.transforms import ToPILImage
from utils import preprocess, prepare_mask_and_masked_image, recover_image, resize_and_crop, read_firebase_db
import webbrowser
import pdb
import matplotlib.pyplot as plt

gr.close_all()
topil = ToPILImage()

pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    revision="fp16",
    torch_dtype=torch.float16,
    safety_checker=None,
)
pipe_inpaint = pipe_inpaint.to("cuda")

## Good params for editing that we used all over the paper --> decent quality and speed   
GUIDANCE_SCALE = 7.5
NUM_INFERENCE_STEPS = 100
DEFAULT_SEED = 1234

# Read from firebase db - stripe webhooks
credentials_path = "./serviceKey.json"




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

def get_image_download_link(img,filename,text):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href =  f'<a href="data:image/jpeg;base64,{img_str}" download="{filename}">{text}</a>'
    return href

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

def run(image, prompt, seed, guidance_scale, num_inference_steps, immunize=False):
    

    try:

        # stripe_webhook_data = read_firebase_db(credentials_path)
        stripe_webhook_data = True

        if not stripe_webhook_data:
            raise ValueError("Please pay to proceed")
        if stripe_webhook_data:

            if seed == '':
                seed = DEFAULT_SEED
            else:
                seed = int(seed)
            torch.manual_seed(seed)

            init_image = Image.fromarray(image['image'])
            init_image = resize_and_crop(init_image, (512,512))
            mask_image = ImageOps.invert(Image.fromarray(image['mask']).convert('RGB'))
            mask_image = resize_and_crop(mask_image, init_image.size)
            
            if immunize:
                immunized_image = immunize_fn(init_image, mask_image)
                # result = get_image_download_link(immunized_image,'immunized_image.jpg','test')
                # immunized_image.save("immunized_image.jpg")  # Save the immunized image to a file
                
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
                return [(immunized_image, 'Immunized Image'), (image_edited, 'Edited After Immunization')]
            else:
                return [(image_edited, 'Edited Image (Without Immunization)')]
    except Exception as e:
        # Display the error message to the Gradio user
        gr.feedback(str(e))
        return
    
def download_button_callback():
    # This function is called when the user clicks the download button
    file_path = "immunized_image.jpg"
    return f'<a href="{file_path}" download="{file_path}">Download Immunized Image</a>'
    # return "immunized_image.jpg"  # Provide the file name to be downloaded


def pay_button_click_handler():
    # Handle the button click event here
    # Replace 'https://example.com' with the actual URL you want to redirect to
    stripe_webhook_data = read_firebase_db(credentials_path)

    url_to_open = 'https://buy.stripe.com/test_00geXHdg2fBE0zmdQQ'
    webbrowser.open(url_to_open)
    

# description='''
# Prevent malicious actors from using your photo to create DeepFakes and spread false information. Protect yourself today!
# <br />
# '''

# examples_list = [
#                     ['./images/hadi_and_trevor.jpg', 'man attending a wedding', '329357', GUIDANCE_SCALE, NUM_INFERENCE_STEPS],
#                     ['./images/trevor_2.jpg', 'two men in prison', '329357', GUIDANCE_SCALE, NUM_INFERENCE_STEPS],
#                     ['./images/elon_2.jpg', 'man in a metro station', '214213', GUIDANCE_SCALE, NUM_INFERENCE_STEPS],
#                 ]


with gr.Blocks() as demo:
    # gr.HTML(value="""<h1 style="font-weight: 900; margin-bottom: 7px; margin-top: 5px;">
    #         DigiShield: Protect your photos online today from DeepFake technologies </h1><br>
    #     """)
    # gr.Markdown(description)
    # with gr.Accordion(label='How to use (step by step):', open=False):
    #     gr.Markdown('''
    #         *First, let's edit your image:*        
    #         + Upload an image (or select from the examples below)
    #         + Use the brush to mask the parts of the image you want to keep unedited (e.g., faces of people)
    #         + Add a prompt to guide the edit (see examples below)
    #         + Play with the seed and click submit until you get a realistic edit that you are happy with (we provided good example seeds for you below)

    #         *Now, let's immunize your image and try again:*
    #         + Click on the "Immunize" button, then submit.
    #         + You will get an immunized version of the image (which should look essentially identical to the original one) as well as its edited version (which should now look rather unrealistic)
    #     ''')

    # with gr.Accordion(label='Example (video):', open=False):
    #     gr.HTML('''
    #         <center>
    #         <iframe width="920" height="600" src="https://www.youtube.com/embed/aTC59Q6ZDNM">
    #         allow="fullscreen;" frameborder="0">
    #         </iframe>
    #         </center>
    #     '''
    #     )

    with gr.Row():  
        with gr.Column():
            imgmask = gr.ImageMask(label='Drawing tool to mask regions you want to keep, e.g. faces')
            prompt = gr.Textbox(label='Prompt', placeholder='A photo of a man in a wedding')
            seed = gr.Textbox(label='Seed (Change to get different edits)', placeholder=str(DEFAULT_SEED), visible=True)
            with gr.Accordion("Advanced Options", open=False):
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=25.0, value=GUIDANCE_SCALE, step=0.1)
                num_steps = gr.Slider(label="Number of Inference Steps", minimum=10, maximum=250, value=NUM_INFERENCE_STEPS, step=5)
            immunize = gr.Checkbox(label='Immunize', value=False)
            b1 = gr.Button('Submit')
        with gr.Column():
            genimages = gr.Gallery(label="Generated images", 
                       show_label=False,
                       show_download_button=False,
                       elem_id="gallery").style(grid=[1,2], height="auto")
            download_link=gr.Textbox("Download Link", placeholder="Download link will appear here", disabled=True)
            # b2 = gr.Button('Download')
           
    # b1.click(pay_button_click_handler)
   
   
    b1.click(run, [imgmask, prompt, seed, scale, num_steps, immunize], [genimages])
    # b2.click(download_button_callback)
    # examples = gr.Examples(examples=examples_list,inputs = [imgmask, prompt, seed, scale, num_steps, immunize],  outputs=[genimages], cache_examples=False, fn=run)


# demo.launch()
demo.launch(server_name='0.0.0.0', share=True, server_port=7860, inline=False)