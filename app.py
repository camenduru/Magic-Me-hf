import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
import gradio as gr
from glob import glob

import logging.config
LOGGING_CONFIG = {
    'version': 1,
    'formatters': {
        'default': {  # This is the formatter named 'default'
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'default',  # Reference to the 'default' formatter
        },
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['console'],
            'level': 'INFO',
        },
    },
}

# Assuming LOGGING_CONFIG is the dictionary defined above
logging.config.dictConfig(LOGGING_CONFIG)

def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    from main import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_custom_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_custom_nodes()


from nodes import (
    EmptyLatentImage,
    CheckpointLoaderSimple,
    NODE_CLASS_MAPPINGS,
    KSamplerAdvanced,
    MagicAlbum3DGaussianNoise,
    CLIPTextEncode,
    VAELoader,
    VAEDecode,
)



            


class MagicMeController:
    def __init__(self):
        self.id_embed_dir = "models/embeddings"
        self.save_dir = "output"
        self.id_embed_list = []
        self.woman_id_embed_list = ["beyonce", "hermione", "lifeifei", "lisa", "mona", "monroe", "taylor", "scarlett"]
        self.refresh_id_embed()
        self.update_id_embed(self.id_embed_list[0])
        with torch.inference_mode():
            vaeloader = VAELoader()
            self.vaeloader_2 = vaeloader.load_vae(
                vae_name="vae-ft-mse-840000-ema-pruned.safetensors"
            )

            checkpointloadersimple = CheckpointLoaderSimple()
            self.checkpointloadersimple_32 = checkpointloadersimple.load_checkpoint(
                ckpt_name="realisticVision_v51.safetensors"
            )


            ultralyticsdetectorprovider = NODE_CLASS_MAPPINGS[
                "UltralyticsDetectorProvider"
            ]()
            self.ultralyticsdetectorprovider_75 = ultralyticsdetectorprovider.doit(
                model_name="bbox/face_yolov8m.pt"
            )

            samloader = NODE_CLASS_MAPPINGS["SAMLoader"]()
            self.samloader_78 = samloader.load_model(
                model_name="sam_vit_b_01ec64.pth", device_mode="AUTO"
            )

            ade_animatediffuniformcontextoptions = NODE_CLASS_MAPPINGS[
                "ADE_AnimateDiffUniformContextOptions"
            ]()
            self.ade_animatediffuniformcontextoptions_102 = (
                ade_animatediffuniformcontextoptions.create_options(
                    context_length=16, context_stride=1, context_overlap=2, closed_loop=False,
                    context_schedule="uniform", fuse_method="flat"
                )
            )

            upscalemodelloader = NODE_CLASS_MAPPINGS["UpscaleModelLoader"]()
            self.upscalemodelloader_157 = upscalemodelloader.load_model(
                model_name="4xUltrasharpV10.pt"
            )

            ade_animatediffloraloader = NODE_CLASS_MAPPINGS["ADE_AnimateDiffLoRALoader"]()
            self.ade_animatediffloraloader_196 = ade_animatediffloraloader.load_motion_lora(
                lora_name="v2_lora_ZoomIn.ckpt", strength=0.6
            )

            impactint = NODE_CLASS_MAPPINGS["ImpactInt"]()
            self.impactint_204 = impactint.doit(value=16)

            self.ade_animatediffloaderwithcontext = NODE_CLASS_MAPPINGS[
                "ADE_AnimateDiffLoaderWithContext"
            ]()
            self.freeu_v2 = NODE_CLASS_MAPPINGS["FreeU_V2"]()
            self.tobasicpipe = NODE_CLASS_MAPPINGS["ToBasicPipe"]()
            self.frombasicpipe = NODE_CLASS_MAPPINGS["FromBasicPipe"]()
            self.bnk_getsigma = NODE_CLASS_MAPPINGS["BNK_GetSigma"]()
            self.emptylatentimage = EmptyLatentImage()
            self.magicalbum3dgaussiannoise = MagicAlbum3DGaussianNoise()
            self.bnk_injectnoise = NODE_CLASS_MAPPINGS["BNK_InjectNoise"]()
            self.ksampleradvanced = KSamplerAdvanced()
            self.vaedecode = VAEDecode()
            self.vhs_videocombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()
            self.impactsimpledetectorsegs_for_ad = NODE_CLASS_MAPPINGS[
                "ImpactSimpleDetectorSEGS_for_AD"
            ]()
            self.segsdetailerforanimatediff = NODE_CLASS_MAPPINGS["SEGSDetailerForAnimateDiff"]()
            self.segspaste = NODE_CLASS_MAPPINGS["SEGSPaste"]()
            self.segspreview = NODE_CLASS_MAPPINGS["SEGSPreview"]()
            self.ultimatesdupscale = NODE_CLASS_MAPPINGS["UltimateSDUpscale"]()
            self.imagecasharpening = NODE_CLASS_MAPPINGS["ImageCASharpening+"]()



    def refresh_id_embed(self):
        id_embed_list = glob(os.path.join(self.id_embed_dir, "*.pt"))
        self.id_embed_list = [os.path.basename(p) for p in id_embed_list]

    def update_id_embed(self, id_embed_dropdown):
        self.selected_id_embed = id_embed_dropdown
        return gr.Dropdown.update()    


    def run_once(self, prompt_text_box, negative_prompt_text_box, id_embed_dropdown):
        if self.selected_id_embed != id_embed_dropdown: self.update_id_embed(id_embed_dropdown)

        category = "woman" if self.selected_id_embed in self.woman_id_embed_list else "man"
        prompt = f"a photo of embedding:{self.selected_id_embed} {category} "  + prompt_text_box
        print("prompt:", prompt)
        print("negative_prompt_text_box:", negative_prompt_text_box)
        with torch.inference_mode():
            cliptextencode = CLIPTextEncode()
            cliptextencode_6 = cliptextencode.encode(
                text=negative_prompt_text_box,
                clip=get_value_at_index(self.checkpointloadersimple_32, 1),
            )
            cliptextencode_274 = cliptextencode.encode(
                text=prompt,
                clip=get_value_at_index(self.checkpointloadersimple_32, 1),
            )
            ade_animatediffloaderwithcontext_261 = (
                            self.ade_animatediffloaderwithcontext.load_mm_and_inject_params(
                                model_name="mm_sd_v15_v2.ckpt",
                                beta_schedule="autoselect",
                                motion_scale=1,
                                apply_v2_models_properly=True,
                                model=get_value_at_index(self.checkpointloadersimple_32, 0),
                                context_options=get_value_at_index(
                                    self.ade_animatediffuniformcontextoptions_102, 0
                                ),
                                motion_lora=get_value_at_index(self.ade_animatediffloraloader_196, 0),
                            )
                        )

            freeu_v2_151 = self.freeu_v2.patch(
                b1=1.1,
                b2=1.2,
                s1=0.9,
                s2=0.4,
                model=get_value_at_index(ade_animatediffloaderwithcontext_261, 0),
            )

            tobasicpipe_42 = self.tobasicpipe.doit(
                model=get_value_at_index(freeu_v2_151, 0),
                clip=get_value_at_index(self.checkpointloadersimple_32, 1),
                vae=get_value_at_index(self.vaeloader_2, 0),
                positive=get_value_at_index(cliptextencode_274, 0),
                negative=get_value_at_index(cliptextencode_6, 0),
            )

            frombasicpipe_52 = self.frombasicpipe.doit(
                basic_pipe=get_value_at_index(tobasicpipe_42, 0)
            )

            bnk_getsigma_254 = self.bnk_getsigma.calc_sigma(
                sampler_name="dpmpp_2m",
                scheduler="karras",
                steps=20,
                start_at_step=0,
                end_at_step=20,
                model=get_value_at_index(frombasicpipe_52, 0),
            )

            emptylatentimage_223 = self.emptylatentimage.generate(
                width=512, height=512, batch_size=get_value_at_index(self.impactint_204, 0)
            )

            magicalbum3dgaussiannoise_262 = self.magicalbum3dgaussiannoise.generate(
                width=512,
                height=512,
                batch_size=get_value_at_index(self.impactint_204, 0),
                seed=random.randint(1, 2**64),
                cov_factor=0.15,
            )

            bnk_injectnoise_253 = self.bnk_injectnoise.inject_noise(
                strength=get_value_at_index(bnk_getsigma_254, 0),
                latents=get_value_at_index(emptylatentimage_223, 0),
                noise=get_value_at_index(magicalbum3dgaussiannoise_262, 0),
            )

            ksampleradvanced_248 = self.ksampleradvanced.sample(
                add_noise="disable",
                noise_seed=random.randint(1, 2**64),
                steps=20,
                cfg=8,
                sampler_name="dpmpp_2m",
                scheduler="karras",
                start_at_step=0,
                end_at_step=20,
                return_with_leftover_noise="disable",
                model=get_value_at_index(frombasicpipe_52, 0),
                positive=get_value_at_index(frombasicpipe_52, 3),
                negative=get_value_at_index(frombasicpipe_52, 4),
                latent_image=get_value_at_index(bnk_injectnoise_253, 0),
            )

            vaedecode_10 = self.vaedecode.decode(
                samples=get_value_at_index(ksampleradvanced_248, 0),
                vae=get_value_at_index(frombasicpipe_52, 2),
            )

            vhs_videocombine_35 = self.vhs_videocombine.combine_video(
                frame_rate=8,
                loop_count=0,
                filename_prefix="orig",
                format="video/h264-mp4",
                pingpong=False,
                save_output=False,
                images=get_value_at_index(vaedecode_10, 0),
                unique_id=2001771405939721385,
            )

            impactsimpledetectorsegs_for_ad_156 = self.impactsimpledetectorsegs_for_ad.doit(
                bbox_threshold=0.5,
                bbox_dilation=0,
                crop_factor=3,
                drop_size=10,
                sub_threshold=0.5,
                sub_dilation=0,
                sub_bbox_expansion=0,
                sam_mask_hint_threshold=0.7,
                masking_mode="Pivot SEGS",
                segs_pivot="Combined mask",
                bbox_detector=get_value_at_index(self.ultralyticsdetectorprovider_75, 0),
                image_frames=get_value_at_index(vaedecode_10, 0),
                sam_model_opt=get_value_at_index(self.samloader_78, 0),
            )

            segsdetailerforanimatediff_41 = self.segsdetailerforanimatediff.doit(
                guide_size=512,
                guide_size_for=False,
                max_size=512,
                seed=random.randint(1, 2**64),
                steps=20,
                cfg=8,
                sampler_name="euler",
                scheduler="normal",
                denoise=0.8,
                refiner_ratio=0.2,
                image_frames=get_value_at_index(vaedecode_10, 0),
                segs=get_value_at_index(impactsimpledetectorsegs_for_ad_156, 0),
                basic_pipe=get_value_at_index(tobasicpipe_42, 0),
            )

            segspaste_49 = self.segspaste.doit(
                feather=5,
                alpha=255,
                image=get_value_at_index(vaedecode_10, 0),
                segs=get_value_at_index(segsdetailerforanimatediff_41, 0),
            )

            vhs_videocombine_51 = self.vhs_videocombine.combine_video(
                frame_rate=8,
                loop_count=0,
                filename_prefix="face_detailer",
                format="video/h264-mp4",
                pingpong=False,
                save_output=False,
                images=get_value_at_index(segspaste_49, 0),
                unique_id=7104489750160636615,
            )

            # segspreview_101 = self.segspreview.doit(
            #     alpha_mode=True,
            #     min_alpha=0.2,
            #     segs=get_value_at_index(impactsimpledetectorsegs_for_ad_156, 0),
            # )

            frombasicpipe_175 = self.frombasicpipe.doit(
                basic_pipe=get_value_at_index(tobasicpipe_42, 0)
            )

            ultimatesdupscale_172 = self.ultimatesdupscale.upscale(
                upscale_by=2,
                seed=random.randint(1, 2**64),
                steps=20,
                cfg=8,
                sampler_name="euler",
                scheduler="normal",
                denoise=0.2,
                mode_type="Linear",
                tile_width=512,
                tile_height=512,
                mask_blur=8,
                tile_padding=32,
                seam_fix_mode="None",
                seam_fix_denoise=1,
                seam_fix_width=64,
                seam_fix_mask_blur=8,
                seam_fix_padding=16,
                force_uniform_tiles=True,
                tiled_decode=False,
                image=get_value_at_index(segspaste_49, 0),
                model=get_value_at_index(frombasicpipe_175, 0),
                positive=get_value_at_index(frombasicpipe_175, 3),
                negative=get_value_at_index(frombasicpipe_175, 4),
                vae=get_value_at_index(frombasicpipe_175, 2),
                upscale_model=get_value_at_index(self.upscalemodelloader_157, 0),
            )

            imagecasharpening_183 = self.imagecasharpening.execute(
                amount=0.2, image=get_value_at_index(ultimatesdupscale_172, 0)
            )

            vhs_videocombine_176 = self.vhs_videocombine.combine_video(
                frame_rate=8,
                loop_count=0,
                filename_prefix="SR",
                format="video/h265-mp4",
                pingpong=False,
                save_output=True,
                images=get_value_at_index(imagecasharpening_183, 0),
                unique_id=5059112282155244564,
            )


        save_sample_path = sorted(glob(self.save_dir, 'SR*.mp4'))[-1]
    
        json_config = {
            "prompt": prompt,
            "n_prompt": negative_prompt_text_box,
            "id_embed_dropdown": id_embed_dropdown,
        }
        return gr.Video.update(value=save_sample_path), gr.Json.update(value=json_config)



import_custom_nodes()
c = MagicMeController()



css = """
.toolbutton {
    margin-buttom: 0em 0em 0em 0em;
    max-width: 2.5em;
    min-width: 2.5em !important;
    height: 2.5em;
}
"""


examples = [
    # 1-ToonYou
    [
        # "toonyou_beta3.safetensors", 
        # "mm_sd_v14.ckpt", 
        "masterpiece, best quality, 1girl, solo, cherry blossoms, hanami, pink flower, white flower, spring season, wisteria, petals, flower, plum blossoms, outdoors, falling petals, white hair, black eyes",
        "worst quality, low quality, nsfw, logo",
        # 512, 512, "13204175718326964000"
    ],
    # 2-Lyriel
    [
        # "lyriel_v16.safetensors", 
        # "mm_sd_v15.ckpt", 
        "A forbidden castle high up in the mountains, pixel art, intricate details2, hdr, intricate details, hyperdetailed5, natural skin texture, hyperrealism, soft light, sharp, game art, key visual, surreal",
        "3d, cartoon, anime, sketches, worst quality, low quality, normal quality, lowres, normal quality, monochrome, grayscale, skin spots, acnes, skin blemishes, bad anatomy, girl, loli, young, large breasts, red eyes, muscular",
        # 512, 512, "6681501646976930000"
    ],
    # 3-RCNZ
    [
        # "rcnzCartoon3d_v10.safetensors", 
        # "mm_sd_v14.ckpt", 
        "Jane Eyre with headphones, natural skin texture,4mm,k textures, soft cinematic light, adobe lightroom, photolab, hdr, intricate, elegant, highly detailed, sharp focus, cinematic look, soothing tones, insane details, intricate details, hyperdetailed, low contrast, soft cinematic light, dim colors, exposure blend, hdr, faded",
        "deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation",
        # 512, 512, "2416282124261060"
    ],
    # 4-MajicMix
    [
        # "majicmixRealistic_v5Preview.safetensors", 
        # "mm_sd_v14.ckpt", 
        "1girl, offshoulder, light smile, shiny skin best quality, masterpiece, photorealistic",
        "bad hand, worst quality, low quality, normal quality, lowres, bad anatomy, bad hands, watermark, moles",
        # 512, 512, "7132772652786303"
    ],
    # 5-RealisticVision
    [
        # "realisticVisionV20_v20.safetensors", 
        # "mm_sd_v15.ckpt", 
        "photo of coastline, rocks, storm weather, wind, waves, lightning, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3",
        "blur, haze, deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation",
        # 512, 512, "1490157606650685400"
    ]
]


def ui():
    with gr.Blocks(css=css) as demo:
        gr.Markdown(
            """
            # AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning
            Yuwei Guo, Ceyuan Yang*, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, Bo Dai (*Corresponding Author)<br>
            [Arxiv Report](https://arxiv.org/abs/2307.04725) | [Project Page](https://animatediff.github.io/) | [Github](https://github.com/guoyww/animatediff/)
            """
        )
        gr.Markdown(
            """
            ### Quick Start
            1. Select desired `Base DreamBooth Model`.
            2. Select `Motion Module` from `mm_sd_v14.ckpt` and `mm_sd_v15.ckpt`. We recommend trying both of them for the best results.
            3. Provide `Prompt` and `Negative Prompt` for each model. You are encouraged to refer to each model's webpage on CivitAI to learn how to write prompts for them. Below are the DreamBooth models in this demo. Click to visit their homepage.
                - [`toonyou_beta3.safetensors`](https://civitai.com/models/30240?modelVersionId=78775)
                - [`lyriel_v16.safetensors`](https://civitai.com/models/22922/lyriel)
                - [`rcnzCartoon3d_v10.safetensors`](https://civitai.com/models/66347?modelVersionId=71009)
                - [`majicmixRealistic_v5Preview.safetensors`](https://civitai.com/models/43331?modelVersionId=79068)
                - [`realisticVisionV20_v20.safetensors`](https://civitai.com/models/4201?modelVersionId=29460)
            4. Click `Generate`, wait for ~1 min, and enjoy.
            """
        )
        with gr.Row():
            with gr.Column():
                # id_embed_dropdown     = gr.Dropdown( label="Base DreamBooth Model", choices=c.base_model_list,    value=c.base_model_list[0],    interactive=True )
                # motion_module_dropdown  = gr.Dropdown( label="Motion Module",  choices=c.motion_module_list, value=c.motion_module_list[0], interactive=True )

                # id_embed_dropdown.change(fn=c.update_base_model,       inputs=[id_embed_dropdown],    outputs=[id_embed_dropdown])
                # motion_module_dropdown.change(fn=c.update_motion_module, inputs=[motion_module_dropdown], outputs=[motion_module_dropdown])
                id_embed_dropdown = gr.Dropdown( label="ID Embedding", choices=c.id_embed_list,    value=c.id_embed_list[0],    interactive=True )
                id_embed_dropdown.change(fn=c.update_id_embed,       inputs=[id_embed_dropdown],    outputs=[id_embed_dropdown])

                prompt_textbox          = gr.Textbox( label="Prompt", info="a photo of <V*> man/woman ",          lines=3, value="in superman costume in the outer space, stars in the background" )
                negative_prompt_textbox = gr.Textbox( label="Negative Prompt", lines=3, value="(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, UnrealisticDream")

                # with gr.Accordion("Advance", open=False):
                #     with gr.Row():
                #         width_slider  = gr.Slider(  label="Width",  value=512, minimum=256, maximum=1024, step=64 )
                #         height_slider = gr.Slider(  label="Height", value=512, minimum=256, maximum=1024, step=64 )
                #     with gr.Row():
                #         seed_textbox = gr.Textbox( label="Seed",  value=-1)
                #         seed_button  = gr.Button(value="\U0001F3B2", elem_classes="toolbutton")
                #         seed_button.click(fn=lambda: gr.Textbox.update(value=random.randint(1, 1e16)), inputs=[], outputs=[seed_textbox])

                generate_button = gr.Button( value="Generate", variant='primary' )

            with gr.Column():
                result_video = gr.Video( label="Generated Animation", interactive=False )
                json_config  = gr.Json( label="Config", value=None )

            inputs  = [prompt_textbox, negative_prompt_textbox, id_embed_dropdown]
            outputs = [result_video, json_config]
            
            generate_button.click( fn=c.run_once, inputs=inputs, outputs=outputs )
                
        # gr.Examples( fn=c.run_once, examples=examples, inputs=inputs, outputs=outputs, cache_examples=True )
        
    return demo


if __name__ == "__main__":
    demo = ui()
    demo.queue(max_size=20)
    demo.launch()