import inspect
import importlib
from typing import Any, Callable, Dict, List, Optional, Union
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import utils
from accelerate import Accelerator
from diffusers import FluxKontextPipeline


def retrieve_latents(
    encoder_output: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    sample_mode: str = "sample",
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class AttDistImgPipeline(FluxKontextPipeline):

    def freeze(self):
        self.transformer.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.vae.requires_grad_(False)

    @torch.no_grad()
    def image2latent(self, image):
        dtype = next(self.vae.parameters()).dtype
        device = self._execution_device
        image = image.to(device=device, dtype=dtype) * 2.0 - 1.0
        latent = retrieve_latents(self.vae.encode(image), sample_mode="argmax")
        latent = (
            latent - self.vae.config.shift_factor
        ) * self.vae.config.scaling_factor
        return latent

    @torch.no_grad()
    def latent2image(self, latent, height, width):
        dtype = next(self.vae.parameters()).dtype
        device = self._execution_device
        latent = latent.to(device=device, dtype=dtype)
        latents = self._unpack_latents(latent, height, width, self.vae_scale_factor)
        latents = (
            latents / self.vae.config.scaling_factor
        ) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        return (image * 0.5 + 0.5).clamp(0, 1)

    def sample(
        self,
        content_image=None,
        style_image=None,
        controller=None,
        loss_fn=None,
        start_time=9999,
        style_beta=0.7,
        core_type="core_ot_attn",
        lr=0.05,
        iters=2,
        mixed_precision="no",
        enable_gradient_checkpoint=False,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
    ):
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        device = self._execution_device
        accelerator = Accelerator(
            mixed_precision=mixed_precision, gradient_accumulation_steps=1
        )
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        self.transformer.to(accelerator.device, dtype=weight_dtype)
        self.vae.to(accelerator.device, dtype=weight_dtype)
        self.text_encoder.to(accelerator.device, dtype=weight_dtype)
        self.text_encoder_2.to(accelerator.device, dtype=weight_dtype)
        self.transformer = accelerator.prepare(self.transformer)
        if enable_gradient_checkpoint:
            self.transformer.enable_gradient_checkpointing()

        (null_embeds, null_pooled_embeds, null_text_ids) = self.encode_prompt(
            prompt="",
            prompt_2=prompt_2,
        )
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        # Prepare latent variables (Kontext returns 4-tuple)
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, _, latent_image_ids, _ = self.prepare_latents(
            None,  # no image for initial noise — just random latent
            num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            null_embeds.dtype,
            device,
            generator,
            latents,
        )

        # --- Content image: Kontext-style concat ---
        print(f"content_image shape: {content_image.shape}")
        content_latent = self.image2latent(content_image)
        cnt_h, cnt_w = content_latent.shape[2], content_latent.shape[3]
        content_latent_packed = self._pack_latents(
            content_latent, 1, num_channels_latents, cnt_h, cnt_w
        )
        # image_ids: same structure as latent_ids but first dim = 1
        content_image_ids = self._prepare_latent_image_ids(
            1, cnt_h // 2, cnt_w // 2, device, null_embeds.dtype
        )
        content_image_ids[..., 0] = 1

        # Combined img_ids for denoising forward (latent_ids + content_image_ids)
        combined_img_ids = torch.cat([latent_image_ids, content_image_ids], dim=0)

        # --- Style image ---
        print(f"style_image shape: {style_image.shape}")
        style_latent = self.image2latent(style_image)
        print(f"style_latent shape: {style_latent.shape}")
        style_latent = self._pack_latents(
            style_latent, 1, num_channels_latents,
            style_latent.shape[2], style_latent.shape[3]
        )

        # null image ids for style forward (positional ids matching style latent)
        sty_h_, sty_w_ = style_image.shape[2], style_image.shape[3]
        sty_lat_h = 2 * (sty_h_ // (self.vae_scale_factor * 2))
        sty_lat_w = 2 * (sty_w_ // (self.vae_scale_factor * 2))
        null_image_id = self._prepare_latent_image_ids(
            1, sty_lat_h // 2, sty_lat_w // 2, device, null_embeds.dtype
        )

        # Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps_fn(
            self.scheduler,
            num_inference_steps,
            device,
            None,
            sigmas,
            mu=mu,
        )

        timesteps = self.scheduler.timesteps
        print(f"timesteps: {timesteps}")
        self._num_timesteps = len(timesteps)

        cache = utils.DataCache()

        utils.register_attn_control(
            self.transformer.transformer_blocks,
            controller=controller,
            cache=cache,
        )
        utils.register_attn_control(
            self.transformer.single_transformer_blocks,
            controller=controller,
            cache=cache,
        )

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full(
                [1], guidance_scale, device=device, dtype=torch.float32
            )
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        null_guidance = torch.full(
            [1], 1, device=device, dtype=torch.float32
        )

        pbar = tqdm(timesteps, desc="Sample")
        for i, t in enumerate(pbar):
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            core_module = importlib.import_module(f"cores.{core_type}")
            importlib.reload(core_module)
            importlib.reload(utils)
            latents = core_module.step(
                i=i,
                t=t,
                timestep=timestep,
                latents=latents,
                guidance=guidance,
                pooled_prompt_embeds=pooled_prompt_embeds,
                prompt_embeds=prompt_embeds,
                text_ids=text_ids,
                latent_image_ids=latent_image_ids,
                num_inference_steps=num_inference_steps,
                timesteps=timesteps,
                style_latent=style_latent,
                scheduler=self.scheduler,
                cache=cache,
                controller=controller,
                transformer=self.transformer,
                null_guidance=null_guidance,
                null_pooled_embeds=null_pooled_embeds,
                null_embeds=null_embeds,
                null_text_ids=null_text_ids,
                null_image_id=null_image_id,
                lr=lr,
                accelerator=accelerator,
                iters=iters,
                loss_fn=loss_fn,
                pbar=pbar,
                start_time=start_time,
                style_beta=style_beta,
                # Kontext-specific: content image latent for concat
                content_latent_packed=content_latent_packed,
                combined_img_ids=combined_img_ids,
            )

        torch.cuda.empty_cache()
        latents = latents.detach()

        # Global color transfer
        import utils as _utils
        style_latent_final = self.image2latent(style_image)
        style_latent_final = self._pack_latents(
            style_latent_final, 1, self.transformer.config.in_channels // 4,
            style_latent_final.shape[2], style_latent_final.shape[3]
        )
        latents = _utils.adain(latents, style_latent_final)

        return self.latent2image(latents, height, width)


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def retrieve_timesteps_fn(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed."
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps
