
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, matplotlib, numbers, numpy, pandas, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchvision, types, typing, uuid, warnings
import operator as op
from dataclasses import dataclass
import numpy as np
from torch import Tensor
__version__ = '1.0.0'
xrange = range
wraps = functools.wraps


import torch


import torch.nn as nn


from torchvision.io import write_video


import torchvision.transforms as T


import numpy as np


from typing import Type


import math


from torchvision.io import read_video


import random


class Preprocess(nn.Module):

    def __init__(self, device, opt, hf_key=None):
        super().__init__()
        self.device = device
        self.sd_version = opt.sd_version
        self.use_depth = False
        None
        if hf_key is not None:
            None
            model_key = hf_key
        elif self.sd_version == '2.1':
            model_key = 'stabilityai/stable-diffusion-2-1-base'
        elif self.sd_version == '2.0':
            model_key = 'stabilityai/stable-diffusion-2-base'
        elif self.sd_version == '1.5' or self.sd_version == 'ControlNet':
            model_key = 'runwayml/stable-diffusion-v1-5'
        elif self.sd_version == 'depth':
            model_key = 'stabilityai/stable-diffusion-2-depth'
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')
        self.model_key = model_key
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder='vae', revision='fp16', torch_dtype=torch.float16)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder='tokenizer')
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder='text_encoder', revision='fp16', torch_dtype=torch.float16)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder='unet', revision='fp16', torch_dtype=torch.float16)
        self.paths, self.frames, self.latents = self.get_data(opt.data_path, opt.n_frames)
        if self.sd_version == 'ControlNet':
            controlnet = ControlNetModel.from_pretrained('lllyasviel/sd-controlnet-canny', torch_dtype=torch.float16)
            control_pipe = StableDiffusionControlNetPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', controlnet=controlnet, torch_dtype=torch.float16)
            self.unet = control_pipe.unet
            self.controlnet = control_pipe.controlnet
            self.canny_cond = self.get_canny_cond()
        elif self.sd_version == 'depth':
            self.depth_maps = self.prepare_depth_maps()
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder='scheduler')
        None

    @torch.no_grad()
    def prepare_depth_maps(self, model_type='DPT_Large', device='cuda'):
        depth_maps = []
        midas = torch.hub.load('intel-isl/MiDaS', model_type)
        midas
        midas.eval()
        midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
        if model_type == 'DPT_Large' or model_type == 'DPT_Hybrid':
            transform = midas_transforms.dpt_transform
        else:
            transform = midas_transforms.small_transform
        for i in range(len(self.paths)):
            img = cv2.imread(self.paths[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            latent_h = img.shape[0] // 8
            latent_w = img.shape[1] // 8
            input_batch = transform(img)
            prediction = midas(input_batch)
            depth_map = torch.nn.functional.interpolate(prediction.unsqueeze(1), size=(latent_h, latent_w), mode='bicubic', align_corners=False)
            depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
            depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
            depth_map = 2.0 * (depth_map - depth_min) / (depth_max - depth_min) - 1.0
            depth_maps.append(depth_map)
        return torch.cat(depth_maps).to(self.device)

    @torch.no_grad()
    def get_canny_cond(self):
        canny_cond = []
        for image in self.frames.cpu().permute(0, 2, 3, 1):
            image = np.uint8(np.array(255 * image))
            low_threshold = 100
            high_threshold = 200
            image = cv2.Canny(image, low_threshold, high_threshold)
            image = image[:, :, None]
            image = np.concatenate([image, image, image], axis=2)
            image = torch.from_numpy(image.astype(np.float32) / 255.0)
            canny_cond.append(image)
        canny_cond = torch.stack(canny_cond).permute(0, 3, 1, 2).to(self.device)
        return canny_cond

    def controlnet_pred(self, latent_model_input, t, text_embed_input, controlnet_cond):
        down_block_res_samples, mid_block_res_sample = self.controlnet(latent_model_input, t, encoder_hidden_states=text_embed_input, controlnet_cond=controlnet_cond, conditioning_scale=1, return_dict=False)
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embed_input, cross_attention_kwargs={}, down_block_additional_residuals=down_block_res_samples, mid_block_additional_residual=mid_block_res_sample, return_dict=False)[0]
        return noise_pred

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, device='cuda'):
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids)[0]
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
        uncond_embeddings = self.text_encoder(uncond_input.input_ids)[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    @torch.no_grad()
    def decode_latents(self, latents):
        decoded = []
        batch_size = 8
        for b in range(0, latents.shape[0], batch_size):
            latents_batch = 1 / 0.18215 * latents[b:b + batch_size]
            imgs = self.vae.decode(latents_batch).sample
            imgs = (imgs / 2 + 0.5).clamp(0, 1)
            decoded.append(imgs)
        return torch.cat(decoded)

    @torch.no_grad()
    def encode_imgs(self, imgs, batch_size=10, deterministic=True):
        imgs = 2 * imgs - 1
        latents = []
        for i in range(0, len(imgs), batch_size):
            posterior = self.vae.encode(imgs[i:i + batch_size]).latent_dist
            latent = posterior.mean if deterministic else posterior.sample()
            latents.append(latent * 0.18215)
        latents = torch.cat(latents)
        return latents

    def get_data(self, frames_path, n_frames):
        paths = [(f'{frames_path}/%05d.png' % i) for i in range(n_frames)]
        if not os.path.exists(paths[0]):
            paths = [(f'{frames_path}/%05d.jpg' % i) for i in range(n_frames)]
        self.paths = paths
        frames = [Image.open(path).convert('RGB') for path in paths]
        if frames[0].size[0] == frames[0].size[1]:
            frames = [frame.resize((512, 512), resample=Image.Resampling.LANCZOS) for frame in frames]
        frames = torch.stack([T.ToTensor()(frame) for frame in frames]).to(torch.float16)
        latents = self.encode_imgs(frames, deterministic=True).to(torch.float16)
        return paths, frames, latents

    @torch.no_grad()
    def ddim_inversion(self, cond, latent_frames, save_path, batch_size, save_latents=True, timesteps_to_save=None):
        timesteps = reversed(self.scheduler.timesteps)
        timesteps_to_save = timesteps_to_save if timesteps_to_save is not None else timesteps
        for i, t in enumerate(tqdm(timesteps)):
            for b in range(0, latent_frames.shape[0], batch_size):
                x_batch = latent_frames[b:b + batch_size]
                model_input = x_batch
                cond_batch = cond.repeat(x_batch.shape[0], 1, 1)
                if self.sd_version == 'depth':
                    depth_maps = torch.cat([self.depth_maps[b:b + batch_size]])
                    model_input = torch.cat([x_batch, depth_maps], dim=1)
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = self.scheduler.alphas_cumprod[timesteps[i - 1]] if i > 0 else self.scheduler.final_alpha_cumprod
                mu = alpha_prod_t ** 0.5
                mu_prev = alpha_prod_t_prev ** 0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5
                eps = self.unet(model_input, t, encoder_hidden_states=cond_batch).sample if self.sd_version != 'ControlNet' else self.controlnet_pred(x_batch, t, cond_batch, torch.cat([self.canny_cond[b:b + batch_size]]))
                pred_x0 = (x_batch - sigma_prev * eps) / mu_prev
                latent_frames[b:b + batch_size] = mu * pred_x0 + sigma * eps
            if save_latents and t in timesteps_to_save:
                torch.save(latent_frames, os.path.join(save_path, 'latents', f'noisy_latents_{t}.pt'))
        torch.save(latent_frames, os.path.join(save_path, 'latents', f'noisy_latents_{t}.pt'))
        return latent_frames

    @torch.no_grad()
    def ddim_sample(self, x, cond, batch_size):
        timesteps = self.scheduler.timesteps
        for i, t in enumerate(tqdm(timesteps)):
            for b in range(0, x.shape[0], batch_size):
                x_batch = x[b:b + batch_size]
                model_input = x_batch
                cond_batch = cond.repeat(x_batch.shape[0], 1, 1)
                if self.sd_version == 'depth':
                    depth_maps = torch.cat([self.depth_maps[b:b + batch_size]])
                    model_input = torch.cat([x_batch, depth_maps], dim=1)
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = self.scheduler.alphas_cumprod[timesteps[i + 1]] if i < len(timesteps) - 1 else self.scheduler.final_alpha_cumprod
                mu = alpha_prod_t ** 0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                mu_prev = alpha_prod_t_prev ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5
                eps = self.unet(model_input, t, encoder_hidden_states=cond_batch).sample if self.sd_version != 'ControlNet' else self.controlnet_pred(x_batch, t, cond_batch, torch.cat([self.canny_cond[b:b + batch_size]]))
                pred_x0 = (x_batch - sigma * eps) / mu
                x[b:b + batch_size] = mu_prev * pred_x0 + sigma_prev * eps
        return x

    @torch.no_grad()
    def extract_latents(self, num_steps, save_path, batch_size, timesteps_to_save, inversion_prompt=''):
        self.scheduler.set_timesteps(num_steps)
        cond = self.get_text_embeds(inversion_prompt, '')[1].unsqueeze(0)
        latent_frames = self.latents
        inverted_x = self.ddim_inversion(cond, latent_frames, save_path, batch_size=batch_size, save_latents=True, timesteps_to_save=timesteps_to_save)
        latent_reconstruction = self.ddim_sample(inverted_x, cond, batch_size=batch_size)
        rgb_reconstruction = self.decode_latents(latent_reconstruction)
        return rgb_reconstruction


VAE_BATCH_SIZE = 10


def load_source_latents_t(t, latents_path):
    latents_t_path = os.path.join(latents_path, f'noisy_latents_{t}.pt')
    assert os.path.exists(latents_t_path), f'Missing latents at t {t} path {latents_t_path}'
    latents = torch.load(latents_t_path)
    return latents


def isinstance_str(x: 'object', cls_name: 'str'):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.
    
    Useful for patching!
    """
    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    return False


def register_batch_idx(diffusion_model, batch_idx):
    for _, module in diffusion_model.named_modules():
        if isinstance_str(module, 'BasicTransformerBlock'):
            setattr(module, 'batch_idx', batch_idx)


def register_extended_attention(model):

    def sa_forward(self):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, encoder_hidden_states=None, attention_mask=None):
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            n_frames = batch_size // 3
            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else x
            q = self.to_q(x)
            k = self.to_k(encoder_hidden_states)
            v = self.to_v(encoder_hidden_states)
            k_source = k[:n_frames]
            k_uncond = k[n_frames:2 * n_frames].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
            k_cond = k[2 * n_frames:].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
            v_source = v[:n_frames]
            v_uncond = v[n_frames:2 * n_frames].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
            v_cond = v[2 * n_frames:].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
            q_source = self.head_to_batch_dim(q[:n_frames])
            q_uncond = self.head_to_batch_dim(q[n_frames:2 * n_frames])
            q_cond = self.head_to_batch_dim(q[2 * n_frames:])
            k_source = self.head_to_batch_dim(k_source)
            k_uncond = self.head_to_batch_dim(k_uncond)
            k_cond = self.head_to_batch_dim(k_cond)
            v_source = self.head_to_batch_dim(v_source)
            v_uncond = self.head_to_batch_dim(v_uncond)
            v_cond = self.head_to_batch_dim(v_cond)
            out_source = []
            out_uncond = []
            out_cond = []
            q_src = q_source.view(n_frames, h, sequence_length, dim // h)
            k_src = k_source.view(n_frames, h, sequence_length, dim // h)
            v_src = v_source.view(n_frames, h, sequence_length, dim // h)
            q_uncond = q_uncond.view(n_frames, h, sequence_length, dim // h)
            k_uncond = k_uncond.view(n_frames, h, sequence_length * n_frames, dim // h)
            v_uncond = v_uncond.view(n_frames, h, sequence_length * n_frames, dim // h)
            q_cond = q_cond.view(n_frames, h, sequence_length, dim // h)
            k_cond = k_cond.view(n_frames, h, sequence_length * n_frames, dim // h)
            v_cond = v_cond.view(n_frames, h, sequence_length * n_frames, dim // h)
            for j in range(h):
                sim_source_b = torch.bmm(q_src[:, j], k_src[:, j].transpose(-1, -2)) * self.scale
                sim_uncond_b = torch.bmm(q_uncond[:, j], k_uncond[:, j].transpose(-1, -2)) * self.scale
                sim_cond = torch.bmm(q_cond[:, j], k_cond[:, j].transpose(-1, -2)) * self.scale
                out_source.append(torch.bmm(sim_source_b.softmax(dim=-1), v_src[:, j]))
                out_uncond.append(torch.bmm(sim_uncond_b.softmax(dim=-1), v_uncond[:, j]))
                out_cond.append(torch.bmm(sim_cond.softmax(dim=-1), v_cond[:, j]))
            out_source = torch.cat(out_source, dim=0).view(h, n_frames, sequence_length, dim // h).permute(1, 0, 2, 3).reshape(h * n_frames, sequence_length, -1)
            out_uncond = torch.cat(out_uncond, dim=0).view(h, n_frames, sequence_length, dim // h).permute(1, 0, 2, 3).reshape(h * n_frames, sequence_length, -1)
            out_cond = torch.cat(out_cond, dim=0).view(h, n_frames, sequence_length, dim // h).permute(1, 0, 2, 3).reshape(h * n_frames, sequence_length, -1)
            out = torch.cat([out_source, out_uncond, out_cond], dim=0)
            out = self.batch_to_head_dim(out)
            return to_out(out)
        return forward
    for _, module in model.unet.named_modules():
        if isinstance_str(module, 'BasicTransformerBlock'):
            module.attn1.forward = sa_forward(module.attn1)
    res_dict = {(1): [1, 2], (2): [0, 1, 2], (3): [0, 1, 2]}
    for res in res_dict:
        for block in res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            module.forward = sa_forward(module)


def register_pivotal(diffusion_model, is_pivotal):
    for _, module in diffusion_model.named_modules():
        if isinstance_str(module, 'BasicTransformerBlock'):
            setattr(module, 'pivotal_pass', is_pivotal)


def register_time(model, t):
    conv_module = model.unet.up_blocks[1].resnets[1]
    setattr(conv_module, 't', t)
    down_res_dict = {(0): [0, 1], (1): [0, 1], (2): [0, 1]}
    up_res_dict = {(1): [0, 1, 2], (2): [0, 1, 2], (3): [0, 1, 2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn2
            setattr(module, 't', t)
    for res in down_res_dict:
        for block in down_res_dict[res]:
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn2
            setattr(module, 't', t)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn1
    setattr(module, 't', t)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn2
    setattr(module, 't', t)


def save_video(raw_frames, save_path, fps=10):
    video_codec = 'libx264'
    video_options = {'crf': '18', 'preset': 'slow'}
    frames = (raw_frames * 255).cpu().permute(0, 2, 3, 1)
    write_video(save_path, frames, fps=fps, video_codec=video_codec, options=video_options)


def batch_cosine_sim(x, y):
    if type(x) is list:
        x = torch.cat(x, dim=0)
    if type(y) is list:
        y = torch.cat(y, dim=0)
    x = x / x.norm(dim=-1, keepdim=True)
    y = y / y.norm(dim=-1, keepdim=True)
    similarity = x @ y.T
    return similarity


def make_tokenflow_attention_block(block_class: 'Type[torch.nn.Module]') ->Type[torch.nn.Module]:


    class TokenFlowBlock(block_class):

        def forward(self, hidden_states, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, timestep=None, cross_attention_kwargs=None, class_labels=None) ->torch.Tensor:
            batch_size, sequence_length, dim = hidden_states.shape
            n_frames = batch_size // 3
            mid_idx = n_frames // 2
            hidden_states = hidden_states.view(3, n_frames, sequence_length, dim)
            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype)
            else:
                norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states.view(3, n_frames, sequence_length, dim)
            if self.pivotal_pass:
                self.pivot_hidden_states = norm_hidden_states
            else:
                idx1 = []
                idx2 = []
                batch_idxs = [self.batch_idx]
                if self.batch_idx > 0:
                    batch_idxs.append(self.batch_idx - 1)
                sim = batch_cosine_sim(norm_hidden_states[0].reshape(-1, dim), self.pivot_hidden_states[0][batch_idxs].reshape(-1, dim))
                if len(batch_idxs) == 2:
                    sim1, sim2 = sim.chunk(2, dim=1)
                    idx1.append(sim1.argmax(dim=-1))
                    idx2.append(sim2.argmax(dim=-1))
                else:
                    idx1.append(sim.argmax(dim=-1))
                idx1 = torch.stack(idx1 * 3, dim=0)
                idx1 = idx1.squeeze(1)
                if len(batch_idxs) == 2:
                    idx2 = torch.stack(idx2 * 3, dim=0)
                    idx2 = idx2.squeeze(1)
            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
            if self.pivotal_pass:
                self.attn_output = self.attn1(norm_hidden_states.view(batch_size, sequence_length, dim), encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None, **cross_attention_kwargs)
                self.kf_attn_output = self.attn_output
            else:
                batch_kf_size, _, _ = self.kf_attn_output.shape
                self.attn_output = self.kf_attn_output.view(3, batch_kf_size // 3, sequence_length, dim)[:, batch_idxs]
            if self.use_ada_layer_norm_zero:
                self.attn_output = gate_msa.unsqueeze(1) * self.attn_output
            if not self.pivotal_pass:
                if len(batch_idxs) == 2:
                    attn_1, attn_2 = self.attn_output[:, 0], self.attn_output[:, 1]
                    attn_output1 = attn_1.gather(dim=1, index=idx1.unsqueeze(-1).repeat(1, 1, dim))
                    attn_output2 = attn_2.gather(dim=1, index=idx2.unsqueeze(-1).repeat(1, 1, dim))
                    s = torch.arange(0, n_frames) + batch_idxs[0] * n_frames
                    p1 = batch_idxs[0] * n_frames + n_frames // 2
                    p2 = batch_idxs[1] * n_frames + n_frames // 2
                    d1 = torch.abs(s - p1)
                    d2 = torch.abs(s - p2)
                    w1 = d2 / (d1 + d2)
                    w1 = torch.sigmoid(w1)
                    w1 = w1.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(3, 1, sequence_length, dim)
                    attn_output1 = attn_output1.view(3, n_frames, sequence_length, dim)
                    attn_output2 = attn_output2.view(3, n_frames, sequence_length, dim)
                    attn_output = w1 * attn_output1 + (1 - w1) * attn_output2
                else:
                    attn_output = self.attn_output[:, 0].gather(dim=1, index=idx1.unsqueeze(-1).repeat(1, 1, dim))
                attn_output = attn_output.reshape(batch_size, sequence_length, dim)
            else:
                attn_output = self.attn_output
            hidden_states = hidden_states.reshape(batch_size, sequence_length, dim)
            hidden_states = attn_output + hidden_states
            if self.attn2 is not None:
                norm_hidden_states = self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                attn_output = self.attn2(norm_hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=encoder_attention_mask, **cross_attention_kwargs)
                hidden_states = attn_output + hidden_states
            norm_hidden_states = self.norm3(hidden_states)
            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            ff_output = self.ff(norm_hidden_states)
            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output
            hidden_states = ff_output + hidden_states
            return hidden_states
    return TokenFlowBlock


def set_tokenflow(model: 'torch.nn.Module'):
    """
    Sets the tokenflow attention blocks in a model.
    """
    for _, module in model.named_modules():
        if isinstance_str(module, 'BasicTransformerBlock'):
            make_tokenflow_block_fn = make_tokenflow_attention_block
            module.__class__ = make_tokenflow_block_fn(module.__class__)
            if not hasattr(module, 'use_ada_layer_norm_zero'):
                module.use_ada_layer_norm = False
                module.use_ada_layer_norm_zero = False
    return model


class TokenFlow(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config['device']
        sd_version = config['sd_version']
        if sd_version == '2.1':
            model_key = 'stabilityai/stable-diffusion-2-1-base'
        elif sd_version == '2.0':
            model_key = 'stabilityai/stable-diffusion-2-base'
        elif sd_version == '1.5':
            model_key = 'runwayml/stable-diffusion-v1-5'
        else:
            raise ValueError(f'Stable-diffusion version {sd_version} not supported.')
        None
        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=torch.float16)
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder='scheduler')
        self.scheduler.set_timesteps(config['n_timesteps'], device=self.device)
        self.scheduler.timesteps = self.scheduler.timesteps[int(1 - config['start'] * len(self.scheduler.timesteps)):]
        None
        self.latents_path = self.get_latents_path()
        self.keyframes_path = [os.path.join(config['data_path'], '%05d.jpg' % idx) for idx in range(self.config['n_frames'])]
        if not os.path.exists(self.keyframes_path[0]):
            self.keyframes_path = [os.path.join(config['data_path'], '%05d.png' % idx) for idx in range(self.config['n_frames'])]
        self.frames, self.latents, self.eps = self.get_data()
        self.text_embeds = self.get_text_embeds(config['prompt'], config['negative_prompt'])
        pnp_inversion_prompt = self.get_pnp_inversion_prompt()
        self.pnp_guidance_embeds = self.get_text_embeds(pnp_inversion_prompt, pnp_inversion_prompt).chunk(2)[0]

    def get_pnp_inversion_prompt(self):
        inv_prompts_path = os.path.join(str(Path(self.latents_path).parent), 'inversion_prompt.txt')
        with open(inv_prompts_path, 'r') as f:
            inv_prompt = f.read()
        return inv_prompt

    def get_latents_path(self):
        latents_path = os.path.join(config['latents_path'], f"sd_{config['sd_version']}", Path(config['data_path']).stem)
        latents_path = [x for x in glob.glob(f'{latents_path}/*/*') if not x.startswith('.')]
        n_frames = [int([x for x in latents_path[i].split('/') if 'nframes' in x][0].split('_')[1]) for i in range(len(latents_path))]
        latents_path = latents_path[np.argmax(n_frames)]
        self.config['n_frames'] = min(max(n_frames), config['n_frames'])
        if self.config['n_frames'] % self.config['batch_size'] != 0:
            self.config['n_frames'] = self.config['n_frames'] - self.config['n_frames'] % self.config['batch_size']
        None
        return os.path.join(latents_path, 'latents')

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, batch_size=1):
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids)[0]
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
        uncond_embeddings = self.text_encoder(uncond_input.input_ids)[0]
        text_embeddings = torch.cat([uncond_embeddings] * batch_size + [text_embeddings] * batch_size)
        return text_embeddings

    @torch.no_grad()
    def encode_imgs(self, imgs, batch_size=VAE_BATCH_SIZE, deterministic=False):
        imgs = 2 * imgs - 1
        latents = []
        for i in range(0, len(imgs), batch_size):
            posterior = self.vae.encode(imgs[i:i + batch_size]).latent_dist
            latent = posterior.mean if deterministic else posterior.sample()
            latents.append(latent * 0.18215)
        latents = torch.cat(latents)
        return latents

    @torch.no_grad()
    def decode_latents(self, latents, batch_size=VAE_BATCH_SIZE):
        latents = 1 / 0.18215 * latents
        imgs = []
        for i in range(0, len(latents), batch_size):
            imgs.append(self.vae.decode(latents[i:i + batch_size]).sample)
        imgs = torch.cat(imgs)
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    def get_data(self):
        frames = [Image.open(self.keyframes_path[idx]).convert('RGB') for idx in range(self.config['n_frames'])]
        if frames[0].size[0] == frames[0].size[1]:
            frames = [frame.resize((512, 512), resample=Image.Resampling.LANCZOS) for frame in frames]
        frames = torch.stack([T.ToTensor()(frame) for frame in frames]).to(torch.float16)
        latents = self.encode_imgs(frames, deterministic=True).to(torch.float16)
        eps = self.get_ddim_eps(latents, range(self.config['n_frames'])).to(torch.float16)
        return frames, latents, eps

    def get_ddim_eps(self, latent, indices):
        noisest = max([int(x.split('_')[-1].split('.')[0]) for x in glob.glob(os.path.join(self.latents_path, f'noisy_latents_*.pt'))])
        latents_path = os.path.join(self.latents_path, f'noisy_latents_{noisest}.pt')
        noisy_latent = torch.load(latents_path)[indices]
        alpha_prod_T = self.scheduler.alphas_cumprod[noisest]
        mu_T, sigma_T = alpha_prod_T ** 0.5, (1 - alpha_prod_T) ** 0.5
        eps = (noisy_latent - mu_T * latent) / sigma_T
        return eps

    @torch.no_grad()
    def denoise_step(self, x, t, indices):
        source_latents = load_source_latents_t(t, self.latents_path)[indices]
        latent_model_input = torch.cat([source_latents] + [x] * 2)
        register_time(self, t.item())
        text_embed_input = torch.cat([self.pnp_guidance_embeds.repeat(len(indices), 1, 1), torch.repeat_interleave(self.text_embeds, len(indices), dim=0)])
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embed_input)['sample']
        _, noise_pred_uncond, noise_pred_cond = noise_pred.chunk(3)
        noise_pred = noise_pred_uncond + self.config['guidance_scale'] * (noise_pred_cond - noise_pred_uncond)
        denoised_latent = self.scheduler.step(noise_pred, t, x)['prev_sample']
        return denoised_latent

    @torch.autocast(dtype=torch.float16, device_type='cuda')
    def batched_denoise_step(self, x, t, indices):
        batch_size = self.config['batch_size']
        denoised_latents = []
        pivotal_idx = torch.randint(batch_size, (len(x) // batch_size,)) + torch.arange(0, len(x), batch_size)
        register_pivotal(self, True)
        self.denoise_step(x[pivotal_idx], t, indices[pivotal_idx])
        register_pivotal(self, False)
        for i, b in enumerate(range(0, len(x), batch_size)):
            register_batch_idx(self, i)
            denoised_latents.append(self.denoise_step(x[b:b + batch_size], t, indices[b:b + batch_size]))
        denoised_latents = torch.cat(denoised_latents)
        return denoised_latents

    def init_method(self):
        register_extended_attention(self)
        set_tokenflow(self.unet)

    def edit_video(self):
        os.makedirs(f"{self.config['output_path']}/img_ode", exist_ok=True)
        self.init_method()
        noise = self.eps if config['use_ddim_noise'] else torch.randn_like(self.eps[[0]]).repeat(self.eps.shape[0])
        noisy_latents = self.scheduler.add_noise(self.latents, noise, self.scheduler.timesteps[0])
        edited_frames = self.sample_loop(noisy_latents, torch.arange(self.config['n_frames']))
        save_video(edited_frames, f"{self.config['output_path']}/tokenflow_SDEdit_fps_10.mp4")
        save_video(edited_frames, f"{self.config['output_path']}/tokenflow_SDEdit_fps_20.mp4", fps=20)
        save_video(edited_frames, f"{self.config['output_path']}/tokenflow_SDEdit_fps_30.mp4", fps=30)
        None

    def sample_loop(self, x, indices):
        os.makedirs(f"{self.config['output_path']}/img_ode", exist_ok=True)
        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc='Sampling')):
            x = self.batched_denoise_step(x, t, indices)
        decoded_latents = self.decode_latents(x)
        for i in range(len(decoded_latents)):
            T.ToPILImage()(decoded_latents[i]).save(f"{self.config['output_path']}/img_ode/%05d.png" % i)
        return decoded_latents

    def per_frame_sde(self):
        os.makedirs(f"{self.config['output_path']}/img_ode", exist_ok=True)
        noisy_latents = self.scheduler.add_noise(self.latents, self.eps, self.scheduler.timesteps[0])
        edited_frames = self.vanilla_sample_loop(noisy_latents, torch.arange(self.config['n_frames']))
        save_video(edited_frames, f"{self.config['output_path']}/vanilla_sde.mp4")
        save_video(edited_frames, f"{self.config['output_path']}/vanilla_sde_fps20.mp4", fps=20)
        save_video(edited_frames, f"{self.config['output_path']}/vanilla_sde_fps30.mp4", fps=30)
        None

    def vanilla_denoise(self, batch, t, text_embed_input):
        latent_model_input = torch.cat([batch] * 2)
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embed_input)['sample']
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.config['guidance_scale'] * (noise_pred_cond - noise_pred_uncond)
        batch = self.scheduler.step(noise_pred, t, batch)['prev_sample']
        return batch

    def batch_vanilla_denoise_step(self, x, t, text_embed_input):
        denoised = []
        for b in range(0, len(x), self.config['batch_size']):
            denoised.append(self.vanilla_denoise(x[b:b + self.config['batch_size']], t, text_embed_input))
        x = torch.cat(denoised)
        return x

    @torch.no_grad()
    def vanilla_sample_loop(self, x, indices):
        os.makedirs(f"{self.config['output_path']}/img_ode_vanilla_sde", exist_ok=True)
        text_embed_input = torch.cat([torch.repeat_interleave(self.text_embeds, config['batch_size'], dim=0)])
        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc='Sampling')):
            x = self.batch_vanilla_denoise_step(x, t, text_embed_input)
        decoded_latents = self.decode_latents(x)
        for i in range(len(decoded_latents)):
            T.ToPILImage()(decoded_latents[i]).save(f"{self.config['output_path']}/img_ode_vanilla_sde/%05d.png" % i)
        return decoded_latents

