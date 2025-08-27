import argparse
import glob
import os
import os.path as osp
from pathlib import Path
import re

import soundfile as sf
import torch
import torchvision
from huggingface_hub import snapshot_download
from moviepy.editor import AudioFileClip, VideoFileClip
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from foleycrafter.models.onset import torch_utils
from foleycrafter.models.time_detector.model import VideoOnsetNet
from foleycrafter.pipelines.auffusion_pipeline import Generator, denormalize_spectrogram
from foleycrafter.utils.util import build_foleycrafter, read_frames_with_moviepy


vision_transform_list = [
    torchvision.transforms.Resize((128, 128)),
    torchvision.transforms.CenterCrop((112, 112)),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]
video_transform = torchvision.transforms.Compose(vision_transform_list)


def args_parse():
    config = argparse.ArgumentParser()
    config.add_argument("--prompt", type=str, default="", help="prompt for audio generation")
    config.add_argument("--nprompt", type=str, default="", help="negative prompt for audio generation")
    config.add_argument("--seed", type=int, default=42, help="ramdom seed")
    config.add_argument("--semantic_scale", type=float, default=1.0, help="visual content scale")
    config.add_argument("--temporal_scale", type=float, default=0.2, help="temporal align scale")
    config.add_argument("--input", type=str, default="examples/sora", help="input video folder path")
    config.add_argument("--ckpt", type=str, default="checkpoints/", help="checkpoints folder path")
    # Default output directory changed back to 'output_analysis' per request
    config.add_argument("--save_dir", type=str, default="output_analysis/", help="generation result save path")
    config.add_argument(
        "--recursive",
        action="store_true",
        help="recursively search for video files inside the input directory (mp4, mov, mkv, webm)",
    )
    config.add_argument(
        "--pre_download_only",
        action="store_true",
        help="only download/load all model weights then exit (warm cache)",
    )
    config.add_argument(
        "--num_inference_steps", type=int, default=25, help="diffusion inference steps (quality vs speed tradeoff)"
    )
    config.add_argument(
        "--pretrain",
        type=str,
        default="auffusion/auffusion-full-no-adapter",
        help="audio generator pretrained checkpoint path",
    )
    config.add_argument("--device", type=str, default="cuda")
    config = config.parse_args()
    return config


def build_models(config):
    print("[Build] Preparing models...")
    # download ckpt (generator)
    pretrained_model_name_or_path = config.pretrain
    if not os.path.isdir(pretrained_model_name_or_path):
        print(f"[Build] Downloading generator weights: {pretrained_model_name_or_path}")
        pretrained_model_name_or_path = snapshot_download(pretrained_model_name_or_path)

    fc_ckpt = "ymzhang319/FoleyCrafter"
    if not os.path.isdir(osp.join(config.ckpt, "vocoder")):
        print("[Build] Downloading FoleyCrafter full checkpoint bundle (vocoder + adapters)...")
        fc_ckpt = snapshot_download(fc_ckpt, local_dir=config.ckpt)
    else:
        fc_ckpt = config.ckpt

    # ckpt path
    temporal_ckpt_path = osp.join(config.ckpt, "temporal_adapter.ckpt")

    # load vocoder
    vocoder_config_path = fc_ckpt
    print("[Build] Loading vocoder...")
    vocoder = Generator.from_pretrained(vocoder_config_path, subfolder="vocoder").to(config.device)

    # load time_detector
    time_detector_ckpt = osp.join(osp.join(config.ckpt, "timestamp_detector.pth.tar"))
    print("[Build] Loading time detector...")
    time_detector = VideoOnsetNet(False)
    time_detector, _ = torch_utils.load_model(time_detector_ckpt, time_detector, device=config.device, strict=True)

    # load adapters
    print("[Build] Loading diffusion pipeline (this can take a while the first time)...")
    pipe = build_foleycrafter().to(config.device)
    ckpt = torch.load(temporal_ckpt_path)

    # load temporal adapter
    if "state_dict" in ckpt.keys():
        ckpt = ckpt["state_dict"]
    load_gligen_ckpt = {}
    for key, value in ckpt.items():
        if key.startswith("module."):
            load_gligen_ckpt[key[len("module.") :]] = value
        else:
            load_gligen_ckpt[key] = value
    m, u = pipe.controlnet.load_state_dict(load_gligen_ckpt, strict=False)
    print(f"[Build] ControlNet loaded (missing={len(m)}, unexpected={len(u)})")

    # load semantic adapter
    pipe.load_ip_adapter(
        osp.join(config.ckpt, "semantic"), subfolder="", weight_name="semantic_adapter.bin", image_encoder_folder=None
    )
    ip_adapter_weight = config.semantic_scale
    pipe.set_ip_adapter_scale(ip_adapter_weight)
    print("[Build] Semantic (IP-Adapter) scale set to", ip_adapter_weight)
    print("[Build] All models ready.")

    return pipe, vocoder, time_detector


def run_inference(config, pipe, vocoder, time_detector):
    controlnet_conditioning_scale = config.temporal_scale
    os.makedirs(config.save_dir, exist_ok=True)
    # Accept both file and directory input
    raw_input_path = osp.abspath(config.input)
    if not osp.exists(raw_input_path):
        # Auto-fallback: if user passed a relative path that actually exists under /data
        candidate = osp.join('/data', config.input.lstrip('/'))
        if osp.exists(candidate):
            print(f"[Info] Input path not found at {raw_input_path}, using data fallback {candidate}")
            raw_input_path = candidate
    patterns = ["*.mp4", "*.MP4", "*.mov", "*.MOV", "*.mkv", "*.MKV", "*.webm", "*.WEBM"]
    if osp.isfile(raw_input_path):
        input_list = [raw_input_path]
    else:
        if not osp.isdir(raw_input_path):
            raise FileNotFoundError(
                f"Input path not found: {raw_input_path}. Provide a file or directory containing videos (checked also /data fallback)."
            )
        input_list = []
        if config.recursive:
            for root, _, _ in os.walk(raw_input_path):
                for p in patterns:
                    input_list.extend(glob.glob(osp.join(root, p)))
        else:
            for p in patterns:
                input_list.extend(glob.glob(osp.join(raw_input_path, p)))
        input_list = sorted(list({osp.abspath(p) for p in input_list}))
    if len(input_list) == 0:
        raise RuntimeError(
            f"No video files found in {raw_input_path} (recursive={config.recursive}). Supported extensions: {patterns}."
        )
    print(f"[Run] Found {len(input_list)} video(s) to process.")

    generator = torch.Generator(device=config.device)
    generator.manual_seed(config.seed)
    image_processor = CLIPImageProcessor()
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter", subfolder="models/image_encoder"
    ).to(config.device)
    input_list.sort()
    with torch.no_grad():
        for input_video in input_list:
            print(f" >>> Begin Inference: {input_video} <<< ")
            frames, duration = read_frames_with_moviepy(input_video, max_frame_nums=150)

            time_frames = torch.FloatTensor(frames).permute(0, 3, 1, 2)
            time_frames = video_transform(time_frames)
            time_frames = {"frames": time_frames.unsqueeze(0).permute(0, 2, 1, 3, 4)}
            preds = time_detector(time_frames)
            preds = torch.sigmoid(preds)

            # duration
            # import ipdb; ipdb.set_trace()
            time_condition = [
                -1 if preds[0][int(i / (1024 / 10 * duration) * 150)] < 0.5 else 1
                for i in range(int(1024 / 10 * duration))
            ]
            time_condition = time_condition + [-1] * (1024 - len(time_condition))
            # w -> b c h w
            time_condition = (
                torch.FloatTensor(time_condition)
                .unsqueeze(0)
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(1, 1, 256, 1)
                .to("cuda")
            )
            images = image_processor(images=frames, return_tensors="pt").to("cuda")
            image_embeddings = image_encoder(**images).image_embeds
            image_embeddings = torch.mean(image_embeddings, dim=0, keepdim=True).unsqueeze(0).unsqueeze(0)
            neg_image_embeddings = torch.zeros_like(image_embeddings)
            image_embeddings = torch.cat([neg_image_embeddings, image_embeddings], dim=1)

            name = Path(input_video).stem
            name = name.replace("+", " ")
            # Sanitize filename: remove illegal chars, leading dashes (ffmpeg treats leading '-' as option)
            safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).lstrip("-_")
            if not safe_name:
                safe_name = "out"

            sample = pipe(
                prompt=config.prompt,
                negative_prompt=config.nprompt,
                ip_adapter_image_embeds=image_embeddings,
                image=time_condition,
                # audio_length_in_s=10,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                num_inference_steps=config.num_inference_steps,
                height=256,
                width=1024,
                output_type="pt",
                generator=generator,
                # guidance_scale=0,
            )
            audio_img = sample.images[0]
            audio = denormalize_spectrogram(audio_img)
            audio = vocoder.inference(audio, lengths=160000)[0]
            audio_save_path = osp.join(config.save_dir, "audio")
            video_save_path = osp.join(config.save_dir, "video")
            os.makedirs(audio_save_path, exist_ok=True)
            os.makedirs(video_save_path, exist_ok=True)
            audio = audio[: int(duration * 16000)]

            save_path = osp.join(audio_save_path, f"{safe_name}.wav")
            sf.write(save_path, audio, 16000)

            # Re-open audio/video and compose final output with robust duration handling.
            # MoviePy sometimes requests audio frames a few ms past the declared duration due to rounding.
            # We therefore trim BOTH streams to the common minimum minus a tiny safety margin.
            final_audio_path = osp.join(audio_save_path, f"{safe_name}.wav")
            os.makedirs(video_save_path, exist_ok=True)
            video_out = osp.join(video_save_path, f"{safe_name}.mp4")

            # Allow user override of safety margin via env var (seconds).
            try:
                safety_margin = float(os.environ.get("FOLEY_SAFETY_MARGIN", "0.02"))
            except ValueError:
                safety_margin = 0.02

            # Open clips inside context managers to ensure proper release.
            with AudioFileClip(final_audio_path) as audio_clip, VideoFileClip(input_video) as video_clip:
                # Determine reliable duration values from actual clips (could differ slightly from 'duration').
                clip_audio_dur = getattr(audio_clip, 'duration', duration)
                clip_video_dur = getattr(video_clip, 'duration', duration)
                final_duration = min(duration, clip_audio_dur, clip_video_dur)
                # Apply safety margin but keep non-negative
                final_duration = max(0.0, final_duration - safety_margin)
                if final_duration <= 0:
                    print(f"[Warn] Non-positive final duration computed for {safe_name}, skipping mux.")
                else:
                    # Trim to aligned common duration
                    trimmed_audio = audio_clip.subclip(0, final_duration)
                    trimmed_video = video_clip.subclip(0, final_duration)
                    composite = trimmed_video.set_audio(trimmed_audio)
                    # Explicitly set audio fps to match generation (16k) for consistency.
                    composite.write_videofile(
                        video_out,
                        audio_fps=16000,
                        verbose=False,
                        logger=None,
                    )
                    print(f"[Done] Wrote {video_out} (duration ~{final_duration:.3f}s)")


if __name__ == "__main__":
    config = args_parse()
    pipe, vocoder, time_detector = build_models(config)
    if config.pre_download_only:
        print("[Exit] Pre-download complete. Exiting as requested.")
    else:
        run_inference(config, pipe, vocoder, time_detector)
