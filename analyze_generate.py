#!/usr/bin/env python3
"""
Analyze and generate audio for custom videos:
 1. Save original video (copy) and extract original audio.
 2. Save mel spectrogram of original audio (PNG + NPY) with x-axis in seconds and twin axis for frame index.
 3. Generate Foley audio with FoleyCrafter pipeline, save generated audio + combined video, and mel spectrogram likewise.
 4. Overlay detected onset times from the timestamp detector on both original and generated mel plots (if applicable).

Usage example:
  python3 analyze_generate.py \
    --input /data/AVSync15/custom/-QlgGKvOyqs_000073_000083_3.5_7.5.mp4 \
    --ckpt checkpoints --save_dir output_analysis --num_inference_steps 10 \
    --semantic_scale 1.2 --temporal_scale 0.3 --seed 123

Directory structure created under save_dir:
  original/video/, original/audio/, original/mel/
  generated/audio/, generated/video/, generated/mel/

Requires: matplotlib, torchaudio.
"""

import argparse
import glob
import os
import os.path as osp
import re
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchvision
from huggingface_hub import snapshot_download
from moviepy.editor import VideoFileClip, AudioFileClip
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from foleycrafter.models.onset import torch_utils
from foleycrafter.models.time_detector.model import VideoOnsetNet
from foleycrafter.pipelines.auffusion_pipeline import Generator, denormalize_spectrogram
from foleycrafter.utils.util import build_foleycrafter, read_frames_with_moviepy

# Video frame preprocessing (same as inference)
vision_transform_list = [
    torchvision.transforms.Resize((128, 128)),
    torchvision.transforms.CenterCrop((112, 112)),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]
video_transform = torchvision.transforms.Compose(vision_transform_list)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=str, required=True, help='input video file or directory')
    p.add_argument('--ckpt', type=str, default='checkpoints', help='checkpoint directory (contains adapters etc.)')
    p.add_argument('--pretrain', type=str, default='auffusion/auffusion-full-no-adapter', help='audio generator pretrained id/path')
    p.add_argument('--save_dir', type=str, default='output_analysis', help='output base directory')
    p.add_argument('--prompt', type=str, default='', help='text prompt')
    p.add_argument('--nprompt', type=str, default='', help='negative prompt')
    p.add_argument('--semantic_scale', type=float, default=1.0, help='semantic adapter scale')
    p.add_argument('--temporal_scale', type=float, default=0.2, help='temporal controlnet scale')
    p.add_argument('--num_inference_steps', type=int, default=25, help='diffusion steps')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--recursive', action='store_true', help='recurse into directories')
    p.add_argument('--max_frames', type=int, default=150, help='max video frames to feed time detector')
    p.add_argument('--onset_threshold', type=float, default=0.5, help='probability threshold for onset')
    # New analysis / export options
    p.add_argument('--save_onset_prob', action='store_true', help='save onset probability curve (npy + png)')
    p.add_argument('--save_mel_compare', action='store_true', help='save side-by-side (stacked) mel comparison plot')
    p.add_argument('--normalize_generated_audio', action='store_true', help='peak-normalize generated audio to -1 dBFS before saving')
    # Memory / speed control options
    p.add_argument('--fp16', action='store_true', help='use float16 for model inference (saves VRAM)')
    p.add_argument('--frame_step', type=int, default=1, help='sample every Nth frame for CLIP/image encoding')
    p.add_argument('--clip_batch_size', type=int, default=24, help='batch size for CLIP vision encoder to reduce peak memory')
    p.add_argument('--ffmpeg_merge', action='store_true', help='use ffmpeg to mux video and generated audio (more robust)')
    # Onset evaluation against ground truth
    p.add_argument('--gt_onsets', type=str, default=None, help='path to ground-truth onset file (txt: one time(s)/line or json list)')
    p.add_argument('--onset_tolerance_ms', type=float, default=100.0, help='matching tolerance (ms) for GT vs predicted onset')
    p.add_argument('--eval_onset', action='store_true', help='enable evaluation & visualization of onset timing errors')
    # Audio-based onset comparison (original vs generated audio)
    p.add_argument('--eval_audio_onset', action='store_true', help='detect onsets from original & generated audio and evaluate')
    p.add_argument('--audio_onset_method', type=str, default='flux', choices=['flux', 'energy'], help='method for audio onset detection')
    p.add_argument('--audio_onset_threshold', type=float, default=0.3, help='threshold (0-1 normalized) for audio onset peak picking')
    p.add_argument('--min_onset_separation_ms', type=float, default=80.0, help='minimum separation between detected onsets (ms)')
    # Auto GT extraction from original audio
    p.add_argument('--auto_gt_from_audio', action='store_true', help='derive GT onset times from original audio when no gt_onsets file found')
    p.add_argument('--auto_gt_audio_method', type=str, default='flux', choices=['flux','energy'], help='method for auto GT audio onset detection')
    p.add_argument('--auto_gt_audio_threshold', type=float, default=0.35, help='threshold for auto GT audio onset detection (0-1)')
    p.add_argument('--auto_gt_min_sep_ms', type=float, default=80.0, help='minimum separation (ms) for auto GT onsets')
    return p.parse_args()


def build_models(cfg):
    print('[Build] Loading models...')
    pretrained = cfg.pretrain
    if not osp.isdir(pretrained):
        print(f'[Build] Download generator weights: {pretrained}')
        pretrained = snapshot_download(pretrained)

    if not osp.exists(osp.join(cfg.ckpt, 'temporal_adapter.ckpt')):
        # Download FoleyCrafter bundle only if needed (adapters/vocoder stored there)
        print('[Build] Download FoleyCrafter resources (adapters, vocoder)...')
        snapshot_download('ymzhang319/FoleyCrafter', local_dir=cfg.ckpt)

    temporal_ckpt_path = osp.join(cfg.ckpt, 'temporal_adapter.ckpt')
    vocoder = Generator.from_pretrained(cfg.ckpt, subfolder='vocoder').to(cfg.device)

    # Time detector
    time_detector_ckpt = osp.join(cfg.ckpt, 'timestamp_detector.pth.tar')
    time_detector = VideoOnsetNet(False)
    time_detector, _ = torch_utils.load_model(time_detector_ckpt, time_detector, device=cfg.device, strict=True)

    pipe = build_foleycrafter().to(cfg.device)
    if cfg.fp16:
        try:
            pipe.to(torch.float16)
            print('[Build] Pipeline moved to fp16')
        except Exception as e:
            print(f'[Warn] Could not move pipeline to fp16: {e}')
    ckpt = torch.load(temporal_ckpt_path)
    if 'state_dict' in ckpt:  # unwrap
        ckpt = ckpt['state_dict']
    load_sd = {}
    for k, v in ckpt.items():
        load_sd[k[len('module.') :]] = v if k.startswith('module.') else v
    missing, unexpected = pipe.controlnet.load_state_dict(load_sd, strict=False)
    print(f'[Build] ControlNet loaded (missing={len(missing)}, unexpected={len(unexpected)})')

    # Semantic adapter
    pipe.load_ip_adapter(osp.join(cfg.ckpt, 'semantic'), subfolder='', weight_name='semantic_adapter.bin', image_encoder_folder=None)
    pipe.set_ip_adapter_scale(cfg.semantic_scale)

    return pipe, vocoder, time_detector


def collect_videos(input_path: str, recursive: bool) -> List[str]:
    path = osp.abspath(input_path)
    patterns = ['*.mp4', '*.MP4', '*.mov', '*.MOV', '*.mkv', '*.MKV', '*.webm', '*.WEBM']
    videos = []
    if osp.isfile(path):
        return [path]
    if not osp.isdir(path):
        raise FileNotFoundError(f'Input path not found: {path}')
    if recursive:
        for r, _, _ in os.walk(path):
            for pat in patterns:
                videos.extend(glob.glob(osp.join(r, pat)))
    else:
        for pat in patterns:
            videos.extend(glob.glob(osp.join(path, pat)))
    videos = sorted(list({osp.abspath(v) for v in videos}))
    if not videos:
        raise RuntimeError(f'No video files in {path}')
    return videos


def load_gt_onsets(path: str) -> List[float]:
    if path is None:
        return []
    if not osp.exists(path):
        print(f'[Warn] GT onset file not found: {path}')
        return []
    times = []
    try:
        if path.lower().endswith('.json'):
            import json
            with open(path, 'r') as f:
                data = json.load(f)
            if isinstance(data, dict) and 'onsets' in data:
                data = data['onsets']
            for v in data:
                try:
                    times.append(float(v))
                except Exception:
                    pass
        else:  # txt
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        times.append(float(line))
                    except Exception:
                        pass
    except Exception as e:
        print(f'[Warn] Failed to load gt_onsets: {e}')
    times = sorted(t for t in times if t >= 0)
    return times


def match_onsets(gt: List[float], pred: List[float], tol: float) -> dict:
    """Greedy one-to-one matching within tolerance (seconds).
    Returns mapping with lists of matched pairs and unmatched indices.
    """
    gt_used = set()
    pred_used = set()
    matches = []  # (gt_time, pred_time, diff_s)
    gi, pi = 0, 0
    while gi < len(gt) and pi < len(pred):
        g, p = gt[gi], pred[pi]
        diff = p - g
        ad = abs(diff)
        if ad <= tol:
            matches.append((g, p, diff))
            gt_used.add(gi)
            pred_used.add(pi)
            gi += 1
            pi += 1
        else:
            # move the earlier one forward
            if p < g:
                pi += 1
            else:
                gi += 1
    unmatched_gt = [gt[i] for i in range(len(gt)) if i not in gt_used]
    unmatched_pred = [pred[i] for i in range(len(pred)) if i not in pred_used]
    return {
        'matches': matches,
        'unmatched_gt': unmatched_gt,
        'unmatched_pred': unmatched_pred,
    }


def evaluate_onsets(gt: List[float], pred: List[float], tolerance_s: float, duration: float, out_dir: str, prefix: str):
    os.makedirs(out_dir, exist_ok=True)
    res = match_onsets(gt, pred, tolerance_s)
    diffs = [m[2] for m in res['matches']]  # signed
    abs_diffs = [abs(d) for d in diffs]
    import json
    metrics = {}
    tp = len(diffs)
    fp = len(res['unmatched_pred'])
    fn = len(res['unmatched_gt'])
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    metrics.update({
        'gt_count': len(gt),
        'pred_count': len(pred),
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'mae_ms': float(np.mean(abs_diffs) * 1000) if abs_diffs else None,
        'median_abs_ms': float(np.median(abs_diffs) * 1000) if abs_diffs else None,
        'max_abs_ms': float(max(abs_diffs) * 1000) if abs_diffs else None,
        'tolerance_ms': tolerance_s * 1000,
    })
    with open(osp.join(out_dir, f'{prefix}_onset_eval.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Histogram of signed diffs (ms)
    if diffs:
        diffs_ms = np.array(diffs) * 1000
        plt.figure(figsize=(6, 3))
        plt.hist(diffs_ms, bins=30, color='steelblue', edgecolor='black')
        plt.axvline(0, color='red', linestyle='--', linewidth=1)
        plt.title(f'Onset Timing Error (pred-gt) ms\nmean={diffs_ms.mean():.1f}, med={np.median(diffs_ms):.1f}')
        plt.xlabel('Error (ms)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(osp.join(out_dir, f'{prefix}_onset_error_hist.png'), dpi=140)
        plt.close()

        # CDF of absolute errors
        abs_ms = np.sort(np.abs(diffs_ms))
        cdf = np.arange(1, len(abs_ms)+1)/len(abs_ms)
        plt.figure(figsize=(5,3))
        plt.plot(abs_ms, cdf, label='CDF')
        plt.axvline(metrics['mae_ms'], color='orange', linestyle='--', label='MAE') if metrics['mae_ms'] else None
        plt.axvline(metrics['median_abs_ms'], color='green', linestyle='--', label='Median') if metrics['median_abs_ms'] else None
        plt.xlabel('Absolute Error (ms)')
        plt.ylabel('Cumulative Fraction')
        plt.title('Absolute Onset Error CDF')
        plt.grid(alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(osp.join(out_dir, f'{prefix}_onset_error_cdf.png'), dpi=140)
        plt.close()

    # Scatter: GT vs Pred with error lines
    gts = [m[0] for m in res['matches']]
    preds = [m[1] for m in res['matches']]
    plt.figure(figsize=(6.4, 3.4))
    plt.scatter(gts, preds, c='royalblue', edgecolors='k', s=30, label='Matched')
    # Ideal line
    lim_min = 0
    lim_max = max(duration, (gts + preds and max(gts+preds)))
    plt.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=1, label='Ideal y=x')
    plt.xlabel('GT Onset (s)')
    plt.ylabel('Pred Onset (s)')
    plt.xlim(lim_min, lim_max)
    plt.ylim(lim_min, lim_max)
    plt.title('GT vs Pred Onset Times')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(osp.join(out_dir, f'{prefix}_onset_scatter.png'), dpi=150)
    plt.close()

    # Timeline plot
    plt.figure(figsize=(8, 2.6))
    y_gt, y_pred = 1.0, 0.0
    for t in gt:
        plt.vlines(t, y_gt-0.3, y_gt+0.3, color='black', linewidth=2)
    for t in pred:
        plt.vlines(t, y_pred-0.3, y_pred+0.3, color='royalblue', linewidth=2)
    plt.yticks([y_pred, y_gt], ['Pred', 'GT'])
    plt.xlim(0, duration)
    plt.xlabel('Time (s)')
    plt.title('Onset Timeline (GT vs Pred)')
    plt.tight_layout()
    plt.savefig(osp.join(out_dir, f'{prefix}_onset_timeline.png'), dpi=140)
    plt.close()

    print(f"[OnsetEval] {prefix} precision={prec:.3f} recall={rec:.3f} f1={f1:.3f} MAE(ms)={metrics['mae_ms']}")
    return metrics


def detect_audio_onsets(audio: np.ndarray, sr: int, method: str = 'flux', threshold: float = 0.3, min_sep_ms: float = 80.0,
                         n_fft: int = 1024, hop: int = 256) -> Tuple[List[float], dict]:
    """Simple audio onset detection (spectral flux or energy).
    Returns (onset_times, debug_info)
    audio: shape (samples,) float32
    threshold: applied after normalization to [0,1]
    """
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)
    # Frame the audio
    pad = (n_fft - hop)
    audio_padded = np.pad(audio, (pad, 0))
    n_frames = 1 + (len(audio_padded) - n_fft) // hop
    window = np.hanning(n_fft).astype(np.float32)
    mags = []
    energy = []
    for i in range(n_frames):
        start = i * hop
        frame = audio_padded[start:start + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        frame_w = frame * window
        spec = np.fft.rfft(frame_w)
        mag = np.abs(spec)
        mags.append(mag)
        energy.append(np.sum(frame_w ** 2))
    mags = np.stack(mags, 0)
    energy = np.array(energy)
    if method == 'flux':
        # Spectral flux (positive changes only)
        norm = mags / (np.sum(mags, axis=1, keepdims=True) + 1e-8)
        diff = norm[1:] - norm[:-1]
        flux = np.maximum(0.0, np.sum(diff, axis=1))
        curve = np.concatenate([[0.0], flux])
    else:
        # Energy difference
        diff = energy[1:] - energy[:-1]
        curve = np.concatenate([[0.0], np.maximum(0.0, diff)])
    # Smooth
    # Optional: attempt to use scipy if available (not required)
    try:
        import scipy.signal as sg  # noqa: F401
    except ImportError:
        sg = None  # type: ignore
    # Simple moving average smoothing
    k = 3
    if len(curve) >= k:
        kernel = np.ones(k) / k
        curve_s = np.convolve(curve, kernel, mode='same')
    else:
        curve_s = curve
    # Normalize
    if curve_s.max() > 0:
        norm_curve = curve_s / curve_s.max()
    else:
        norm_curve = curve_s
    # Peak picking
    min_sep_frames = int((min_sep_ms / 1000.0) * sr / hop)
    peaks = []
    last_peak = -1e9
    for i in range(1, len(norm_curve) - 1):
        if norm_curve[i] >= threshold and norm_curve[i] > norm_curve[i - 1] and norm_curve[i] >= norm_curve[i + 1]:
            if i - last_peak >= min_sep_frames:
                peaks.append(i)
                last_peak = i
    onset_times = [p * hop / sr for p in peaks]
    dbg = {
        'curve': norm_curve.astype(np.float32),
        'peaks_frames': peaks,
        'hop': hop,
        'sr': sr,
    }
    return onset_times, dbg


def save_audio_onset_plot(dbg_orig: dict, dbg_gen: dict, onsets_orig: List[float], onsets_gen: List[float], duration: float, out_path: str):
    curve_o = dbg_orig['curve'] if dbg_orig else None
    curve_g = dbg_gen['curve'] if dbg_gen else None
    hop = dbg_orig['hop'] if dbg_orig else dbg_gen['hop']
    sr = dbg_orig['sr'] if dbg_orig else dbg_gen['sr']
    t_curve_o = np.arange(len(curve_o)) * hop / sr if curve_o is not None else None
    t_curve_g = np.arange(len(curve_g)) * hop / sr if curve_g is not None else None
    plt.figure(figsize=(10, 3.2))
    if curve_o is not None:
        plt.plot(t_curve_o, curve_o, label='Orig curve', alpha=0.7)
    if curve_g is not None:
        plt.plot(t_curve_g, curve_g, label='Gen curve', alpha=0.7)
    for t in onsets_orig:
        plt.axvline(t, color='black', linestyle='-', linewidth=1.2, alpha=0.8)
    for t in onsets_gen:
        plt.axvline(t, color='royalblue', linestyle='--', linewidth=1.0, alpha=0.8)
    plt.xlim(0, duration)
    plt.xlabel('Time (s)')
    plt.ylabel('Norm curve')
    plt.title('Audio Onset Curves & Detected Onsets')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def sanitize_name(name: str) -> str:
    safe = re.sub(r'[^A-Za-z0-9._-]+', '_', name).lstrip('-_')
    return safe or 'out'


def extract_original_audio(video_path: str, target_sr: int = 16000) -> Optional[np.ndarray]:
    """Robustly extract mono audio at target_sr.

    Strategy:
      1. Try direct to_soundarray (fast path).
      2. On failure, write audio to a temp WAV via ffmpeg (moviepy) then load with soundfile.
    Returns None if no audio track or extraction failed.
    """
    try:
        clip = VideoFileClip(video_path)
    except Exception as e:
        print(f"[Warn] Could not open video for audio extraction: {e}")
        return None
    if clip.audio is None:
        clip.close()
        return None
    # Fast path
    try:
        arr = clip.audio.to_soundarray(fps=target_sr)
        clip.close()
        if arr.ndim == 2:
            arr = arr.mean(axis=1)
        return arr.astype(np.float32)
    except Exception as e:
        print(f"[Info] Fast audio extraction failed ({e}); falling back to temp WAV export.")
    # Fallback path
    try:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(tmp_fd)
        # Suppress verbose moviepy logging
        clip.audio.write_audiofile(tmp_path, fps=target_sr, verbose=False, logger=None)
        clip.close()
        data, sr = sf.read(tmp_path)
        os.remove(tmp_path)
        if data.ndim == 2:
            data = data.mean(axis=1)
        if sr != target_sr:
            # Resample if needed (rare if fps param respected)
            data_t = torch.from_numpy(data).float().unsqueeze(0)
            data = torchaudio.functional.resample(data_t, sr, target_sr).squeeze(0).numpy()
        return data.astype(np.float32)
    except Exception as e2:
        print(f"[Warn] Fallback audio extraction failed: {e2}")
        try:
            clip.close()
        except Exception:
            pass
        return None


def save_mel(waveform: np.ndarray, sr: int, out_npy: str, out_png: str, title: str, video_duration: float, video_fps: float,
             onset_times: Optional[List[float]] = None, onset_label: str = 'Onsets'):
    # waveform shape [T]
    if waveform.dtype != np.float32:
        waveform = waveform.astype(np.float32)
    tensor = torch.from_numpy(waveform).unsqueeze(0)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=1024, hop_length=256, n_mels=128, center=True, power=2.0
    )
    mel = mel_transform(tensor)  # [1, n_mels, time]
    mel_db = 10 * torch.log10(mel + 1e-10).squeeze(0)  # [n_mels, time]
    np.save(out_npy, mel_db.cpu().numpy())

    # time axis for mel frames
    hop = 256
    times = np.arange(mel_db.shape[1]) * hop / sr  # seconds
    fig, ax = plt.subplots(figsize=(8, 3))
    extent = [0, times[-1] if len(times)>0 else video_duration, 0, mel_db.shape[0]]
    im = ax.imshow(mel_db.cpu().numpy(), origin='lower', aspect='auto', cmap='magma', extent=extent)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mel bin')
    ax.set_title(title)
    fig.colorbar(im, ax=ax, pad=0.01, label='dB')

    # Align x-limit to video duration for consistency
    ax.set_xlim(0, video_duration)

    # Twin axis for frame index
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    frame_ticks_sec = np.linspace(0, video_duration, num=min(6, int(video_duration) + 2))
    ax_top.set_xticks(frame_ticks_sec)
    ax_top.set_xticklabels([f'{int(s * video_fps)}f' for s in frame_ticks_sec])
    ax_top.set_xlabel('Video Frame Index')

    # Onset overlay
    if onset_times:
        for t in onset_times:
            ax.axvline(t, color='cyan', alpha=0.6, linewidth=1)
        if onset_times:
            ax.legend([onset_label], loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    return mel_db.cpu().numpy()


def compute_mel_db(waveform: np.ndarray, sr: int) -> np.ndarray:
    if waveform.dtype != np.float32:
        waveform = waveform.astype(np.float32)
    tensor = torch.from_numpy(waveform).unsqueeze(0)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=1024, hop_length=256, n_mels=128, center=True, power=2.0
    )
    mel = mel_transform(tensor)
    mel_db = 10 * torch.log10(mel + 1e-10).squeeze(0)
    return mel_db.cpu().numpy()


def save_mel_compare(orig_wave: Optional[np.ndarray], gen_wave: np.ndarray, sr: int, duration: float, fps: float,
                     onset_times: List[float], out_png: str, safe_name: str):
    try:
        mel_orig = compute_mel_db(orig_wave, sr) if orig_wave is not None else None
        mel_gen = compute_mel_db(gen_wave, sr)
        hop = 256
        def time_axis(m):
            return np.arange(m.shape[1]) * hop / sr
        times_gen = time_axis(mel_gen)
        fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
        if mel_orig is not None:
            im0 = axes[0].imshow(mel_orig, origin='lower', aspect='auto', cmap='magma', extent=[0, duration, 0, mel_orig.shape[0]])
            axes[0].set_ylabel('Mel bin')
            axes[0].set_title(f'Original Mel: {safe_name}')
            for t in onset_times:
                axes[0].axvline(t, color='cyan', alpha=0.5, linewidth=0.8)
            fig.colorbar(im0, ax=axes[0], pad=0.01)
        else:
            axes[0].text(0.5, 0.5, 'No original audio', ha='center', va='center')
            axes[0].set_xlim(0, duration)
            axes[0].set_ylabel('Mel bin')
        im1 = axes[1].imshow(mel_gen, origin='lower', aspect='auto', cmap='magma', extent=[0, duration, 0, mel_gen.shape[0]])
        axes[1].set_ylabel('Mel bin')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_title('Generated Mel')
        for t in onset_times:
            axes[1].axvline(t, color='cyan', alpha=0.5, linewidth=0.8)
        fig.colorbar(im1, ax=axes[1], pad=0.01)
        # Frame tick axis on top of top subplot
        axes[0].set_xlim(0, duration)
        axes[1].set_xlim(0, duration)
        frame_ticks_sec = np.linspace(0, duration, num=min(6, int(duration) + 2))
        ax_top = axes[0].twiny()
        ax_top.set_xlim(axes[0].get_xlim())
        ax_top.set_xticks(frame_ticks_sec)
        ax_top.set_xticklabels([f'{int(s*fps)}f' for s in frame_ticks_sec])
        ax_top.set_xlabel('Video Frame Index')
        if onset_times:
            axes[1].legend(['Onsets'], loc='upper right', fontsize=8)
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f'[Warn] Failed to save mel comparison: {e}')


def detect_onsets(frames: np.ndarray, time_detector: VideoOnsetNet, duration: float, device: str, threshold: float) -> List[float]:
    # frames shape [N, H, W, C]
    tframes = torch.FloatTensor(frames).permute(0, 3, 1, 2)
    tframes = video_transform(tframes)
    tframes = {'frames': tframes.unsqueeze(0).permute(0, 2, 1, 3, 4)}
    preds = time_detector(tframes)
    preds = torch.sigmoid(preds)[0].detach().cpu().numpy()  # [N]
    onset_idx = np.where(preds > threshold)[0]
    onset_times = (onset_idx / len(preds)) * duration
    return onset_times.tolist()


def generate_audio(pipe, vocoder, image_embeddings: torch.Tensor, time_condition: torch.Tensor, cfg, generator) -> np.ndarray:
    sample = pipe(
        prompt=cfg.prompt,
        negative_prompt=cfg.nprompt,
        ip_adapter_image_embeds=image_embeddings,
        image=time_condition,
        controlnet_conditioning_scale=cfg.temporal_scale,
        num_inference_steps=cfg.num_inference_steps,
        height=256,
        width=1024,
        output_type='pt',
        generator=generator,
    )
    audio_img = sample.images[0]
    audio = denormalize_spectrogram(audio_img)
    audio = vocoder.inference(audio, lengths=160000)[0]
    # Support both torch.Tensor and numpy array return types
    if isinstance(audio, torch.Tensor):
        return audio.detach().cpu().numpy()
    return np.asarray(audio)


def build_time_condition(preds_tensor: torch.Tensor, duration: float, device: str) -> torch.Tensor:
    # Reproduce inference logic mapping 1024 width ~ 10s segments
    preds = torch.sigmoid(preds_tensor)[0]
    time_condition = [
        -1 if preds[int(i / (1024 / 10 * duration) * 150)] < 0.5 else 1 for i in range(int(1024 / 10 * duration))
    ]
    time_condition += [-1] * (1024 - len(time_condition))
    cond = (
        torch.FloatTensor(time_condition).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1, 1, 256, 1).to(device)
    )
    return cond


def process_video(video_path: str, cfg, pipe, vocoder, time_detector, gt_onsets: Optional[List[float]] = None):
    safe_name = sanitize_name(Path(video_path).stem)
    print(f'[*] Processing {video_path} -> {safe_name}')

    # Directories
    orig_video_dir = osp.join(cfg.save_dir, 'original', 'video')
    orig_audio_dir = osp.join(cfg.save_dir, 'original', 'audio')
    orig_mel_dir = osp.join(cfg.save_dir, 'original', 'mel')
    gen_audio_dir = osp.join(cfg.save_dir, 'generated', 'audio')
    gen_video_dir = osp.join(cfg.save_dir, 'generated', 'video')
    gen_mel_dir = osp.join(cfg.save_dir, 'generated', 'mel')
    for d in [orig_video_dir, orig_audio_dir, orig_mel_dir, gen_audio_dir, gen_video_dir, gen_mel_dir]:
        os.makedirs(d, exist_ok=True)

    # Copy original video
    dst_video_path = osp.join(orig_video_dir, f'{safe_name}_orig.mp4')
    if not osp.exists(dst_video_path):
        shutil.copy2(video_path, dst_video_path)

    # Extract original audio
    sample_rate = 16000
    orig_audio = extract_original_audio(video_path, sample_rate)
    orig_wav_path = None
    onset_times = []

    clip = VideoFileClip(video_path)
    duration = clip.duration
    fps = clip.fps if clip.fps else 30.0
    clip.close()

    frames, _ = read_frames_with_moviepy(video_path, max_frame_nums=cfg.max_frames)
    # Frame sampling (for CLIP embedding only) to reduce memory usage
    sampled_frames = frames[:: max(1, cfg.frame_step)]
    # Onsets from time detector (needs transformed frames)
    with torch.no_grad():
        tframes = torch.FloatTensor(frames).permute(0, 3, 1, 2)
        tframes_t = video_transform(tframes)
        inputs = {'frames': tframes_t.unsqueeze(0).permute(0, 2, 1, 3, 4)}
        preds_tensor = time_detector(inputs)
        preds_prob = torch.sigmoid(preds_tensor)[0].detach().cpu().numpy()
        onset_times = detect_onsets(frames, time_detector, duration, cfg.device, cfg.onset_threshold)

    # Save onset probability curve if requested
    if cfg.save_onset_prob:
        analysis_dir = osp.join(cfg.save_dir, 'analysis')
        os.makedirs(analysis_dir, exist_ok=True)
        prob_npy = osp.join(analysis_dir, f'{safe_name}_onset_prob.npy')
        np.save(prob_npy, preds_prob)
        try:
            x = np.linspace(0, duration, num=len(preds_prob))
            plt.figure(figsize=(8, 2.2))
            plt.plot(x, preds_prob, label='Onset Prob', linewidth=1)
            plt.axhline(cfg.onset_threshold, color='red', linestyle='--', label='Threshold')
            for t in onset_times:
                plt.axvline(t, color='cyan', alpha=0.4, linewidth=0.8)
            plt.xlim(0, duration)
            plt.xlabel('Time (s)')
            plt.ylabel('Prob')
            plt.title(f'Onset Probability: {safe_name}')
            plt.legend(loc='upper right', fontsize=7)
            plt.tight_layout()
            plt.savefig(osp.join(analysis_dir, f'{safe_name}_onset_prob.png'), dpi=150)
            plt.close()
        except Exception as e:
            print(f'[Warn] Could not save onset probability plot: {e}')

    # Save original audio + mel
    mel_orig_db = None
    if orig_audio is not None:
        orig_wav_path = osp.join(orig_audio_dir, f'{safe_name}_orig.wav')
        sf.write(orig_wav_path, orig_audio, sample_rate)
        mel_orig_db = save_mel(
            orig_audio,
            sample_rate,
            osp.join(orig_mel_dir, f'{safe_name}_orig_mel.npy'),
            osp.join(orig_mel_dir, f'{safe_name}_orig_mel.png'),
            f'Original Mel: {safe_name}',
            duration,
            fps,
            onset_times=onset_times,
            onset_label='Pred Onsets',
        )
    else:
        print('[Info] No original audio track present.')

    # Prepare inputs for generation
    # time condition from preds_tensor (NOT thresholded) replicating pipeline usage
    time_condition = build_time_condition(preds_tensor, duration, cfg.device)

    image_processor = CLIPImageProcessor()
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        'h94/IP-Adapter', subfolder='models/image_encoder'
    ).to(cfg.device)
    if cfg.fp16:
        try:
            image_encoder.to(torch.float16)
        except Exception as e:
            print(f'[Warn] Could not move image encoder to fp16: {e}')

    # Batch encode sampled frames to avoid OOM
    clip_batch = cfg.clip_batch_size
    embeds_list = []
    image_encoder.eval()
    with torch.no_grad():
        for i in range(0, len(sampled_frames), clip_batch):
            batch_frames = sampled_frames[i:i+clip_batch]
            batch_inputs = image_processor(images=batch_frames, return_tensors='pt')
            batch_inputs = {k: v.to(cfg.device) for k, v in batch_inputs.items()}
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=cfg.fp16):
                batch_embeds = image_encoder(**batch_inputs).image_embeds  # [B, D]
            embeds_list.append(batch_embeds.float())  # keep in float32 for downstream stability
            # Free batch tensors
            del batch_inputs, batch_embeds
            torch.cuda.empty_cache()
    image_embeddings = torch.cat(embeds_list, dim=0).mean(dim=0, keepdim=True)  # [1, D]
    image_embeddings = image_embeddings.unsqueeze(0).unsqueeze(0)  # [1,1,1,D]
    neg_image_embeddings = torch.zeros_like(image_embeddings)
    image_embeddings = torch.cat([neg_image_embeddings, image_embeddings], dim=1)
    del embeds_list
    torch.cuda.empty_cache()

    torch.manual_seed(cfg.seed)
    generator = torch.Generator(device=cfg.device).manual_seed(cfg.seed)

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=cfg.fp16):
            gen_audio = generate_audio(pipe, vocoder, image_embeddings, time_condition, cfg, generator)
    # Basic sanity amplitude scaling if vocoder returned int16-range values
    if gen_audio.dtype != np.float32:
        gen_audio = gen_audio.astype(np.float32)
    peak_abs = float(np.max(np.abs(gen_audio))) if gen_audio.size else 0.0
    if peak_abs > 2.0:  # likely int16 scale
        print(f'[Info] Auto-scaling generated audio from peak {peak_abs:.1f} (>2.0) assuming int16 range.')
        gen_audio = gen_audio / 32768.0
        peak_abs = float(np.max(np.abs(gen_audio)))
    # Trim/pad to video duration
    max_len = int(duration * 16000)
    gen_audio = gen_audio[:max_len]

    # Optional normalization
    if cfg.normalize_generated_audio:
        if gen_audio.dtype != np.float32:
            gen_audio = gen_audio.astype(np.float32)
        peak = float(np.max(np.abs(gen_audio)))
        if peak > 1e-6:
            target_peak = 10 ** (-1/20)  # -1 dBFS
            gain = target_peak / peak
            gen_audio = np.clip(gen_audio * gain, -1.0, 1.0).astype(np.float32)
            print(f'[Norm] Applied peak normalization (orig_peak={peak:.6f}, gain={gain:.6f})')
        else:
            print(f'[Norm] Skip normalization (silence peak={peak:.2e}).')
    gen_wav_path = osp.join(gen_audio_dir, f'{safe_name}_gen.wav')
    sf.write(gen_wav_path, gen_audio, sample_rate)

    # Save mel of generated
    mel_gen_db = save_mel(
    gen_audio,
    sample_rate,
        osp.join(gen_mel_dir, f'{safe_name}_gen_mel.npy'),
        osp.join(gen_mel_dir, f'{safe_name}_gen_mel.png'),
        f'Generated Mel: {safe_name}',
        duration,
        fps,
        onset_times=onset_times,
        onset_label='Pred Onsets',
    )

    # Mel comparison
    if cfg.save_mel_compare:
        compare_png = osp.join(cfg.save_dir, 'analysis', f'{safe_name}_mel_compare.png')
        os.makedirs(osp.dirname(compare_png), exist_ok=True)
        save_mel_compare(
            orig_audio if orig_audio is not None else None,
            gen_audio,
            sample_rate,
            duration,
            fps,
            onset_times,
            compare_png,
            safe_name,
        )

    # Combine generated audio + original video (duration aligned)
    out_video_path = osp.join(gen_video_dir, f'{safe_name}_gen.mp4')
    safe_end = max(0.0, duration - 1e-3)
    if cfg.ffmpeg_merge:
        # Prefer ffmpeg merge for robustness
        import subprocess, shlex
        # -t safe_end 으로 양쪽 동일 트림, 오디오/비디오 길이 초과 접근 방지
        cmd = f"ffmpeg -y -loglevel error -i {shlex.quote(video_path)} -i {shlex.quote(gen_wav_path)} -t {safe_end:.3f} -c:v copy -c:a aac -shortest {shlex.quote(out_video_path)}"
        try:
            subprocess.check_call(cmd, shell=True)
        except subprocess.CalledProcessError as e:
            print(f"[Warn] ffmpeg merge failed ({e}); falling back to MoviePy merge.")
            cfg.ffmpeg_merge = False  # fallback
    if not cfg.ffmpeg_merge:
        from moviepy.audio.AudioClip import AudioArrayClip
        try:
            # Use in-memory array to avoid t>duration issues
            arr = gen_audio.copy()
            arr = arr[: int(safe_end * 16000)]
            if arr.ndim == 1:
                arr_st = np.stack([arr, arr], axis=1)  # stereo duplicate
            else:
                arr_st = arr
            audio_clip = AudioArrayClip(arr_st, fps=16000)
            video_clip = VideoFileClip(video_path).subclip(0, safe_end)
            video_clip.audio = audio_clip
            video_clip.write_videofile(out_video_path)
        finally:
            try:
                audio_clip.close()
            except Exception:
                pass
            try:
                video_clip.close()
            except Exception:
                pass

    print(f'[Done] {safe_name}: orig_audio={bool(orig_audio is not None)} generated -> {out_video_path}')

    # Auto-GT generation from original audio if requested and no GT provided
    auto_gt_used = False
    if cfg.eval_onset and (not gt_onsets) and cfg.auto_gt_from_audio and (orig_audio is not None):
        try:
            auto_gt, dbg_auto = detect_audio_onsets(
                orig_audio.copy(), sample_rate,
                method=cfg.auto_gt_audio_method,
                threshold=cfg.auto_gt_audio_threshold,
                min_sep_ms=cfg.auto_gt_min_sep_ms
            )
            gt_onsets = auto_gt
            auto_gt_used = True
            analysis_dir = osp.join(cfg.save_dir, 'analysis')
            os.makedirs(analysis_dir, exist_ok=True)
            np.save(osp.join(analysis_dir, f'{safe_name}_auto_gt_onsets.npy'), np.array(auto_gt, dtype=np.float32))
            with open(osp.join(analysis_dir, f'{safe_name}_auto_gt_onsets.txt'), 'w') as fgt:
                for t in auto_gt:
                    fgt.write(f'{t:.6f}\n')
        except Exception as e:
            print(f'[Warn] auto GT extraction failed: {e}')

    # Onset evaluation (predicted visual onsets vs GT list — GT may be auto)
    if cfg.eval_onset and gt_onsets:
        analysis_dir = osp.join(cfg.save_dir, 'analysis')
        evaluate_onsets(
            gt_onsets,
            onset_times,  # predicted from video frames
            cfg.onset_tolerance_ms / 1000.0,
            duration,
            analysis_dir,
            safe_name + ('_autoGT' if auto_gt_used else ''),
        )

    # Audio-based onset evaluation (original vs generated audio)
    if cfg.eval_audio_onset and (orig_audio is not None) and (gen_audio is not None):
        analysis_dir = osp.join(cfg.save_dir, 'analysis')
        os.makedirs(analysis_dir, exist_ok=True)
        try:
            onsets_orig, dbg_o = detect_audio_onsets(orig_audio.copy(), sample_rate, method=cfg.audio_onset_method,
                                                     threshold=cfg.audio_onset_threshold, min_sep_ms=cfg.min_onset_separation_ms)
            onsets_gen, dbg_g = detect_audio_onsets(gen_audio.copy(), sample_rate, method=cfg.audio_onset_method,
                                                    threshold=cfg.audio_onset_threshold, min_sep_ms=cfg.min_onset_separation_ms)
            # Save curves / peaks
            np.save(osp.join(analysis_dir, f'{safe_name}_audio_onsets_orig.npy'), np.array(onsets_orig, dtype=np.float32))
            np.save(osp.join(analysis_dir, f'{safe_name}_audio_onsets_gen.npy'), np.array(onsets_gen, dtype=np.float32))
            save_audio_onset_plot(dbg_o, dbg_g, onsets_orig, onsets_gen, duration,
                                  osp.join(analysis_dir, f'{safe_name}_audio_onset_curves.png'))
            # Evaluate timing error using orig as GT
            evaluate_onsets(onsets_orig, onsets_gen, cfg.onset_tolerance_ms / 1000.0, duration, analysis_dir,
                            f'{safe_name}_audio')
        except Exception as e:
            print(f'[Warn] audio onset evaluation failed: {e}')


def main():
    cfg = parse_args()
    os.makedirs(cfg.save_dir, exist_ok=True)
    pipe, vocoder, time_detector = build_models(cfg)
    vids = collect_videos(cfg.input, cfg.recursive)
    print(f'[Info] Found {len(vids)} video(s).')
    # Pre-load global GT if single file (exists)
    global_gt: List[float] = []
    global_gt_loaded = False
    if cfg.gt_onsets:
        if osp.isfile(cfg.gt_onsets):
            global_gt = load_gt_onsets(cfg.gt_onsets)
            global_gt_loaded = True
        elif osp.isdir(cfg.gt_onsets):
            print(f'[Info] Using GT onset directory: {cfg.gt_onsets}')
        else:
            # treat as basename pattern later
            print(f'[Info] GT onset spec will be resolved per-video: {cfg.gt_onsets}')

    def resolve_gt_for_video(video_path: str) -> List[float]:
        if not cfg.gt_onsets:
            return []
        if global_gt_loaded:
            return global_gt
        # Directory mode
        if osp.isdir(cfg.gt_onsets):
            stem = osp.splitext(osp.basename(video_path))[0]
            cand_txt = osp.join(cfg.gt_onsets, stem + '.txt')
            cand_json = osp.join(cfg.gt_onsets, stem + '.json')
            if osp.exists(cand_txt):
                return load_gt_onsets(cand_txt)
            if osp.exists(cand_json):
                return load_gt_onsets(cand_json)
            return []
        # Basename mode (user passed something like gunshots_onsets.txt)
        base_candidate = cfg.gt_onsets
        # Look in video directory first
        vdir = osp.dirname(video_path)
        cand = osp.join(vdir, base_candidate)
        if osp.exists(cand):
            return load_gt_onsets(cand)
        # Fallback: relative to CWD
        if osp.exists(base_candidate):
            return load_gt_onsets(base_candidate)
        return []

    for v in vids:
        per_gt = resolve_gt_for_video(v)
        if cfg.eval_onset and cfg.gt_onsets and not per_gt:
            stem = osp.splitext(osp.basename(v))[0]
            print(f'[Warn] No GT onsets found for video {stem}')
        process_video(v, cfg, pipe, vocoder, time_detector, gt_onsets=per_gt)


if __name__ == '__main__':
    main()
