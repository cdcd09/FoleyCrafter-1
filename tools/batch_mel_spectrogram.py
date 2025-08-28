"""Batch Mel Spectrogram Generator

주어진 오디오 디렉토리를 순회하며 각 파일의 Mel Spectrogram 이미지를 생성합니다.

기능 요약:
 - 지원 확장자: .wav .mp3 .flac .ogg .m4a .aac
 - librosa 로 로드 (mono 변환 옵션)
 - 멜 스펙트로그램 계산 후 dB 변환
 - X축을 ms 단위로 (촘촘하지만 과밀하지 않도록 자동 간격 선택)
 - PNG (또는 지정 포맷) 저장 + 선택적으로 numpy .npy 저장
 - 진행 상황 tqdm 표시
 - 이미 존재하는 결과는 기본 스킵(옵션으로 덮어쓰기)

사용 예:
 python3 tools/batch_mel_spectrogram.py \
  --input_dir dataset/custom/gen/audio \
  --output_dir dataset/custom/gen \
  --sr 16000 --n_fft 1024 --hop_length 320 --n_mels 256 \
  --colorbar --save_npy --verbose

축(ms) 간격 자동 규칙:
  전체 길이(ms)에 따라 8~30개 tick 목표로 미리 정의된 후보 간격(ms)

필요 패키지: librosa, matplotlib, soundfile (requirements.txt 이미 포함)
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List

import librosa
import numpy as np
import soundfile as sf  # noqa: F401 (librosa backend가 soundfile 필요)
from tqdm import tqdm

import matplotlib

# Headless 환경 안전
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SUPPORTED_AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
SUPPORTED_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
SUPPORTED_EXT = SUPPORTED_AUDIO_EXT | SUPPORTED_VIDEO_EXT

# moviepy 가 동일 패턴으로 반복 실패할 경우 남은 파일에서는 시도 건너뛰기 위한 플래그
_MOVIEPY_DISABLED = False


def iter_audio_files(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT:
            yield p


def choose_tick_interval(total_ms: float) -> int:
    """전체 길이(ms)에 따라 적당한 tick 간격(ms)을 선택.

    목표 tick 수: 8~30 사이.
    """
    candidates = [5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000]
    for c in candidates:
        n = total_ms / c
        if 8 <= n <= 30:
            return c
    # fallback: 가장 가까운 값 선택
    best = min(candidates, key=lambda c: abs((total_ms / c) - 18))
    return best


def ms_ticks(ax, total_frames: int, hop_length: int, sr: int):
    total_ms = (total_frames * hop_length / sr) * 1000.0
    interval = choose_tick_interval(total_ms)
    max_ms = int(round(total_ms))
    ticks_ms = list(range(0, max_ms + 1, interval))
    # frame index -> 시간(ms): frame * hop_length / sr * 1000
    frames_per_ms = sr / (hop_length * 1000.0)
    tick_positions = [ms * frames_per_ms for ms in ticks_ms]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(ms) for ms in ticks_ms], rotation=0, fontsize=8)
    ax.set_xlabel("Time (ms)")


def make_mel(
    y: np.ndarray,
    sr: int,
    n_mels: int,
    hop_length: int,
    n_fft: int,
    fmin: int,
    fmax: int | None,
    power: float,
) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=power,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db


def save_figure(
    mel_db: np.ndarray,
    out_img: Path,
    sr: int,
    hop_length: int,
    cmap: str,
    add_colorbar: bool,
):
    plt.figure(figsize=(10, 3), dpi=150)
    ax = plt.gca()
    img = ax.imshow(
        mel_db,
        origin="lower",
        aspect="auto",
        interpolation="none",
        cmap=cmap,
    )
    if add_colorbar:
        plt.colorbar(img, ax=ax, pad=0.01, fraction=0.04, label="dB")
    ms_ticks(ax, mel_db.shape[1], hop_length, sr)
    ax.set_ylabel("Mel Bin")
    ax.set_title(out_img.stem)
    plt.tight_layout()
    out_img.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_img)
    plt.close()


def _extract_audio_ffmpeg(path: Path, args: argparse.Namespace):
    """ffmpeg CLI 로 오디오 추출 (임시 wav) 후 librosa 로드.

    장점: moviepy 내부 파이프라인 오류 회피, 다양한 코덱 호환.
    """
    channels = 1 if args.mono else 2
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_name = tmp.name
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-vn",
        "-ac",
        str(channels),
        "-ar",
        str(args.sr),
        "-y",
        tmp_name,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:  # noqa: BLE001
        if os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except OSError:
                pass
        raise RuntimeError(f"ffmpeg 추출 실패: {e.stderr.decode(errors='ignore')[:180]}") from e
    try:
        y, sr = librosa.load(tmp_name, sr=args.sr, mono=args.mono)
    finally:
        try:
            os.remove(tmp_name)
        except OSError:
            pass
    return y, sr


def load_media(path: Path, args: argparse.Namespace):
    """오디오 또는 비디오 파일 로드 -> (waveform, sr)

    비디오의 경우 moviepy 로 오디오를 추출 후 numpy 반환.
    """
    ext = path.suffix.lower()
    if ext in SUPPORTED_AUDIO_EXT:
        # librosa 로드
        y, sr = librosa.load(path, sr=args.sr, mono=args.mono)
        return y, sr

    global _MOVIEPY_DISABLED  # noqa: PLW0603
    if ext in SUPPORTED_VIDEO_EXT:
        backend_order = []
        if args.video_backend == "auto":
            # moviepy 전역 비활성화 되었으면 ffmpeg 먼저
            backend_order = (["moviepy"] if not _MOVIEPY_DISABLED else []) + ["ffmpeg"]
        elif args.video_backend == "moviepy":
            backend_order = ["moviepy"]
        else:
            backend_order = ["ffmpeg"]

        last_err = None
        for backend in backend_order:
            if backend == "moviepy":
                try:
                    from moviepy.editor import VideoFileClip  # 지연 임포트
                    with VideoFileClip(str(path)) as clip:
                        if clip.audio is None:
                            raise RuntimeError("오디오 트랙 없음")
                        audio_fps = args.sr
                        arr = clip.audio.to_soundarray(fps=audio_fps)
                        if not isinstance(arr, np.ndarray):
                            raise RuntimeError("moviepy to_soundarray 반환 타입 오류")
                        if arr.ndim == 2:
                            if args.mono:
                                arr = arr.mean(axis=1)
                            else:
                                arr = arr[:, 0]
                        y = arr.astype(np.float32)
                        peak = np.max(np.abs(y)) or 1.0
                        if peak > 1.0:
                            y = y / peak
                        return y, audio_fps
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    if args.verbose:
                        print(f"[WARN] moviepy 추출 실패, ffmpeg 시도: {e}")
                    # 특정 에러 패턴이면 앞으로 moviepy 건너뜀
                    msg = str(e)
                    if "arrays to stack" in msg or "to_soundarray" in msg:
                        _MOVIEPY_DISABLED = True
                        if args.verbose:
                            print("[INFO] moviepy 반복 실패 패턴 감지 -> 이후 ffmpeg만 사용")
            elif backend == "ffmpeg":
                try:
                    return _extract_audio_ffmpeg(path, args)
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    if args.verbose:
                        print(f"[WARN] ffmpeg 추출 실패: {e}")
        raise RuntimeError(f"비디오 오디오 추출 실패 (backends={backend_order}): {last_err}")

    raise RuntimeError(f"지원하지 않는 확장자: {ext}")


def process_file(
    path: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> None:
    rel = path.relative_to(args.input_dir)
    base_no_ext = rel.with_suffix("")
    out_img = out_dir / base_no_ext.parent / f"{base_no_ext.name}_mel.{args.image_format}"
    out_npy = out_dir / base_no_ext.parent / f"{base_no_ext.name}_mel.npy"

    if not args.overwrite:
        img_exists = out_img.exists()
        npy_exists = out_npy.exists()
        # 스킵 조건: 이미지가 있고, (npy 저장 요청이 없거나 이미 npy도 있음)
        if img_exists and (not args.save_npy or (args.save_npy and npy_exists)):
            if args.verbose:
                print(f"[SKIP] {rel} (cached: img{'+npy' if npy_exists else ''})")
            return

    try:
        y, sr = load_media(path, args)
    except Exception as e:  # noqa: BLE001
        print(f"[ERR] load fail {rel}: {e}", file=sys.stderr)
        return

    mel_db = make_mel(
        y=y,
        sr=sr,
        n_mels=args.n_mels,
        hop_length=args.hop_length,
        n_fft=args.n_fft,
        fmin=args.fmin,
        fmax=args.fmax if args.fmax > 0 else None,
        power=args.power,
    )

    if args.save_npy:
        try:
            out_npy.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_npy, mel_db)
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] npy save 실패 {rel}: {e}")

    save_figure(
        mel_db=mel_db,
        out_img=out_img,
        sr=sr,
        hop_length=args.hop_length,
        cmap=args.cmap,
        add_colorbar=args.colorbar,
    )
    if args.verbose:
        print(f"[OK] {rel} -> {out_img.relative_to(out_dir)}")


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch generate mel spectrogram images with ms axis.")
    p.add_argument("--input_dir", required=True, type=Path, help="오디오 파일 루트 디렉토리")
    p.add_argument("--output_dir", required=True, type=Path, help="결과 저장 디렉토리 (이미지 / npy)")
    p.add_argument("--sr", type=int, default=16000, help="리샘플링 샘플레이트")
    p.add_argument("--n_fft", type=int, default=1024)
    p.add_argument("--hop_length", type=int, default=320, help="프레임 hop (ms 해상도=hop_length/sr*1000)")
    p.add_argument("--n_mels", type=int, default=256)
    p.add_argument("--fmin", type=int, default=20)
    p.add_argument("--fmax", type=int, default=8000)
    p.add_argument("--power", type=float, default=2.0)
    p.add_argument("--cmap", type=str, default="magma")
    p.add_argument("--image_format", type=str, default="png", choices=["png", "jpg", "jpeg", "webp"])
    p.add_argument("--save_npy", action="store_true", help="멜 dB 행렬을 .npy 로도 저장")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--mono", action="store_true", default=True, help="모노 변환 (기본 True)")
    p.add_argument("--no-mono", dest="mono", action="store_false", help="스테레오 유지")
    p.add_argument("--colorbar", action="store_true", help="컬러바 표시")
    p.add_argument(
        "--video-backend",
        type=str,
        default="auto",
        choices=["auto", "moviepy", "ffmpeg"],
        help="비디오 오디오 추출 백엔드 선택",
    )
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="처리 파일 수 제한 (디버그) 0=무제한")
    args = p.parse_args(argv)
    return args


def sanity_args(args: argparse.Namespace):
    if not args.input_dir.is_dir():
        raise SystemExit(f"입력 디렉토리 없음: {args.input_dir}")
    if args.limit < 0:
        raise SystemExit("--limit 는 0 또는 양수")
    if args.fmax > 0 and args.fmax <= args.fmin:
        raise SystemExit("fmax 는 fmin 보다 커야 함 (또는 fmax=0 으로 자동)")


def main(argv: List[str] | None = None):
    args = parse_args(argv or sys.argv[1:])
    sanity_args(args)

    files = list(iter_audio_files(args.input_dir))
    if args.limit:
        files = files[: args.limit]
    if not files:
        print("[INFO] 처리할 오디오 파일이 없습니다.")
        return

    print(f"[INFO] 대상 파일 수: {len(files)} (오디오+비디오)")
    print(
        f"[INFO] hop={args.hop_length} -> 시간 분해능 ≈ {args.hop_length/args.sr*1000:.1f} ms / frame"
    )

    for path in tqdm(files, desc="Mel", unit="file"):
        process_file(path, args.output_dir, args)

    print("[DONE] 생성 완료")


if __name__ == "__main__":  # pragma: no cover
    main()
