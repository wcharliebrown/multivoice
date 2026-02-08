#!/usr/bin/env python3
"""
CLI script for voice cloning using Qwen3-TTS.

Usage:
    python qwen3_tts_clone.py --ref-audio sample.wav --text "Hello world" --output out.wav
    python qwen3_tts_clone.py --ref-audio sample.wav --ref-text "transcript" --text "Hello" -o out.wav
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate speech using Qwen3-TTS voice cloning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (x-vector only mode, no transcript needed)
  python qwen3_tts_clone.py --ref-audio speaker.wav --text "Hello world" -o output.wav

  # With reference transcript (higher quality)
  python qwen3_tts_clone.py --ref-audio speaker.wav --ref-text "Original speech" --text "New speech" -o output.wav

  # Specify voice characteristics (gender, accent, tone)
  python qwen3_tts_clone.py --ref-audio speaker.wav --text "Hello" -o output.wav --instruct "Male, American accent"
  python qwen3_tts_clone.py --ref-audio speaker.wav --text "Hello" -o output.wav --instruct "Female, British accent, calm tone"

  # Generate multiple samples with different seeds
  python qwen3_tts_clone.py --ref-audio speaker.wav --text "Hello" -o output.wav --samples 5
        """,
    )
    parser.add_argument(
        "--ref-audio",
        "-r",
        type=str,
        required=True,
        help="Path to reference .wav file for voice cloning",
    )
    parser.add_argument(
        "--ref-text",
        "-t",
        type=str,
        default=None,
        help="Transcript of the reference audio (improves quality if provided)",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text to synthesize",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output.wav",
        help="Output .wav file path (default: output.wav)",
    )
    parser.add_argument(
        "--language",
        "-l",
        type=str,
        default="English",
        choices=[
            "Chinese",
            "English",
            "Japanese",
            "Korean",
            "German",
            "French",
            "Russian",
            "Portuguese",
            "Spanish",
            "Italian",
        ],
        help="Language for synthesis (default: English)",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="1.7B",
        choices=["0.6B", "1.7B"],
        help="Model size to use (default: 1.7B)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run on (default: cuda:0, use 'cpu' for CPU)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--no-flash-attn",
        action="store_true",
        help="Disable FlashAttention 2 (use if not supported)",
    )
    parser.add_argument(
        "--end-padding",
        type=float,
        default=0.3,
        help="Silence padding in seconds at the end to prevent clipping (default: 0.3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42, use -1 for random)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (lower=more consistent, default: 0.7)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.8,
        help="Top-p sampling (lower=more consistent, default: 0.8)",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy decoding (most consistent, ignores temperature/top-p)",
    )
    parser.add_argument(
        "--samples",
        "-n",
        type=int,
        default=1,
        help="Generate N samples with different seeds (default: 1)",
    )
    parser.add_argument(
        "--instruct",
        "-i",
        type=str,
        default=None,
        help="Voice instruction (e.g., 'Male, American accent' or 'Female, British accent, calm tone')",
    )
    return parser.parse_args()


def get_dtype(dtype_str: str) -> torch.dtype:
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_map[dtype_str]


def create_silence(duration_sec: float, sample_rate: int) -> np.ndarray:
    """Create silence array of specified duration."""
    num_samples = int(duration_sec * sample_rate)
    return np.zeros(num_samples, dtype=np.float32)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main():
    args = parse_args()

    # Validate reference audio exists
    ref_audio_path = Path(args.ref_audio)
    if not ref_audio_path.exists():
        print(f"Error: Reference audio file not found: {args.ref_audio}", file=sys.stderr)
        sys.exit(1)

    # Select model
    model_name = f"Qwen/Qwen3-TTS-12Hz-{args.model_size}-Base"
    print(f"Loading model: {model_name}")

    # Determine attention implementation
    attn_impl = "eager" if args.no_flash_attn else "flash_attention_2"

    # Handle device selection
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU", file=sys.stderr)
        device = "cpu"
        attn_impl = "eager"  # FlashAttention requires CUDA

    # Handle MPS (Apple Silicon)
    if device == "mps" or (device == "cpu" and torch.backends.mps.is_available()):
        device = "mps"
        attn_impl = "eager"  # FlashAttention not supported on MPS

    dtype = get_dtype(args.dtype)

    # Load model
    model = Qwen3TTSModel.from_pretrained(
        model_name,
        device_map=device,
        dtype=dtype,
        attn_implementation=attn_impl,
    )

    # Build generation kwargs for consistency
    gen_kwargs = {}
    if args.greedy:
        gen_kwargs["do_sample"] = False
        print("Using greedy decoding (most consistent)")
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["top_p"] = args.top_p
        print(f"Using temperature={args.temperature}, top_p={args.top_p}")

    if args.instruct:
        print(f"Voice instruction: {args.instruct}")

    print(f"Generating speech from reference: {args.ref_audio}")

    def generate_audio(seed: int) -> tuple[np.ndarray, int]:
        """Generate audio for the full text with given seed."""
        set_seed(seed)

        # Build clone kwargs
        clone_kwargs = {
            "text": args.text,
            "language": args.language,
            "ref_audio": str(ref_audio_path),
            **gen_kwargs,
        }

        if args.instruct:
            clone_kwargs["instruct"] = args.instruct

        if args.ref_text:
            clone_kwargs["ref_text"] = args.ref_text
            wavs, sr = model.generate_voice_clone(**clone_kwargs)
        else:
            clone_kwargs["ref_text"] = ""
            clone_kwargs["x_vector_only_mode"] = True
            wavs, sr = model.generate_voice_clone(**clone_kwargs)

        # Add end padding to prevent clipping
        if args.end_padding > 0:
            wav_with_padding = np.concatenate([
                wavs[0],
                create_silence(args.end_padding, sr)
            ])
            return wav_with_padding, sr

        return wavs[0], sr

    # Generate samples
    output_path = Path(args.output)
    base_seed = args.seed if args.seed >= 0 else 0

    if args.samples == 1:
        # Single sample - use original filename
        seed = base_seed if args.seed >= 0 else np.random.randint(0, 100000)
        print(f"Using seed: {seed}")
        final_audio, sr = generate_audio(seed)
        sf.write(str(output_path), final_audio, sr)
        print(f"Saved output to: {output_path}")
    else:
        # Multiple samples - add seed to filename
        stem = output_path.stem
        suffix = output_path.suffix
        parent = output_path.parent

        for sample_idx in range(args.samples):
            seed = base_seed + sample_idx
            sample_path = parent / f"{stem}_seed{seed}{suffix}"
            print(f"\n[Sample {sample_idx+1}/{args.samples}] seed={seed}")
            final_audio, sr = generate_audio(seed)
            sf.write(str(sample_path), final_audio, sr)
            print(f"Saved: {sample_path}")


if __name__ == "__main__":
    main()
