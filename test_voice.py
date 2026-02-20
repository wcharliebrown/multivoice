#!/usr/bin/env python3
"""
Quick voice test script.

Generates a small set of WAV files to test emphasis, stage directions,
sentence pauses, and tricky pronunciations. Uses the MARTIN voice (clone).

Output goes to: audio_output/test_voice/

Usage:
    python test_voice.py
    python test_voice.py --dry-run
"""

import argparse
import sys
import soundfile as sf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tts_generator import VoiceConfigManager, TTSGenerator, VOICE_CONFIG_FILE, OUTPUT_DIR

# ---------------------------------------------------------------------------
# Test cases: (label, character, text, stage_direction)
# See test_voice.md for what each test is checking.
# ---------------------------------------------------------------------------
TESTS = [
    # Baseline
    (
        "01_baseline",
        "MARTIN",
        "This is a normal line with no special treatment.",
        None,
    ),
    # Sentence pauses — three sentences should have audible gaps between them
    (
        "02_sentence_pauses",
        "MARTIN",
        "First sentence. Second sentence. Third sentence.",
        None,
    ),
    # Italic emphasis — *word* should be stressed, not spelled out
    (
        "03_emphasis_italic",
        "MARTIN",
        "I said *upgrade* your brain. Not downgrade.",
        None,
    ),
    # Bold emphasis — **word** should be strongly stressed
    (
        "04_emphasis_bold",
        "MARTIN",
        "Number One: **Upgrade** your brain. Number Two: Don't be stupid.",
        None,
    ),
    # Shouting stage direction
    (
        "05_shouting",
        "MARTIN",
        "Get out of my way! I'm coming through!",
        "shouting",
    ),
    # Whispering stage direction
    (
        "06_whispering",
        "MARTIN",
        "Don't move. Don't make a sound. They can hear us.",
        "whispering",
    ),
    # Beat pause — explicit mid-sentence gap
    (
        "07_beat_pause",
        "MARTIN",
        "I thought about it. [beat] And then I thought about it some more.",
        None,
    ),
    # Pronunciation: dominoes (the tile game)
    (
        "08_dominoes_game",
        "MARTIN",
        "I could really go for a game of dominoes right now.",
        None,
    ),
    # Pronunciation: Domino's (the pizza chain)
    (
        "09_dominos_pizza",
        "MARTIN",
        "I ordered a Domino's pizza. Extra cheese, extra sauce.",
        None,
    ),
    # Abbreviation safety — Mr. should not split into two sentences
    (
        "10_abbreviation",
        "MARTIN",
        "Mr. Henderson called. He wants his money back.",
        None,
    ),
]


def main():
    parser = argparse.ArgumentParser(description="Voice quality test script")
    parser.add_argument("--dry-run", action="store_true", help="Parse only, no audio")
    args = parser.parse_args()

    out_dir = OUTPUT_DIR / "test_voice"
    out_dir.mkdir(parents=True, exist_ok=True)

    voice_config = VoiceConfigManager(VOICE_CONFIG_FILE)
    tts = TTSGenerator(voice_config, dry_run=args.dry_run)

    print(f"\nOutput directory: {out_dir}")
    print(f"Running {len(TESTS)} test cases...\n")

    for label, character, text, stage_dir in TESTS:
        direction_info = f" ({stage_dir})" if stage_dir else ""
        print(f"[{label}] {character}{direction_info}")
        print(f"  Text: {text}")

        wav, sr = tts.generate_audio(text, character, stage_dir)

        if wav is not None:
            out_path = out_dir / f"{label}.wav"
            sf.write(str(out_path), wav, sr)
            print(f"  -> {out_path.name}")
        print()

    print("=== Test complete ===")
    print(f"Listen to files in: {out_dir}")


if __name__ == "__main__":
    main()
