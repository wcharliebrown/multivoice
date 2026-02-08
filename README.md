# Multivoice

A screenplay-to-audiobook pipeline using [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) running locally. Parses screenplay-formatted markdown files and generates multi-voice audio with distinct character voices, emotion/delivery cues, and voice cloning from reference samples.

## Features

- **Screenplay parsing** - Reads standard screenplay format (scene headers, character names, stage directions, dialogue) from markdown files
- **Voice cloning** - Uses reference WAV samples for major characters via the Qwen3-TTS Base model
- **Voice design** - Generates voices from text descriptions for characters without samples via the VoiceDesign model
- **Emotion and delivery** - Extracts stage directions and inline cues (e.g., `(shouting)`, `(whispering)`) and passes them as TTS instructions
- **Emphasis** - Markdown `*emphasis*` and `**bold**` markers are converted to TTS emphasis instructions
- **Beat pauses** - `[beat]` and `[long beat]` tags in dialogue insert real silence gaps (0.7s and 1.5s)
- **Audio post-processing** - Leading artifact trimming, RMS loudness normalization, noise floor, speed adjustment, end padding
- **Chapter assembly** - Stitches individual line WAVs into complete chapter audio files
- **Selective regeneration** - Regenerate specific lines without re-running the entire chapter

## Requirements

- Python 3.12+
- macOS (Apple Silicon with MPS) or Linux (NVIDIA GPU with CUDA)
- ~8GB RAM for the 1.7B model

## Setup

```bash
# Create conda environment and install dependencies
bash setup_tts.sh

# Activate the environment
conda activate qwen3-tts
```

## Usage

```bash
# Process all chapters
python tts_generator.py

# Process a specific chapter and assemble into a single file
python tts_generator.py --chapter 1 --assemble

# Dry run (parse only, no audio generation)
python tts_generator.py --dry-run

# List all characters and their voice modes
python tts_generator.py --list-characters

# Regenerate specific lines
python tts_generator.py --chapter 1 --line 11
python tts_generator.py --chapter 1 --line 11,15,20
python tts_generator.py --chapter 1 --line 11-20

# Adjust playback speed (default: 0.75, where 1.0 = original TTS pace)
python tts_generator.py --chapter 1 --speed 0.8

# Regenerate a line and re-assemble the chapter
python tts_generator.py --chapter 1 --line 11 --assemble
```

## Project Structure

```
.
├── tts_generator.py          # Main pipeline script
├── voice_config.json         # Character voice descriptions
├── character_samples/        # Reference WAV files for voice cloning
│   ├── Martin_Van_Buren_sample.wav
│   ├── Angel_sample.wav
│   ├── Thurber_Mingus_sample.wav
│   ├── Agent_John_Bonneville_sample.wav
│   ├── Nanna_sample.wav
│   ├── Reggie_Hammond_sample.wav
│   ├── qwen3_tts_clone.py   # Standalone voice cloning script
│   └── batch.sh             # Batch sample generation
├── chapters/                 # Screenplay markdown files (not in repo)
│   └── screenplay/           # Screenplay-formatted versions
├── audio_output/             # Generated audio (not in repo)
├── requirements.txt
├── setup_tts.sh
└── README.md
```

## Screenplay Format

The parser expects markdown files with standard screenplay formatting:

```markdown
### INT. COFFEE SHOP - DAY

MARTIN
My name's Martin Van Buren, like the vice president.

ANGEL (cutting him off)
I'm Angel. You want to party?

MARTIN (V.O.)
She was pretty rude back then. [beat] Maybe it was my fault.

MARTIN (shouting)
I can cure *any* disease!
```

- `### LOCATION - TIME` - Scene headers
- `CHARACTER_NAME` - Character identification (all caps on its own line)
- `(stage direction)` - Delivery cues after character name
- `[direction]` - Inline delivery cues within dialogue
- `[beat]` / `[long beat]` - Explicit pause markers
- `*word*` / `**word**` - Emphasis markers

## Voice Configuration

Edit `voice_config.json` to customize character voices. Each entry maps a character name to a text description passed to the TTS model:

```json
{
  "voices": {
    "MARTIN": "American English, A middle-aged male voice with a gravelly tone...",
    "ANGEL": "American English, A bright, youthful female voice..."
  }
}
```

Characters with WAV samples in `character_samples/` use voice cloning (Base model) for higher quality. Characters without samples fall back to voice design from the text description.

## Voice Samples

To add a voice sample for a new character:

1. Place a WAV file in `character_samples/` named `Character_Name_sample.wav`
2. The filename should match the character name in the screenplay (spaces become underscores)
3. Use a clean 10-30 second recording of natural speech
