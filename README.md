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

## Preparing Your Book

### 1. Export your manuscript

Export your book as rich text format (`.rtf`). If you use [Vellum](https://vellum.pub), export from there. Any word processor that can save as RTF or DOCX will work.

### 2. Convert to Markdown

```bash
brew install pandoc
pandoc Your-Book.rtf -o Your-Book.md
```

For DOCX files, use `pandoc Your-Book.docx -o Your-Book.md`.

### 3. Split into chapters using Claude Code

```
Prompt: Split the file @Your-Book.md into separate .md files, one per chapter.
         Save them in chapters/
```

### 4. Create character list using Claude Code

```
Prompt: Scan the chapter files one at a time and create a file of major and minor
         characters with descriptions characters.md
```

### 5. Create screenplay from novel format using Claude Code

```
Prompt: For each chapter file in chapters/ create a new file in chapters/screenplay/
         containing the screenplay format of the chapter to be used later when creating
         a multi-voice audiobook
```

### 6. Record or source voice samples (optional but recommended)

For each major character, record a clean 10-30 second WAV of natural speech and place it in `character_samples/` named `Character_Name_sample.wav` (spaces become underscores). Characters with samples use voice cloning for significantly better quality. Characters without samples will use AI-generated voices from text descriptions.

### 7. Configure character voices using Claude Code

```
Prompt: Read characters.md and generate a voice_config.json with detailed voice
         descriptions for each character suitable for text-to-speech synthesis
```

Edit `voice_config.json` afterward to fine-tune any descriptions.

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

## End-to-End Workflow

```
Export manuscript (.rtf)
        |
    pandoc -> .md
        |
    Split into chapters/
        |
    Create characters.md
        |
    Create chapters/screenplay/
        |
    Record voice samples -> character_samples/
        |
    Generate voice_config.json
        |
    python tts_generator.py --dry-run          # verify parsing
        |
    python tts_generator.py --chapter 1 --assemble   # generate chapter 1
        |
    Listen, edit screenplay, regenerate lines   # iterate
        |
    python tts_generator.py --assemble          # generate all chapters
```

## Tuning Your Audio

After generating a chapter, listen through and iterate:

1. **Adjust pacing in the screenplay** - Add `[beat]` for a 0.7s pause or `[long beat]` for a 1.5s pause anywhere in dialogue
2. **Fix emphasis** - Wrap words in `*asterisks*` to stress them, or remove emphasis that sounds unnatural
3. **Tweak delivery** - Add or change stage directions like `(whispering)`, `(shouting)`, `(sarcastic)` after the character name
4. **Regenerate only what changed** - Use `--line` to regenerate specific lines without reprocessing the entire chapter:
   ```bash
   python tts_generator.py --chapter 1 --line 11 --assemble
   ```
5. **Adjust overall speed** - If the pace feels rushed or slow, use `--speed` (default 0.75, lower = slower)

Individual line WAVs are saved in `audio_output/chapter_NN/` so you can also listen to them individually before assembling.

## Performance

Generation time depends on your hardware and chapter length. As a rough guide:

| Hardware | Chapter (125 lines) | Per line |
|----------|-------------------|----------|
| Apple M1 (MPS) | ~55 minutes | ~25 seconds |
| NVIDIA GPU (CUDA) | faster | varies by GPU |

The first run downloads the Qwen3-TTS models (~3.5GB). Subsequent runs use the cached models.

## Project Structure

```
.
├── tts_generator.py          # Main pipeline script
├── voice_config.json         # Character voice descriptions
├── character_samples/        # Reference WAV files for voice cloning
│   ├── Character_Name_sample.wav
│   └── ...
├── chapters/                 # Chapter markdown files (not in repo)
│   └── screenplay/           # Screenplay-formatted versions
├── audio_output/             # Generated audio (not in repo)
│   ├── chapter_01/           # Individual line WAVs
│   └── chapter_01_complete.wav
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

Good voice descriptions include: language/accent, age, gender, tone, pace, texture, and personality. The more specific the description, the more distinct the generated voice.

Characters with WAV samples in `character_samples/` use voice cloning (Base model) for higher quality. Characters without samples fall back to voice design from the text description.

## Troubleshooting

- **"flash-attn is not installed"** - This warning is harmless on Mac. Flash Attention requires NVIDIA CUDA GPUs. On Apple Silicon, the pipeline uses MPS (Metal Performance Shaders) instead.
- **"Setting pad_token_id to eos_token_id"** - Harmless warning from the transformers library during generation. Can be ignored.
- **"torch_dtype is deprecated"** - Cosmetic warning. Does not affect output.
- **bfloat16 errors on Mac** - MPS does not support bfloat16. The pipeline automatically uses float16 on Apple Silicon.
- **Out of memory** - Try closing other applications. The 1.7B model needs ~8GB RAM. If still failing, the 0.6B model uses less memory (edit `MODEL_VOICE_DESIGN` and `MODEL_VOICE_CLONE` in `tts_generator.py`).
