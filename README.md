# Multivoice

A novel-to-screenplay-to-audiobook pipeline using [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) running locally. Parses screenplay-formatted markdown files and generates multi-voice audio with distinct character voices, emotion/delivery cues, and voice cloning from reference samples.

## Features

- **Screenplay parsing** - Reads screenplay format (character names, stage directions, dialogue) from markdown files, with or without scene headers
- **Voice cloning** - Uses reference WAV samples for major characters via the Qwen3-TTS Base model
- **Voice design** - Generates voices from text descriptions for characters without samples via the VoiceDesign model
- **Emotion and delivery** - Extracts stage directions and inline cues (e.g., `(shouting)`, `(whispering)`) and passes them as TTS instructions
- **Emphasis** - Markdown `*emphasis*` and `**bold**` markers are passed as stress instructions to the TTS
- **Beat pauses** - `[beat]` and `[long beat]` tags in dialogue insert real silence gaps (0.7s and 1.5s)
- **Sentence pauses** - Multi-sentence lines are automatically split and a 0.3s pause is inserted between sentences for natural pacing
- **Quality scoring** - Each line is scored by [UTMOSv2](https://github.com/sarulab-speech/UTMOSv2) (MOS 1-5); if the first take scores below threshold, additional takes are generated (up to 5) and the best is kept
- **Audio post-processing** - RMS loudness normalization, noise floor, end padding; direction-aware volume scaling (shouting is louder, whispering is softer) and high-pass filtering for whisper directions
- **Chapter assembly** - Stitches individual line WAVs into complete chapter audio files
- **Selective regeneration** - Regenerate specific lines without re-running the entire chapter
- **Stats logging** - Per-chapter stats file with MOS scores, take counts, and text for each line

## Preparing Your Book

### 1. Export your manuscript as ePub

Export your book as an ePub file (`.epub`). Most writing and publishing tools support this — [Vellum](https://vellum.pub), Scrivener, Apple Books Author, or any word processor with an ePub export option.

### 2. Convert ePub to Markdown chapters

```bash
pip install beautifulsoup4
python epub_to_markdown.py Your-Book.epub
```

This splits the ePub into one Markdown file per chapter in `chapters/` and generates `chapters/characters.md` with character names extracted by frequency analysis. Filenames follow the pattern `NN-Chapter-Title.md`.

### 3. Convert chapters to screenplay format using Claude Code

```
Prompt: For each chapter file in chapters/ create a new file in chapters/screenplay/
         containing the screenplay format of the chapter to be used later when creating
         a multi-voice audiobook
```

### 4. Review and expand characters.md using Claude Code

```
Prompt: Review chapters/characters.md and scan the chapter files to add descriptions,
         roles, and personality notes for each character
```

### 5. Record or source voice samples (optional but recommended)

For each major character, record a clean 10-30 second WAV of natural speech and place it in `character_samples/` named `Character_Name_sample.wav` (spaces become underscores). See `character_samples_examples/README.md` for naming conventions and requirements. Characters with samples use voice cloning for significantly better quality. Characters without samples will use AI-generated voices from text descriptions.

### 6. Configure character voices using Claude Code

Copy the example config as a starting point:

```bash
cp voice_config_example.json voice_config.json
```

Then use Claude Code to populate it from your character list:

```
Prompt: Read characters/characters.md and generate a voice_config.json with detailed
         voice descriptions for each character suitable for text-to-speech synthesis
```

Edit `voice_config.json` afterward to fine-tune any descriptions. This file is kept out of the repo (via `.gitignore`) since it contains project-specific character data.

## Requirements

- Python 3.12+
- macOS (Apple Silicon with MPS) or Linux (NVIDIA GPU with CUDA)
- ~8GB RAM for the TTS models
- Claude Code account. Currently $20/mo Used for the conversion from manuscript to screenplay and extraction of character descriptions.

## Setup

```bash
# Create conda environment and install dependencies
bash setup_tts.sh

# Activate the environment
conda activate qwen3-tts

# Install UTMOSv2 for quality scoring
pip install git+https://github.com/sarulab-speech/UTMOSv2.git
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

# Regenerate a line and re-assemble the chapter
python tts_generator.py --chapter 1 --line 11 --assemble
```

## End-to-End Workflow

```
Export manuscript (.epub)
        |
    python epub_to_markdown.py book.epub   # splits into chapters/ + characters.md
        |
    Claude Code: convert chapters to screenplay format in chapters/screenplay/
        |
    Claude Code: expand characters.md with descriptions
        |
    Record voice samples -> character_samples/   (optional)
        |
    Claude Code: generate voice_config.json from characters.md
        |
    python tts_generator.py --dry-run          # verify parsing
        |
    python tts_generator.py --chapter 1 --assemble   # generate chapter 1
        |
    Review stats file + listen, edit screenplay   # iterate
        |
    python tts_generator.py --chapter 1 --line 11 --assemble  # fix lines
        |
    python tts_generator.py --assemble          # generate all chapters
```

## Quality Scoring

Each line is generated and immediately scored using [UTMOSv2](https://github.com/sarulab-speech/UTMOSv2), which predicts a Mean Opinion Score (MOS) from 1.0 to 5.0. The scoring works as follows:

1. Generate take 1, score it
2. If MOS >= 3.5 (threshold), keep it and move on
3. If MOS < 3.5, generate another take (up to 5 max)
4. Keep the highest-scoring take

The threshold (`MOS_THRESHOLD`) and maximum takes (`MAX_TAKES`) are class constants in `TTSGenerator` that can be adjusted.

After each chapter, a stats file is written to `audio_output/chapter_NN/chNN_stats.txt`:

```
Chapter 1: Born Yesterday
Line   Character                 Takes  Best   Scores                                   Text
------------------------------------------------------------------------------------------------------------------------
1      MARTIN (V.O.)             1      3.82   [3.82]                                   Chapter one. Born Yesterday.
2      MARTIN                    3      3.67   [3.21, 3.45, 3.67]                       I'm a super-intelligent robot from the
3      MARTIN (V.O.)             1      4.01   [4.01]                                   I said. I had a homeless guy cornered
```

Use this to identify lines that may need re-recording with `--line`.

## Tuning Your Audio

After generating a chapter, review the stats file and listen through:

1. **Check the stats file** - Look for lines with low MOS scores or lines that needed many takes; these are candidates for re-recording or screenplay edits
2. **Adjust pacing in the screenplay** - Add `[beat]` for a 0.7s pause or `[long beat]` for a 1.5s pause anywhere in dialogue; multi-sentence lines automatically get a 0.3s inter-sentence pause (adjustable via `SENTENCE_PAUSE_SECONDS` in `TTSGenerator`)
3. **Fix emphasis** - Wrap words in `*asterisks*` to stress them, or remove emphasis that sounds unnatural
4. **Tweak delivery** - Add or change stage directions like `(whispering)`, `(shouting)`, `(sarcastic)` after the character name
5. **Regenerate only what changed** - Use `--line` to regenerate specific lines without reprocessing the entire chapter:
   ```bash
   python tts_generator.py --chapter 1 --line 11 --assemble
   ```

Individual line WAVs are saved in `audio_output/chapter_NN/` so you can also listen to them individually before assembling.

## Performance

Generation time depends on your hardware, chapter length, and how many retakes are needed. As a rough guide:

| Hardware | Per line (1 take) | Per line (avg with retakes) |
|----------|-------------------|---------------------------|
| Apple M1 (MPS) | ~25 seconds | ~40-60 seconds |
| NVIDIA GPU (CUDA) | faster | varies by GPU |

The first run downloads the Qwen3-TTS models (~3.5GB) and UTMOSv2 weights. Subsequent runs use cached models.

## Project Structure

```
.
├── tts_generator.py          # Main pipeline script
├── epub_to_markdown.py       # ePub → per-chapter Markdown converter
├── voice_config.json         # Character voice descriptions (not in repo)
├── voice_config_example.json # Example voice config template
├── character_samples/        # Reference WAV files for voice cloning (not in repo)
│   ├── Character_Name_sample.wav
│   └── ...
├── character_samples_examples/  # Documentation for voice samples
├── chapters/                 # Chapter markdown files (not in repo)
│   └── screenplay/           # Screenplay-formatted versions
├── audio_output/             # Generated audio (not in repo)
│   ├── chapter_01/           # Individual line WAVs
│   ├── chapter_01/ch01_stats.txt  # Quality scores per line
│   └── chapter_01_complete.wav
├── requirements.txt
├── setup_tts.sh
└── README.md
```

## Screenplay Format

The parser reads markdown files with screenplay formatting. Scene headers are optional — files without them are treated as a single scene.

```markdown
MARTIN
My name's Martin Van Buren, like the vice president.

ANGEL (cutting him off)
I'm Angel. You want to party?

MARTIN (V.O.)
She was pretty rude back then. [beat] Maybe it was my fault.

MARTIN (shouting)
I can cure *any* disease!
```

- `CHARACTER_NAME` - Character identification (all caps on its own line)
- `(stage direction)` - Delivery cues after character name
- `[direction]` - Inline delivery cues within dialogue
- `[beat]` / `[long beat]` - Explicit pause markers (0.7s / 1.5s of silence)
- `*word*` / `**word**` - Emphasis markers (passed as stress instructions to TTS)
- `---` - Section dividers (ignored by parser)
- `### SCENE N: LOCATION - TIME` - Optional scene headers

## Voice Configuration

Copy `voice_config_example.json` to `voice_config.json` and edit it to customize character voices. Each entry maps a character name to a text description passed to the TTS model:

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
- **TTS hangs on a line** - Usually caused by trailing ellipsis (`...`) in the screenplay text. The preprocessor converts these to periods, but if you hit a hang, check the text for unusual punctuation.
- **Out of memory** - Try closing other applications. The models need ~8GB RAM. If still failing, the 0.6B model uses less memory (edit `MODEL_VOICE_DESIGN` and `MODEL_VOICE_CLONE` in `tts_generator.py`).
