# Character Samples Examples

Place voice reference WAV files in `character_samples/` (not this folder) for voice cloning.

## Naming Convention

Files must be named `Character_Name_sample.wav`, where the character name matches the screenplay (spaces become underscores):

```
character_samples/
├── John_Smith_sample.wav
├── Jane_Doe_sample.wav
└── Narrator_sample.wav
```

## Requirements

- Format: WAV (mono or stereo, any sample rate)
- Length: 10-30 seconds of clear speech
- Content: Natural conversational speech in the character's voice
- Quality: Clean recording, minimal background noise

## How It Works

Characters with a matching WAV file in `character_samples/` use **voice cloning** via the Qwen3-TTS Base model, which produces higher quality results. Characters without a sample fall back to **voice design**, where the voice is generated from the text description in `voice_config.json`.

## Example Mapping

If your screenplay has:

```
DETECTIVE HARRIS (shouting)
Stop right there!
```

Then name the file: `Detective_Harris_sample.wav`
