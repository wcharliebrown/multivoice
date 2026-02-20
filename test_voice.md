# Voice Test Cases

Run with:
```bash
python test_voice.py
```

Output goes to `audio_output/test_voice/`. All tests use the **MARTIN** voice (cloned from `character_samples/Martin_Van_Buren_sample.wav`).

---

## Test Cases

| File | What to listen for |
|------|-------------------|
| `01_baseline.wav` | Baseline — plain speech, no effects. Use as reference. |
| `02_sentence_pauses.wav` | Three sentences. Should have a brief (~0.3s) audible gap between each. |
| `03_emphasis_italic.wav` | The word *upgrade* should be noticeably stressed. Should NOT be spelled out as "U-P-grade". |
| `04_emphasis_bold.wav` | **Upgrade** should be strongly stressed. Also checks sentence pauses between numbered items. |
| `05_shouting.wav` | Should sound loud and forceful. |
| `06_whispering.wav` | Should sound hushed and quiet. Also tests sentence pauses in a whisper. |
| `07_beat_pause.wav` | `[beat]` should produce a ~0.7s pause in the middle of the line. |
| `08_dominoes_game.wav` | "dominoes" — the tile game. Should rhyme with "ponies". |
| `09_dominos_pizza.wav` | "Domino's" — the pizza chain. Same pronunciation, but listen for consistency. |
| `10_abbreviation.wav` | "Mr." should NOT split into two sentences. "Mr. Henderson called" should be one phrase. |

---

## Tuning

If a test sounds off, the relevant constants to adjust are in `tts_generator.py`:

| Issue | Where to fix |
|-------|-------------|
| Sentence pause too short/long | `TTSGenerator.SENTENCE_PAUSE_SECONDS` (default 0.3) |
| Beat pause too short/long | `TTSGenerator.BEAT_DURATIONS` (default: beat=0.7, long beat=1.5) |
| Emphasis not strong enough | Add more words to the instruct in `_generate_segment` |
| Shouting/whispering wrong | Edit the mapping in `EMOTION_MAPPINGS` |
| Pronunciation wrong | Edit the text in the screenplay (e.g. spell out "doh-mih-noze") |

---

## Adding More Tests

Edit `test_voice.py` and add a tuple to the `TESTS` list:

```python
("11_my_test", "MARTIN", "The text to speak.", "optional_stage_direction"),
```

To test a different character, change `"MARTIN"` to any character name in `voice_config.json`.
