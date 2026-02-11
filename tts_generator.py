#!/usr/bin/env python3
"""
Temporal Fuse Multi-Voice TTS Generator

Processes screenplay-formatted markdown files and generates multi-voice
audiobook audio files using Qwen3-TTS locally.

Usage:
    python tts_generator.py                     # Process all chapters
    python tts_generator.py --chapter 1         # Process specific chapter
    python tts_generator.py --list-characters   # List all characters found
    python tts_generator.py --dry-run           # Parse without generating audio
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# TTS dependencies are imported lazily to allow dry-run without them
torch = None
sf = None
Qwen3TTSModel = None


def _load_tts_dependencies():
    """Lazily load TTS dependencies."""
    global torch, sf, Qwen3TTSModel
    if torch is None:
        import torch as _torch
        import soundfile as _sf
        from qwen_tts import Qwen3TTSModel as _Qwen3TTSModel
        torch = _torch
        sf = _sf
        Qwen3TTSModel = _Qwen3TTSModel


# =============================================================================
# Configuration
# =============================================================================

SCREENPLAY_DIR = Path(__file__).parent / "chapters" / "screenplay"
OUTPUT_DIR = Path(__file__).parent / "audio_output"
VOICE_CONFIG_FILE = Path(__file__).parent / "voice_config.json"
SAMPLE_DIR = Path(__file__).parent / "character_samples"

# Maps screenplay character names to voice sample filenames in SAMPLE_DIR.
# Characters with samples use the Base model (voice cloning) for higher quality.
# Characters without samples fall back to the VoiceDesign model.
CHARACTER_SAMPLE_MAP = {
    "MARTIN": "Martin_Van_Buren_sample.wav",
    "MARTIN (V.O.)": "Martin_Van_Buren_sample.wav",
    "NARRATOR": "Martin_Van_Buren_sample.wav",
    "ANGEL": "Angel_sample.wav",
    "NANNA": "Nanna_sample.wav",
    "THURBER": "Thurber_Mingus_sample.wav",
    "THURBER MINGUS": "Thurber_Mingus_sample.wav",
    "AGENT BONNEVILLE": "Agent_John_Bonneville_sample.wav",
    "BONNEVILLE": "Agent_John_Bonneville_sample.wav",
    "REGGIE": "Reggie_Hammond_sample.wav",
    "REGGIE HAMMOND": "Reggie_Hammond_sample.wav",
    "ORIGINAL REGGIE": "Reggie_Hammond_sample.wav",
    "PA SYSTEM (MARTIN'S VOICE)": "Martin_Van_Buren_sample.wav",
    "PA SYSTEM (REGGIE'S VOICE)": "Reggie_Hammond_sample.wav",
}

# Model names
MODEL_VOICE_DESIGN = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
MODEL_VOICE_CLONE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

# Default voice descriptions for Qwen3-TTS VoiceDesign model
# These are natural language descriptions used to synthesize character voices
DEFAULT_VOICE_DESCRIPTIONS = {
    "MARTIN": "A middle-aged male voice with a distinctive gravelly, raspy quality similar to Danny DeVito. Warm but sardonic tone, slightly nasal, with occasional comedic timing in delivery.",
    "MARTIN (V.O.)": "A middle-aged male voice with a distinctive gravelly, raspy quality similar to Danny DeVito. Warm but sardonic tone for narration, slightly nasal, reflective and introspective quality.",
    "ANGEL": "A young adult female voice in her twenties. Street-smart and tough with an edge, but capable of warmth. American accent, slightly husky, quick-tempered delivery.",
    "NANNA": "An elderly woman's voice, warm and grandmotherly. Gentle with an infectious, childlike giggle. Slightly frail but spirited.",
    "NARRATOR": "A neutral, professional male narrator voice. Clear, measured delivery suitable for chapter summaries and front/back matter.",
    "HOMELESS GUY": "A tired, world-weary older male voice. Slow, mumbling delivery suggesting exhaustion and disinterest.",
    "THOMAS": "A cold, artificial-sounding male voice with precise diction. Condescending and manipulative tone.",
    "THURBER": "A sophisticated male voice with a British accent. Intelligent, measured, and slightly menacing undertone.",
    "REGGIE": "A warm, patient male voice with calm demeanor. Thoughtful and genuinely concerned, friendly like a helpful assistant.",
    "KENT": "A young adult male voice, enthusiastic and nerdy. American accent, eager and somewhat naive.",
    "JAMES DIXON": "A young adult male voice, quieter and more reserved than Kent. Thoughtful and observant.",
    "ANITA": "A female AI voice, clear and precise with slight artificial quality. Quick and intelligent delivery.",
    "HUDSON": "A male AI voice, neutral and measured. Calm and analytical.",
    "M": "A young adult female voice with enhanced confidence. Adventurous, self-reliant, curious intonation.",
    "AGENT BONNEVILLE": "A middle-aged male FBI agent voice. Professional, persistent, slightly gruff American accent.",
    "BRAD MURPHY": "A strong, masculine farmer's voice. Midwestern American accent, straightforward and caring.",
    "TEMPLETON": "A middle-aged male professional voice. Slightly officious, measured bureaucratic tone.",
    "QUAN": "A young boy's voice around ten years old. Curious and excited, with slight Chinese accent.",
}

# Emotion and delivery mappings for stage directions
# These are combined with character voice descriptions when stage directions are present
EMOTION_MAPPINGS = {
    # Loudness
    "shouting": "Speak loudly and forcefully, almost yelling.",
    "yelling": "Speak very loudly, yelling with intensity.",
    "loudly": "Speak in a loud, raised voice.",
    "whispering": "Speak in a soft, hushed whisper.",
    "whispered": "Speak in a soft, hushed whisper.",
    "quietly": "Speak softly and quietly.",
    "muttering": "Speak in a low, mumbled tone under your breath.",
    "murmuring": "Speak softly in a low, gentle murmur.",
    "hissing": "Speak in a sharp, forceful whisper with sibilant emphasis.",

    # Emotions - Positive
    "laughing": "Speak while laughing, with joy and amusement in the voice.",
    "giggling": "Speak with a light, playful giggle.",
    "chuckling": "Speak with a warm, amused chuckle.",
    "smiling": "Speak with a warm smile evident in the voice.",
    "grinning": "Speak with a broad, mischievous grin in the voice.",
    "happily": "Speak with happiness and joy.",
    "excitedly": "Speak with excitement and enthusiasm.",
    "cheerfully": "Speak in a cheerful, upbeat manner.",
    "enthusiastically": "Speak with great enthusiasm and energy.",

    # Emotions - Negative
    "angry": "Speak with anger and frustration in the voice.",
    "angrily": "Speak with intense anger.",
    "furious": "Speak with explosive fury and rage.",
    "frustrated": "Speak with evident frustration and annoyance.",
    "annoyed": "Speak with irritation and annoyance.",
    "growling": "Speak with a low, threatening growl.",
    "snarling": "Speak with an aggressive snarl.",
    "bitter": "Speak with bitterness and resentment.",

    # Emotions - Fear/Anxiety
    "scared": "Speak with fear and trembling in the voice.",
    "frightened": "Speak with fear, voice shaking slightly.",
    "terrified": "Speak with extreme terror, voice trembling.",
    "nervous": "Speak nervously, with slight hesitation.",
    "anxious": "Speak with anxiety and worry.",
    "panicking": "Speak in a panicked, breathless rush.",
    "worried": "Speak with concern and worry.",

    # Emotions - Sad
    "sad": "Speak with sadness and melancholy.",
    "sadly": "Speak with deep sadness in the voice.",
    "crying": "Speak while crying, voice choked with tears.",
    "sobbing": "Speak through sobs, voice breaking.",
    "tearfully": "Speak with tears in the voice.",
    "mournfully": "Speak with grief and mourning.",
    "sighing": "Speak with a weary, resigned sigh.",

    # Emotions - Surprise
    "surprised": "Speak with surprise and astonishment.",
    "shocked": "Speak with shock and disbelief.",
    "amazed": "Speak with wonder and amazement.",
    "incredulous": "Speak with disbelief and incredulity.",
    "stunned": "Speak as if stunned, momentarily at a loss.",

    # Delivery Style
    "sarcastically": "Speak with heavy sarcasm and irony.",
    "mockingly": "Speak in a mocking, derisive tone.",
    "smugly": "Speak with smug self-satisfaction.",
    "condescendingly": "Speak in a condescending, patronizing manner.",
    "dismissively": "Speak dismissively, as if uninterested.",

    "seriously": "Speak in a serious, earnest tone.",
    "firmly": "Speak firmly and decisively.",
    "sternly": "Speak in a stern, authoritative manner.",
    "demanding": "Speak demandingly, expecting compliance.",
    "commanding": "Speak with authority and command.",

    "gently": "Speak gently and softly.",
    "tenderly": "Speak with tenderness and affection.",
    "soothingly": "Speak in a soothing, calming manner.",
    "reassuringly": "Speak reassuringly, offering comfort.",

    "hesitantly": "Speak with hesitation and uncertainty.",
    "nervously": "Speak nervously, with slight tremor.",
    "uncertainly": "Speak with uncertainty and doubt.",
    "confused": "Speak with confusion, as if puzzled.",
    "bewildered": "Speak with bewilderment and confusion.",

    "quickly": "Speak rapidly and hurriedly.",
    "slowly": "Speak slowly and deliberately.",
    "urgently": "Speak with urgency and importance.",
    "breathlessly": "Speak breathlessly, as if out of breath.",

    "thoughtfully": "Speak thoughtfully, as if considering carefully.",
    "dreamily": "Speak in a dreamy, distant manner.",
    "wistfully": "Speak with wistful longing.",

    "dramatically": "Speak with theatrical drama.",
    "deadpan": "Speak in a flat, emotionless deadpan.",
    "dryly": "Speak in a dry, understated manner.",
    "meekly": "Speak meekly and submissively, voice small.",
    "feebly": "Speak feebly and weakly.",
    "gritted teeth": "Speak through gritted teeth, with suppressed anger.",
    "through gritted teeth": "Speak through gritted teeth, with restrained fury.",
    "winking": "Speak with a playful, knowing tone as if winking.",
    "calling": "Speak loudly as if calling out to someone.",
    "calling out": "Speak loudly, calling out to someone in the distance.",
    "preacher voice": "Speak in a theatrical, evangelical preacher style.",
    "southern preacher": "Speak in a theatrical Southern evangelical preacher style.",

    # Common stage direction phrases
    "too quickly": "Speak too quickly, rushing the words.",
    "too loudly": "Speak too loudly, almost shouting.",
    "cutting him off": "Speak abruptly, interrupting.",
    "cutting her off": "Speak abruptly, interrupting.",
    "interrupting": "Speak while interrupting, cutting in.",
    "trailing off": "Let the voice trail off at the end.",
    "under breath": "Mutter quietly under your breath.",
    "to himself": "Speak quietly to oneself.",
    "to herself": "Speak quietly to oneself.",
    "groaning": "Speak with a pained groan.",
    "moaning": "Speak with a moan of discomfort.",
    "snorting": "Speak with a derisive snort.",
    "scoffing": "Speak with a dismissive scoff.",
    "pleading": "Speak pleadingly, begging.",
    "begging": "Speak desperately, begging.",
    "threatening": "Speak with menace and threat.",
    "teasing": "Speak in a playful, teasing manner.",
    "flirting": "Speak flirtatiously.",
    "seductively": "Speak in a seductive, alluring manner.",
    "impatient": "Speak with impatience.",
    "impatiently": "Speak impatiently, rushing.",
    "exasperated": "Speak with exasperation.",
    "resigned": "Speak with weary resignation.",
    "defeated": "Speak as if defeated, voice low.",
    "triumphant": "Speak triumphantly, with victory.",
    "proud": "Speak with pride.",
    "sheepish": "Speak sheepishly, embarrassed.",
    "embarrassed": "Speak with embarrassment.",
    "apologetic": "Speak apologetically.",
    "defensive": "Speak defensively.",
    "accusatory": "Speak accusingly.",
    "suspicious": "Speak with suspicion.",
    "curious": "Speak with curiosity.",
    "excited": "Speak with excitement.",
    "bored": "Speak with boredom, disinterested.",
    "tired": "Speak tiredly, with fatigue.",
    "exhausted": "Speak with exhaustion, voice weak.",
    "weak": "Speak weakly, voice fading.",
    "hoarse": "Speak in a hoarse, strained voice.",
    "raspy": "Speak in a raspy voice.",
    "croaking": "Speak in a croaky voice.",
}

# Model configuration


def _get_device():
    """Get the best available device."""
    _load_tts_dependencies()
    if torch.cuda.is_available():
        return "cuda:0"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_dtype():
    """Get the appropriate dtype for the device."""
    _load_tts_dependencies()
    if torch.cuda.is_available():
        return torch.bfloat16
    elif torch.backends.mps.is_available():
        # MPS supports float16 but not bfloat16
        return torch.float16
    return torch.float32


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DialogueLine:
    """Represents a single line of dialogue or narration."""
    character: str
    text: str
    stage_direction: Optional[str] = None
    line_number: int = 0


@dataclass
class Scene:
    """Represents a scene in the screenplay."""
    number: int
    location: str
    time_of_day: str
    lines: list[DialogueLine] = field(default_factory=list)


@dataclass
class Chapter:
    """Represents a complete chapter."""
    number: int
    title: str
    scenes: list[Scene] = field(default_factory=list)
    summary: Optional[str] = None


# =============================================================================
# Screenplay Parser
# =============================================================================

class ScreenplayParser:
    """Parses screenplay-formatted markdown files."""

    # Regex patterns for parsing
    CHAPTER_HEADER = re.compile(r'^#\s+Chapter\s+(\d+):\s+(.+)$', re.IGNORECASE)
    SCENE_HEADER = re.compile(r'^###\s+SCENE\s+(\d+):\s+(.+)$', re.IGNORECASE)
    CHARACTER_LINE = re.compile(r'^([A-Z][A-Z\s\-_]+(?:\s*\([^)]+\))?)$')
    STAGE_DIRECTION = re.compile(r'^\(([^)]+)\)$')
    CHAPTER_END = re.compile(r'^\*\*END OF CHAPTER', re.IGNORECASE)
    CHAPTER_SUMMARY = re.compile(r'^###\s+CHAPTER SUMMARY', re.IGNORECASE)

    # Known time-of-day values for scene headers
    TIME_OF_DAY_VALUES = {'DAY', 'NIGHT', 'CONTINUOUS', 'MORNING', 'EVENING', 'AFTERNOON', 'DAWN', 'DUSK', 'LATER'}

    @classmethod
    def _parse_scene_location(cls, scene_text: str) -> tuple[str, str]:
        """Parse scene text into location and time of day."""
        # Split by " - " and check if last segment is a time of day
        parts = scene_text.split(' - ')
        if len(parts) > 1 and parts[-1].upper() in cls.TIME_OF_DAY_VALUES:
            return ' - '.join(parts[:-1]), parts[-1]
        return scene_text, ""

    def parse_file(self, filepath: Path) -> Chapter:
        """Parse a screenplay markdown file into a Chapter object."""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        lines = content.split('\n')
        chapter = Chapter(number=0, title="Unknown")
        current_scene = None
        current_character = None
        current_text_lines = []
        current_stage_direction = None
        in_summary = False
        line_counter = 0

        def flush_dialogue():
            """Save accumulated dialogue to current scene."""
            nonlocal current_character, current_text_lines, current_stage_direction, line_counter
            if current_character and current_text_lines and current_scene:
                text = ' '.join(current_text_lines).strip()
                if text:
                    current_scene.lines.append(DialogueLine(
                        character=current_character,
                        text=text,
                        stage_direction=current_stage_direction,
                        line_number=line_counter
                    ))
                    line_counter += 1
            current_text_lines = []
            current_stage_direction = None

        for line in lines:
            line = line.rstrip()

            # Skip empty lines and separators
            if not line or line == '---':
                continue

            # Check for chapter header
            match = self.CHAPTER_HEADER.match(line)
            if match:
                chapter.number = int(match.group(1))
                chapter.title = match.group(2).strip()
                continue

            # Check for chapter end marker
            if self.CHAPTER_END.match(line):
                flush_dialogue()
                continue

            # Check for chapter summary section
            if self.CHAPTER_SUMMARY.match(line):
                flush_dialogue()
                in_summary = True
                continue

            # Check for scene header
            match = self.SCENE_HEADER.match(line)
            if match:
                flush_dialogue()
                scene_text = match.group(2).strip()
                location, time_of_day = self._parse_scene_location(scene_text)
                current_scene = Scene(
                    number=int(match.group(1)),
                    location=location,
                    time_of_day=time_of_day
                )
                chapter.scenes.append(current_scene)
                current_character = None
                in_summary = False
                continue

            # Skip screenplay format subtitle
            if line.startswith('## Screenplay Format'):
                continue

            # Check for character name
            match = self.CHARACTER_LINE.match(line)
            if match and not line.startswith('**'):
                flush_dialogue()
                current_character = match.group(1).strip()
                continue

            # Check for stage direction
            match = self.STAGE_DIRECTION.match(line)
            if match:
                direction = match.group(1).strip()
                # If we have accumulated text, this is a mid-dialogue direction
                if current_text_lines:
                    # Append direction info to text for context
                    current_text_lines.append(f"[{direction}]")
                else:
                    # This is a pre-dialogue stage direction
                    current_stage_direction = direction
                continue

            # Accumulate dialogue text
            if current_character:
                # Keep markdown emphasis markers (*word*, **word**) intact;
                # TextPreprocessor will convert them to UPPERCASE for TTS emphasis
                clean_line = line.strip()
                if clean_line:
                    current_text_lines.append(clean_line)

        # Flush any remaining dialogue
        flush_dialogue()

        return chapter

    def get_all_characters(self, chapters: list[Chapter]) -> set[str]:
        """Extract all unique character names from chapters."""
        characters = set()
        for chapter in chapters:
            for scene in chapter.scenes:
                for line in scene.lines:
                    characters.add(line.character)
        return characters


# =============================================================================
# Voice Configuration Manager
# =============================================================================

class VoiceConfigManager:
    """Manages voice configurations for characters."""

    def __init__(self, config_file: Path):
        self.config_file = config_file
        self.voice_descriptions = DEFAULT_VOICE_DESCRIPTIONS.copy()
        self._load_config()

    def _load_config(self):
        """Load custom voice configurations if available."""
        if self.config_file.exists():
            with open(self.config_file, 'r', encoding='utf-8') as f:
                custom_config = json.load(f)
                self.voice_descriptions.update(custom_config.get('voices', {}))

    def save_config(self):
        """Save current voice configurations."""
        config = {'voices': self.voice_descriptions}
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

    def get_voice_description(self, character: str) -> str:
        """Get voice description for a character."""
        # Try exact match first
        if character in self.voice_descriptions:
            return self.voice_descriptions[character]

        # Try base character name (without V.O. or other modifiers)
        base_name = re.sub(r'\s*\([^)]+\)', '', character).strip()
        if base_name in self.voice_descriptions:
            return self.voice_descriptions[base_name]

        # Return a generic description
        return f"A clear, natural speaking voice appropriate for the character {character}."

    def add_character(self, character: str, description: str):
        """Add or update a character's voice description."""
        self.voice_descriptions[character] = description

    def get_voice_sample(self, character: str) -> Optional[Path]:
        """Get the voice sample WAV path for a character, if one exists."""
        # Try exact match
        if character in CHARACTER_SAMPLE_MAP:
            sample_path = SAMPLE_DIR / CHARACTER_SAMPLE_MAP[character]
            if sample_path.exists():
                return sample_path

        # Try base character name (without V.O. or other modifiers)
        base_name = re.sub(r'\s*\([^)]+\)', '', character).strip()
        if base_name in CHARACTER_SAMPLE_MAP:
            sample_path = SAMPLE_DIR / CHARACTER_SAMPLE_MAP[base_name]
            if sample_path.exists():
                return sample_path

        return None

    def has_voice_sample(self, character: str) -> bool:
        """Check whether a character has a voice sample available."""
        return self.get_voice_sample(character) is not None


# =============================================================================
# Emotion Parser
# =============================================================================

class EmotionParser:
    """Parses stage directions to extract emotion and delivery instructions."""

    def __init__(self):
        self.emotion_mappings = EMOTION_MAPPINGS

    def parse_stage_direction(self, direction: Optional[str]) -> str:
        """Parse a stage direction and return emotion/delivery instruction."""
        if not direction:
            return ""

        direction_lower = direction.lower().strip()
        instructions = []

        # Check for exact matches first
        if direction_lower in self.emotion_mappings:
            return self.emotion_mappings[direction_lower]

        # Check for partial matches (direction contains keyword)
        for keyword, instruction in self.emotion_mappings.items():
            if keyword in direction_lower:
                instructions.append(instruction)

        # If we found matches, combine them
        if instructions:
            # Remove duplicates while preserving order
            seen = set()
            unique_instructions = []
            for inst in instructions:
                if inst not in seen:
                    seen.add(inst)
                    unique_instructions.append(inst)
            return " ".join(unique_instructions)

        # If no matches, try to create a natural instruction from the direction
        # Filter out purely physical actions that don't affect voice
        physical_actions = [
            'looking', 'turning', 'walking', 'sitting', 'standing', 'pointing',
            'gesturing', 'shrugging', 'nodding', 'shaking', 'waving', 'reaching',
            'grabbing', 'holding', 'putting', 'taking', 'picking', 'opening',
            'closing', 'reading', 'writing', 'typing', 'watching', 'glancing',
            'staring', 'examining', 'checking', 'working', 'wiping', 'rubbing',
            'scratching', 'adjusting', 'fixing', 'moving', 'stepping', 'leaning',
            'bending', 'kneeling', 'crouching', 'jumping', 'running', 'eating',
            'drinking', 'chewing', 'swallowing', 'biting', 'licking'
        ]

        # Check if direction is primarily physical
        is_physical = any(action in direction_lower for action in physical_actions)

        if not is_physical and len(direction) < 100:
            # Return a contextual instruction based on the direction
            return f"Speak as if {direction_lower}."

        return ""

    def parse_inline_directions(self, text: str) -> tuple[str, list[str]]:
        """Extract inline stage directions from text and return clean text + instructions.

        Inline directions are marked as [direction] in the text.
        """
        instructions = []
        clean_text = text

        # Find all [direction] markers
        pattern = r'\[([^\]]+)\]'
        matches = re.findall(pattern, text)

        for match in matches:
            instruction = self.parse_stage_direction(match)
            if instruction:
                instructions.append(instruction)

        # Remove the markers from text
        clean_text = re.sub(pattern, '', text).strip()
        # Clean up extra spaces
        clean_text = re.sub(r'\s+', ' ', clean_text)

        return clean_text, instructions

    def build_full_instruction(self, base_voice: str, stage_direction: Optional[str],
                                inline_directions: list[str]) -> str:
        """Build full instruction combining voice description with emotions."""
        parts = [base_voice]

        # Add stage direction emotion if present
        if stage_direction:
            direction_instruction = self.parse_stage_direction(stage_direction)
            if direction_instruction:
                parts.append(direction_instruction)

        # Add inline direction emotions
        for instruction in inline_directions:
            if instruction and instruction not in parts:
                parts.append(instruction)

        return " ".join(parts)


# =============================================================================
# Text Preprocessor
# =============================================================================

class TextPreprocessor:
    """Cleans and adjusts text before sending to TTS for better output quality."""

    def process(self, text: str) -> tuple[str, list[str]]:
        """Apply all text preprocessing steps. Returns (cleaned_text, emphasized_words)."""
        text = self._fix_dashes(text)
        text, emphasized = self._extract_emphasis(text)
        text = self._expand_pauses(text)
        text = self._clean_whitespace(text)
        return text, emphasized

    def _fix_dashes(self, text: str) -> str:
        """Replace dashes/hyphens used as pauses so they aren't read as 'minus'."""
        # Em-dash (—) to comma-pause
        text = text.replace('—', ', ')
        # Spaced dash " - " to comma-pause (but not inside hyphenated words)
        text = re.sub(r'\s+-\s+', ', ', text)
        # Double dash "--" to comma-pause
        text = text.replace('--', ', ')
        return text

    def _extract_emphasis(self, text: str) -> tuple[str, list[str]]:
        """Strip emphasis markers and collect emphasized words for instruction."""
        emphasized = []
        # Bold **word** -> plain word, collect it
        def collect_bold(m):
            emphasized.append(m.group(1))
            return m.group(1)
        text = re.sub(r'\*\*([^*]+)\*\*', collect_bold, text)
        # Italic *word* -> plain word, collect it
        def collect_italic(m):
            emphasized.append(m.group(1))
            return m.group(1)
        text = re.sub(r'\*([^*]+)\*', collect_italic, text)
        return text, emphasized

    def _expand_pauses(self, text: str) -> str:
        """Expand punctuation pause markers for longer TTS pauses."""
        # Ellipsis -> longer pause marker
        text = text.replace('...', ' ... ')
        text = text.replace('…', ' ... ')
        return text

    def _clean_whitespace(self, text: str) -> str:
        """Normalize whitespace after other transformations."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+([,.\?!;:])', r'\1', text)
        return text.strip()


# =============================================================================
# Audio Post-Processor
# =============================================================================

class AudioPostProcessor:
    """Post-processes generated audio to fix common TTS artifacts."""

    # Configurable thresholds
    LEADING_TRIM_THRESHOLD = 0.02   # RMS threshold for detecting speech start
    LEADING_TRIM_WINDOW = 480       # Window size in samples (20ms at 24kHz)
    LEADING_TRIM_MAX_MS = 150       # Max ms to search for artifact (don't trim real speech)
    TARGET_RMS = 0.08               # Target RMS for loudness normalization
    NOISE_FLOOR_AMPLITUDE = 0.0003  # Nearly inaudible noise floor
    END_PADDING_SECONDS = 0.15      # Silence at end to prevent clipping

    def process(self, wav, sr: int):
        """Apply all audio post-processing steps."""
        import numpy as np

        wav = np.array(wav, dtype=np.float32)

        wav = self._normalize_loudness(wav)
        wav = self._add_noise_floor(wav, sr)
        wav = self._add_end_padding(wav, sr)

        return wav

    def _trim_leading_artifact(self, wav, sr: int):
        """Remove the initial 'burp'/artifact before real speech begins."""
        import numpy as np

        max_samples = int(self.LEADING_TRIM_MAX_MS / 1000.0 * sr)
        window = self.LEADING_TRIM_WINDOW

        # Walk through the beginning in small windows
        # Find first window that's silent AFTER initial energy (the artifact)
        # Then find where real speech starts after that
        if len(wav) < window * 3:
            return wav

        # Compute short-time energy for the beginning
        num_windows = min(max_samples // window, len(wav) // window)
        energies = []
        for i in range(num_windows):
            chunk = wav[i * window:(i + 1) * window]
            rms = np.sqrt(np.mean(chunk ** 2))
            energies.append(rms)

        if not energies:
            return wav

        # Strategy: find the first sustained silence gap in the initial portion,
        # then trim everything before the speech that follows it.
        # This removes the initial burst (artifact) before real speech.
        found_energy = False
        found_gap = False
        trim_point = 0

        for i, rms in enumerate(energies):
            if not found_energy and rms > self.LEADING_TRIM_THRESHOLD:
                found_energy = True
            elif found_energy and not found_gap and rms < self.LEADING_TRIM_THRESHOLD * 0.5:
                found_gap = True
            elif found_gap and rms > self.LEADING_TRIM_THRESHOLD:
                # Real speech starts here - trim everything before
                trim_point = i * window
                # Back up slightly to not clip the attack
                trim_point = max(0, trim_point - window // 2)
                break

        if trim_point > 0:
            return wav[trim_point:]

        return wav

    def _normalize_loudness(self, wav):
        """Normalize audio to consistent RMS loudness."""
        import numpy as np

        rms = np.sqrt(np.mean(wav ** 2))
        if rms > 0:
            wav = wav * (self.TARGET_RMS / rms)
            # Clip to prevent distortion
            wav = np.clip(wav, -1.0, 1.0)

        return wav

    def _add_noise_floor(self, wav, sr: int):
        """Add nearly inaudible noise floor to avoid dead silence."""
        import numpy as np

        noise = np.random.normal(0, self.NOISE_FLOOR_AMPLITUDE, len(wav)).astype(np.float32)
        wav = wav + noise
        wav = np.clip(wav, -1.0, 1.0)

        return wav

    def _add_end_padding(self, wav, sr: int):
        """Add short silence at end to prevent clipping."""
        import numpy as np

        padding = np.zeros(int(self.END_PADDING_SECONDS * sr), dtype=np.float32)
        return np.concatenate([wav, padding])


# =============================================================================
# TTS Generator
# =============================================================================

class TTSGenerator:
    """Generates audio using Qwen3-TTS with voice cloning and voice design.

    Characters with voice samples in character_samples/ use the Base model
    (voice cloning) for higher quality. Characters without samples fall back
    to the VoiceDesign model.
    """

    def __init__(self, voice_config: VoiceConfigManager, dry_run: bool = False, takes: int = 3):
        self.voice_config = voice_config
        self.emotion_parser = EmotionParser()
        self.text_preprocessor = TextPreprocessor()
        self.audio_postprocessor = AudioPostProcessor()
        self.dry_run = dry_run
        self.takes = takes
        self.clone_model = None
        self.design_model = None
        self.quality_model = None  # UTMOSv2 for scoring takes
        self._clone_prompts = {}  # Cache reusable voice clone prompts
        self._needs_clone = False
        self._needs_design = False

    def _get_model_kwargs(self) -> dict:
        """Build common model loading kwargs."""
        device = _get_device()
        dtype = _get_dtype()

        kwargs = {
            "device_map": device,
            "dtype": dtype,
        }

        # Try to use flash attention if available
        try:
            import flash_attn
            kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            pass

        return kwargs

    def _determine_needed_models(self, chapters: list) -> None:
        """Scan chapters to decide which models to load."""
        for chapter in chapters:
            for scene in chapter.scenes:
                for line in scene.lines:
                    if self.voice_config.has_voice_sample(line.character):
                        self._needs_clone = True
                    else:
                        self._needs_design = True
                    if self._needs_clone and self._needs_design:
                        return

    def _unload_model(self, model_attr: str):
        """Unload a model to free GPU memory."""
        model = getattr(self, model_attr)
        if model is not None:
            del model
            setattr(self, model_attr, None)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _load_clone_model(self):
        """Load the Base model for voice cloning."""
        if self.clone_model is not None:
            return

        _load_tts_dependencies()
        # Unload design model to free VRAM
        self._unload_model("design_model")
        kwargs = self._get_model_kwargs()

        print(f"Loading voice clone model: {MODEL_VOICE_CLONE}")
        print(f"  Device: {kwargs.get('device_map')}, Dtype: {kwargs.get('dtype')}")
        self.clone_model = Qwen3TTSModel.from_pretrained(MODEL_VOICE_CLONE, **kwargs)
        print(f"  Clone model loaded (VRAM: {torch.cuda.memory_allocated()/1024**3:.1f}GB)" if torch.cuda.is_available() else "  Clone model loaded")

    def _load_design_model(self):
        """Load the VoiceDesign model for characters without samples."""
        if self.design_model is not None:
            return

        _load_tts_dependencies()
        # Unload clone model to free VRAM
        self._unload_model("clone_model")
        self._clone_prompts.clear()
        kwargs = self._get_model_kwargs()

        print(f"Loading voice design model: {MODEL_VOICE_DESIGN}")
        print(f"  Device: {kwargs.get('device_map')}, Dtype: {kwargs.get('dtype')}")
        self.design_model = Qwen3TTSModel.from_pretrained(MODEL_VOICE_DESIGN, **kwargs)
        print("  Design model loaded")

    def _load_quality_model(self):
        """Load UTMOSv2 model for scoring TTS quality."""
        if self.quality_model is not None:
            return
        import utmosv2
        device = _get_device()
        print(f"Loading UTMOSv2 quality model on {device}")
        self.quality_model = utmosv2.create_model(pretrained=True, device=device)
        print("  UTMOSv2 loaded")

    def _score_quality(self, wav, sr: int) -> float:
        """Score audio quality using UTMOSv2. Returns MOS score (1.0-5.0)."""
        import torch
        import numpy as np
        self._load_quality_model()
        # UTMOSv2 expects 16kHz; resample if needed
        if sr != 16000:
            import torchaudio
            wav_tensor = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
            wav_tensor = torchaudio.functional.resample(wav_tensor, sr, 16000)
            data = wav_tensor.squeeze(0)
        else:
            data = torch.tensor(wav, dtype=torch.float32)
        score = self.quality_model.predict(data=data, sr=16000)
        # predict returns a tensor; extract scalar
        if hasattr(score, 'item'):
            return score.item()
        return float(score)

    def _get_clone_prompt(self, character: str):
        """Get or create a cached voice clone prompt for a character."""
        if character in self._clone_prompts:
            return self._clone_prompts[character]

        sample_path = self.voice_config.get_voice_sample(character)
        if sample_path is None:
            return None

        self._load_clone_model()

        print(f"  Creating voice clone prompt for {character} from {sample_path.name}")
        prompt = self.clone_model.create_voice_clone_prompt(
            ref_audio=str(sample_path),
            ref_text="",
            x_vector_only_mode=True,
        )

        # Cache for all character variants that share this sample
        self._clone_prompts[character] = prompt

        # Also cache for other characters that use the same sample file
        sample_name = sample_path.name
        for char_name, mapped_sample in CHARACTER_SAMPLE_MAP.items():
            if mapped_sample == sample_name and char_name not in self._clone_prompts:
                self._clone_prompts[char_name] = prompt

        return prompt

    # Beat pause durations in seconds
    BEAT_DURATIONS = {
        'beat': 0.7,
        'long beat': 1.5,
    }

    def _extract_beats(self, text: str) -> list:
        """Split text on [beat] and [long beat] markers.

        Returns a list of segments: either ('text', str) or ('pause', float).
        """
        # Match [beat] and [long beat] (case-insensitive)
        pattern = r'\[(long\s+beat|beat)\]'
        parts = re.split(pattern, text, flags=re.IGNORECASE)

        segments = []
        for i, part in enumerate(parts):
            if i % 2 == 0:
                # Text segment
                stripped = part.strip()
                if stripped:
                    segments.append(('text', stripped))
            else:
                # Beat marker
                key = part.lower().strip()
                duration = self.BEAT_DURATIONS.get(key, 0.7)
                segments.append(('pause', duration))

        return segments

    def generate_audio(self, text: str, character: str,
                       stage_direction: Optional[str] = None) -> tuple:
        """Generate audio for a line of dialogue with emotion/delivery context."""
        import numpy as np

        # Extract [beat] / [long beat] markers before other processing
        segments = self._extract_beats(text)
        has_beats = any(s[0] == 'pause' for s in segments)

        # If no beats, process normally as a single segment
        if not has_beats:
            return self._generate_segment(text, character, stage_direction)

        # Multiple segments separated by beats — generate each and stitch
        if self.dry_run:
            for seg_type, seg_value in segments:
                if seg_type == 'pause':
                    print(f"      [PAUSE {seg_value}s]")
                else:
                    wav, sr = self._generate_segment(seg_value, character, stage_direction)
            return None, None

        audio_parts = []
        sr = None
        for seg_type, seg_value in segments:
            if seg_type == 'pause':
                if sr:
                    silence = np.zeros(int(seg_value * sr), dtype=np.float32)
                    audio_parts.append(silence)
            else:
                wav, sr = self._generate_segment(seg_value, character, stage_direction)
                if wav is not None:
                    audio_parts.append(wav)

        if not audio_parts or sr is None:
            return None, None

        return np.concatenate(audio_parts), sr

    def _generate_segment(self, text: str, character: str,
                          stage_direction: Optional[str] = None) -> tuple:
        """Generate audio for a single text segment (no beat markers)."""
        # Parse inline directions from text (marked as [direction])
        clean_text, inline_instructions = self.emotion_parser.parse_inline_directions(text)

        # Apply text preprocessing (fix dashes, emphasis, pauses)
        clean_text, emphasized_words = self.text_preprocessor.process(clean_text)

        # Get base voice description
        base_voice = self.voice_config.get_voice_description(character)

        # Build full instruction with emotions
        full_instruction = self.emotion_parser.build_full_instruction(
            base_voice, stage_direction, inline_instructions
        )

        # Add emphasis instruction for words that had *markdown emphasis*
        if emphasized_words:
            words_list = ", ".join(f"'{w}'" for w in emphasized_words)
            full_instruction += f" Stress and emphasize these words: {words_list}."

        # Add question intonation instruction when text ends with ?
        if clean_text.rstrip().endswith('?'):
            full_instruction += " Use rising intonation at the end, clearly sounding like a question."

        # Check if this character has a voice sample
        has_sample = self.voice_config.has_voice_sample(character)
        mode = "clone" if has_sample else "design"

        if self.dry_run:
            emotion_info = ""
            if stage_direction:
                emotion_info = f" [{stage_direction}]"
            if inline_instructions:
                emotion_info += f" (inline: {len(inline_instructions)} cues)"
            sample_info = f" [CLONE: {self.voice_config.get_voice_sample(character).name}]" if has_sample else " [DESIGN]"
            print(f"  [DRY RUN] {character}{emotion_info}{sample_info}: {clean_text[:50]}...")
            if stage_direction or inline_instructions:
                print(f"      Instruction: {full_instruction[:80]}...")
            return None, None

        # Generate multiple takes and keep the best by UTMOSv2 quality score
        candidates = []
        sr = None
        for take in range(self.takes):
            if mode == "clone":
                wav, sr = self._generate_clone(clean_text, character, full_instruction)
            else:
                wav, sr = self._generate_design(clean_text, full_instruction)
            wav = self.audio_postprocessor.process(wav, sr)
            candidates.append(wav)

        if self.takes > 1:
            scores = []
            for wav in candidates:
                score = self._score_quality(wav, sr)
                scores.append(score)
            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            best_wav = candidates[best_idx]
            score_strs = ", ".join(f"{s:.2f}" for s in scores)
            print(f"        (MOS: [{score_strs}] -> kept take {best_idx + 1}/{self.takes})")
        else:
            best_wav = candidates[0]

        return best_wav, sr

    def _generate_clone(self, text: str, character: str, instruct: str) -> tuple:
        """Generate audio using voice cloning from a reference sample."""
        self._load_clone_model()
        clone_prompt = self._get_clone_prompt(character)

        wavs, sr = self.clone_model.generate_voice_clone(
            text=text,
            language="English",
            voice_clone_prompt=clone_prompt,
            instruct=instruct,
        )

        return wavs[0], sr

    def _generate_design(self, text: str, instruct: str) -> tuple:
        """Generate audio using voice design from a text description."""
        self._load_design_model()

        wavs, sr = self.design_model.generate_voice_design(
            text=text,
            language="English",
            instruct=instruct,
        )

        return wavs[0], sr

    def generate_chapter_audio(self, chapter: Chapter, output_dir: Path) -> list[Path]:
        """Generate audio files for an entire chapter."""
        output_dir.mkdir(parents=True, exist_ok=True)
        audio_files = []

        print(f"\nProcessing Chapter {chapter.number}: {chapter.title}")
        print(f"  Scenes: {len(chapter.scenes)}")

        total_lines = sum(len(scene.lines) for scene in chapter.scenes)
        print(f"  Total dialogue lines: {total_lines}")

        # Pre-scan to determine which models are needed for this chapter
        if not self.dry_run:
            self._determine_needed_models([chapter])
            if self._needs_clone:
                self._load_clone_model()
            if self._needs_design:
                self._load_design_model()

        line_index = 0
        for scene in chapter.scenes:
            print(f"\n  Scene {scene.number}: {scene.location} - {scene.time_of_day}")

            for line in scene.lines:
                line_index += 1
                filename = f"ch{chapter.number:02d}_s{scene.number:02d}_l{line_index:04d}_{self._sanitize_filename(line.character)}.wav"
                output_path = output_dir / filename

                # Show stage direction if present
                direction_info = f" ({line.stage_direction})" if line.stage_direction else ""
                print(f"    [{line_index}/{total_lines}] {line.character}{direction_info}: {line.text[:40]}...")

                wav, sr = self.generate_audio(line.text, line.character, line.stage_direction)

                if wav is not None:
                    sf.write(str(output_path), wav, sr)
                    audio_files.append(output_path)
                    print(f"      -> Saved: {filename}")

        return audio_files

    def regenerate_lines(self, chapter: Chapter, output_dir: Path, line_numbers: set[int]) -> list[Path]:
        """Regenerate only specific lines within a chapter."""
        output_dir.mkdir(parents=True, exist_ok=True)
        regenerated = []

        print(f"\nRegenerating lines in Chapter {chapter.number}: {chapter.title}")

        if not self.dry_run:
            self._determine_needed_models([chapter])
            if self._needs_clone:
                self._load_clone_model()
            if self._needs_design:
                self._load_design_model()

        total_lines = sum(len(scene.lines) for scene in chapter.scenes)
        line_index = 0
        for scene in chapter.scenes:
            for line in scene.lines:
                line_index += 1
                if line_index not in line_numbers:
                    continue

                filename = f"ch{chapter.number:02d}_s{scene.number:02d}_l{line_index:04d}_{self._sanitize_filename(line.character)}.wav"
                output_path = output_dir / filename

                direction_info = f" ({line.stage_direction})" if line.stage_direction else ""
                print(f"    [{line_index}/{total_lines}] {line.character}{direction_info}: {line.text[:40]}...")

                wav, sr = self.generate_audio(line.text, line.character, line.stage_direction)

                if wav is not None:
                    sf.write(str(output_path), wav, sr)
                    regenerated.append(output_path)
                    print(f"      -> Saved: {filename}")

        if not regenerated:
            print(f"  Warning: No matching lines found for {sorted(line_numbers)}")

        return regenerated

    def _sanitize_filename(self, name: str) -> str:
        """Convert character name to safe filename."""
        return re.sub(r'[^a-zA-Z0-9]', '_', name).lower()


# =============================================================================
# Chapter Audio Assembler
# =============================================================================

class AudioAssembler:
    """Assembles individual audio files into complete chapter audio."""

    def __init__(self, silence_duration: float = 0.5):
        self.silence_duration = silence_duration

    def assemble_chapter(self, audio_files: list[Path], output_path: Path):
        """Combine individual audio files into a single chapter file."""
        if not audio_files:
            print("No audio files to assemble")
            return

        _load_tts_dependencies()
        import numpy as np

        print(f"\nAssembling {len(audio_files)} audio files...")

        # Read all audio files
        audio_segments = []
        sample_rate = None

        for audio_file in sorted(audio_files):
            data, sr = sf.read(str(audio_file))
            if sample_rate is None:
                sample_rate = sr
            audio_segments.append(data)

        # Create silence between segments
        silence = np.zeros(int(self.silence_duration * sample_rate))

        # Concatenate with silence
        combined = []
        for i, segment in enumerate(audio_segments):
            combined.append(segment)
            if i < len(audio_segments) - 1:
                combined.append(silence)

        final_audio = np.concatenate(combined)

        # Write combined audio
        sf.write(str(output_path), final_audio, sample_rate)
        print(f"Chapter audio saved: {output_path}")


# =============================================================================
# Main CLI
# =============================================================================

def find_screenplay_files() -> list[Path]:
    """Find all screenplay markdown files."""
    if not SCREENPLAY_DIR.exists():
        print(f"Error: Screenplay directory not found: {SCREENPLAY_DIR}")
        sys.exit(1)

    files = sorted(SCREENPLAY_DIR.glob("*-screenplay.md"))
    return files


def parse_all_screenplays(files: list[Path]) -> list[Chapter]:
    """Parse all screenplay files."""
    parser = ScreenplayParser()
    chapters = []

    for filepath in files:
        print(f"Parsing: {filepath.name}")
        chapter = parser.parse_file(filepath)
        chapters.append(chapter)

    return chapters


def _parse_line_spec(spec: str) -> set[int]:
    """Parse a line specification like '11', '11,15,20', or '11-20' into a set of line numbers."""
    lines = set()
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-', 1)
            lines.update(range(int(start), int(end) + 1))
        else:
            lines.add(int(part))
    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-voice audiobook from screenplay markdown files using Qwen3-TTS"
    )
    parser.add_argument(
        '--chapter', '-c',
        type=int,
        help="Process only specific chapter number"
    )
    parser.add_argument(
        '--list-characters',
        action='store_true',
        help="List all characters found in screenplays"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Parse files without generating audio"
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory for audio files (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        '--assemble',
        action='store_true',
        help="Assemble individual audio files into chapter files"
    )
    parser.add_argument(
        '--line', '-l',
        type=str,
        help="Regenerate specific line(s) only. Examples: '11', '11,15,20', '11-20'. Requires --chapter."
    )
    parser.add_argument(
        '--takes',
        type=int,
        default=3,
        help="Number of TTS takes per line; keeps the shortest to reduce artifacts (default: 3)"
    )
    parser.add_argument(
        '--save-voice-config',
        action='store_true',
        help="Save default voice configuration to file for customization"
    )

    args = parser.parse_args()

    # Validate --line requires --chapter
    if args.line and args.chapter is None:
        parser.error("--line requires --chapter")

    # Find screenplay files
    screenplay_files = find_screenplay_files()
    print(f"Found {len(screenplay_files)} screenplay files")

    # Initialize voice configuration
    voice_config = VoiceConfigManager(VOICE_CONFIG_FILE)

    if args.save_voice_config:
        voice_config.save_config()
        print(f"Voice configuration saved to: {VOICE_CONFIG_FILE}")
        print("Edit this file to customize character voices.")
        return

    # Parse screenplays
    chapters = parse_all_screenplays(screenplay_files)

    # List characters mode
    if args.list_characters:
        parser_obj = ScreenplayParser()
        all_characters = parser_obj.get_all_characters(chapters)
        print("\n=== Characters Found ===")
        for char in sorted(all_characters):
            desc = voice_config.get_voice_description(char)
            sample = voice_config.get_voice_sample(char)
            mode = f"CLONE ({sample.name})" if sample else "DESIGN"
            print(f"\n{char}: [{mode}]")
            print(f"  Voice: {desc[:80]}...")
        return

    # Filter to specific chapter if requested
    if args.chapter is not None:
        chapters = [c for c in chapters if c.number == args.chapter]
        if not chapters:
            print(f"Error: Chapter {args.chapter} not found")
            sys.exit(1)

    # Parse --line spec into a set of line numbers
    line_numbers = None
    if args.line:
        line_numbers = _parse_line_spec(args.line)
        print(f"Regenerating line(s): {sorted(line_numbers)}")

    # Initialize TTS generator
    tts = TTSGenerator(voice_config, dry_run=args.dry_run, takes=args.takes)
    assembler = AudioAssembler()

    # Process chapters
    for chapter in chapters:
        chapter_output_dir = args.output_dir / f"chapter_{chapter.number:02d}"

        if line_numbers:
            # Regenerate only specific lines
            audio_files = tts.regenerate_lines(chapter, chapter_output_dir, line_numbers)
            # Re-assemble using all existing files in chapter dir
            if args.assemble:
                all_audio = sorted(chapter_output_dir.glob("ch*.wav"))
                if all_audio:
                    chapter_audio_path = args.output_dir / f"chapter_{chapter.number:02d}_complete.wav"
                    assembler.assemble_chapter(all_audio, chapter_audio_path)
        else:
            audio_files = tts.generate_chapter_audio(chapter, chapter_output_dir)
            if args.assemble and audio_files:
                chapter_audio_path = args.output_dir / f"chapter_{chapter.number:02d}_complete.wav"
                assembler.assemble_chapter(audio_files, chapter_audio_path)

    print("\n=== Processing Complete ===")


if __name__ == "__main__":
    main()
