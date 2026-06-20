#!/usr/bin/env python3
"""
Script to run ASR transcription with diarization on TEST.wav file.
Input: data/input/TEST.wav
Output: data/output/transcription_result.json
"""

import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import Settings, get_settings
from audio.pipeline import ASRPipeline


def run_transcription():
    """Run transcription and diarization on TEST.wav file."""
    
    # Initialize settings and pipeline
    print("Loading configuration...")
    cfg = get_settings()
    pipeline = ASRPipeline(cfg)
    
    # Define paths
    input_file = Path("data/input/TEST.wav")
    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "transcription_result.json"
    
    print(f"Input file: {input_file}")
    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        return
    
    # Run transcription with diarization
    print("Starting transcription...")
    result = pipeline.transcribe(str(input_file), output_dir=output_dir)
    
    # Save results
    import json
    from dataclasses import asdict
    
    # Convert segments to serializable format
    output_data = {
        "segments": [asdict(seg) for seg in result.segments],
        "audio_info": result.audio_info,
        "total_duration": result.total_duration
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nTranscription complete!")
    print(f"Results saved to: {output_file}")
    print(f"Total segments: {len(result.segments)}")
    print(f"Total duration: {result.total_duration:.2f} seconds")
    
    # Print summary
    print("\n--- Transcription Summary ---")
    for i, seg in enumerate(result.segments[:5], 1):  # Show first 5 segments
        speaker = f"[{seg.speaker_id}]" if seg.speaker_id else "[unknown]"
        text = seg.text[:80] + "..." if len(seg.text) > 80 else seg.text
        print(f"{i:2d}. {speaker:10s} | {text}")
    
    if len(result.segments) > 5:
        print(f"... and {len(result.segments) - 5} more segments")


if __name__ == "__main__":
    run_transcription()
