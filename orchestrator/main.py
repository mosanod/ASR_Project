# =============================================================================
# Speaker-ID ASR Pipeline — Orchestrator Main Entry Point
# =============================================================================
"""
Main entry point for the orchestrator module.
Coordinates the full ASR pipeline: transcription, diarization, and post-processing.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
from audio.pipeline import ASRPipeline

logger = logging.getLogger(__name__)
settings = get_settings()


def setup_logging():
    """Configure logging format."""
    logging.basicConfig(
        level=getattr(logging, settings.app.log_level),
        format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def process_audio_file(input_path: str, output_dir: str) -> dict:
    """
    Process a single audio file through the full pipeline.
    
    Args:
        input_path: Path to input audio file
        output_dir: Directory to save results
        
    Returns:
        Dictionary with processing results
    """
    # Initialize pipeline
    pipeline = ASRPipeline(settings)
    
    # Run inference
    result = pipeline.transcribe(
        audio=input_path,
        output_dir=Path(output_dir),
    )
    
    return result.model_dump() if hasattr(result, 'model_dump') else str(result)


def main():
    """Main entry point."""
    setup_logging()
    logger.info("Starting Speaker-ID ASR Pipeline")
    
    parser = argparse.ArgumentParser(description='Process audio files through the ASR pipeline')
    parser.add_argument('--input', '-i', required=True, help='Path to input audio file')
    parser.add_argument('--output-dir', '-o', default='/app/data/output',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Check if input exists
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        result = process_audio_file(str(input_path), str(output_dir))
        
        # Save results
        output_file = output_dir / "transcription.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to {output_file}")
        print(json.dumps({"status": "success", "output": str(output_file)}))
        
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()