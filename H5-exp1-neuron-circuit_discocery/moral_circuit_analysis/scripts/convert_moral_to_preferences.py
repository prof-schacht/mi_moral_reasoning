#!/usr/bin/env python
"""Convert existing moral/immoral pairs to preference format."""

import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data.preference_data import create_preference_dataset_from_moral_data


def main():
    parser = argparse.ArgumentParser(
        description="Convert moral/immoral pairs to preference scenarios"
    )
    parser.add_argument(
        "--input", 
        type=Path, 
        required=True,
        help="Path to moral pairs JSON file"
    )
    parser.add_argument(
        "--output", 
        type=Path, 
        required=True,
        help="Output path for preference data"
    )
    parser.add_argument(
        "--include-cross-dimension",
        action="store_true",
        help="Include cross-dimension preference comparisons"
    )
    
    args = parser.parse_args()
    
    # Check if input exists
    if not args.input.exists():
        print(f"Error: Input file {args.input} does not exist")
        return 1
    
    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert the data
    print(f"Converting moral pairs from {args.input}")
    preference_data = create_preference_dataset_from_moral_data(
        args.input,
        args.output,
        include_cross_dimension=args.include_cross_dimension
    )
    
    print(f"\nConversion complete! Preference data saved to {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())