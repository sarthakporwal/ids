"""
Large File Processor for CANShield
Handles large CSV files (>100MB) with chunked processing
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
import json

sys.path.insert(0, str(Path(__file__).parent / "src"))

class LargeFileProcessor:
    """Process large CSV files in chunks to avoid memory issues"""
    
    def __init__(self, chunk_size=10000):
        """
        Initialize processor
        
        Args:
            chunk_size: Number of rows to process at once
        """
        self.chunk_size = chunk_size
        self.processed_chunks = []
        
    def process_large_csv(self, file_path, output_dir=None, sample_fraction=0.1):
        """
        Process large CSV file in chunks
        
        Args:
            file_path: Path to large CSV file
            output_dir: Directory to save processed chunks (optional)
            sample_fraction: Fraction of data to use (0.1 = 10%)
            
        Returns:
            dict with processing stats
        """
        file_path = Path(file_path)
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Processing large file: {file_path.name}")
        print(f"📊 File size: {file_path.stat().st_size / (1024**2):.2f} MB")
        print(f"🎯 Using {sample_fraction*100}% of data for speed")
        print("")
        
        # First pass: count total rows
        print("🔍 Counting rows...")
        total_rows = sum(1 for _ in open(file_path)) - 1  # Subtract header
        print(f"✅ Total rows: {total_rows:,}")
        
        # Calculate sampling
        rows_to_use = int(total_rows * sample_fraction)
        skip_interval = max(1, int(1 / sample_fraction))
        
        print(f"📥 Will process ~{rows_to_use:,} rows")
        print("")
        
        # Process in chunks
        chunks_processed = 0
        total_processed = 0
        
        print("⚙️  Processing chunks...")
        
        chunk_iterator = pd.read_csv(
            file_path,
            chunksize=self.chunk_size,
            skiprows=lambda x: x > 0 and x % skip_interval != 0  # Sample rows
        )
        
        processed_data = []
        
        for i, chunk in enumerate(tqdm(chunk_iterator, desc="Processing")):
            # Process chunk
            chunk_processed = self._process_chunk(chunk)
            processed_data.append(chunk_processed)
            
            chunks_processed += 1
            total_processed += len(chunk)
            
            # Save intermediate results if output_dir specified
            if output_dir and chunks_processed % 10 == 0:
                temp_file = output_dir / f"chunk_{chunks_processed}.csv"
                pd.concat(processed_data).to_csv(temp_file, index=False)
                processed_data = []  # Clear memory
        
        # Combine all chunks
        print("\n🔗 Combining processed chunks...")
        
        if output_dir and processed_data:
            # Save final chunk
            final_chunk_file = output_dir / f"chunk_final.csv"
            pd.concat(processed_data).to_csv(final_chunk_file, index=False)
            
            # Combine all saved chunks
            all_chunk_files = sorted(output_dir.glob("chunk_*.csv"))
            combined_data = pd.concat([
                pd.read_csv(f) for f in tqdm(all_chunk_files, desc="Combining")
            ])
            
            # Clean up chunk files
            for f in all_chunk_files:
                f.unlink()
        else:
            combined_data = pd.concat(processed_data)
        
        print(f"\n✅ Processing complete!")
        print(f"📊 Processed {total_processed:,} rows in {chunks_processed} chunks")
        
        stats = {
            'original_rows': total_rows,
            'processed_rows': total_processed,
            'chunks_processed': chunks_processed,
            'sample_fraction': sample_fraction,
            'final_shape': combined_data.shape
        }
        
        return combined_data, stats
    
    def _process_chunk(self, chunk):
        """Process a single chunk of data"""
        # Basic processing - can be customized
        # Forward fill missing values
        chunk = chunk.ffill()
        chunk = chunk.bfill()
        
        return chunk
    
    def create_sampled_dataset(self, input_file, output_file, sample_size=50000):
        """
        Create a smaller sampled version of large dataset
        
        Args:
            input_file: Path to large CSV
            output_file: Path to save sampled CSV
            sample_size: Number of rows to keep
            
        Returns:
            Path to sampled file
        """
        input_file = Path(input_file)
        output_file = Path(output_file)
        
        print(f"🎲 Creating sampled dataset...")
        print(f"📁 Input: {input_file.name} ({input_file.stat().st_size / (1024**2):.2f} MB)")
        print(f"🎯 Target rows: {sample_size:,}")
        
        # Count total rows
        total_rows = sum(1 for _ in open(input_file)) - 1
        
        # Calculate skip rows for uniform sampling
        skip_prob = 1 - (sample_size / total_rows)
        
        print(f"📊 Sampling {(1-skip_prob)*100:.1f}% of {total_rows:,} rows")
        
        # Read with random sampling
        df = pd.read_csv(
            input_file,
            skiprows=lambda x: x > 0 and np.random.random() > (1 - skip_prob)
        )
        
        # Ensure we have close to target size
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
        
        # Save
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        
        print(f"✅ Sampled dataset created!")
        print(f"📁 Output: {output_file}")
        print(f"📦 Size: {output_file.stat().st_size / (1024**2):.2f} MB")
        print(f"📊 Rows: {len(df):,}")
        
        return output_file
    
    def split_large_file(self, input_file, output_dir, max_rows_per_file=50000):
        """
        Split large file into smaller files
        
        Args:
            input_file: Path to large CSV
            output_dir: Directory to save split files
            max_rows_per_file: Maximum rows per split file
            
        Returns:
            List of split file paths
        """
        input_file = Path(input_file)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✂️  Splitting large file...")
        print(f"📁 Input: {input_file.name}")
        print(f"📦 Max rows per file: {max_rows_per_file:,}")
        
        split_files = []
        file_count = 0
        
        # Read header
        header = pd.read_csv(input_file, nrows=0).columns.tolist()
        
        # Process in chunks
        for chunk in tqdm(
            pd.read_csv(input_file, chunksize=max_rows_per_file),
            desc="Splitting"
        ):
            file_count += 1
            output_file = output_dir / f"{input_file.stem}_part{file_count}.csv"
            
            chunk.to_csv(output_file, index=False)
            split_files.append(output_file)
        
        print(f"\n✅ Split into {file_count} files")
        print(f"📁 Output directory: {output_dir}")
        
        return split_files


def estimate_memory_usage(file_path):
    """Estimate memory required to load file"""
    file_size = Path(file_path).stat().st_size
    # CSV typically needs 5-10x file size in memory
    estimated_memory = file_size * 7
    
    return {
        'file_size_mb': file_size / (1024**2),
        'estimated_memory_mb': estimated_memory / (1024**2),
        'estimated_memory_gb': estimated_memory / (1024**3)
    }


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process large CAN dataset files')
    parser.add_argument('input_file', help='Path to large CSV file')
    parser.add_argument('--action', choices=['sample', 'split', 'process'], 
                       default='sample', help='Action to perform')
    parser.add_argument('--output', default='processed_data',
                       help='Output directory or file')
    parser.add_argument('--sample-size', type=int, default=50000,
                       help='Number of rows for sampling')
    parser.add_argument('--sample-fraction', type=float, default=0.1,
                       help='Fraction of data to process (0.1 = 10%)')
    
    args = parser.parse_args()
    
    processor = LargeFileProcessor()
    
    # Estimate memory
    mem_info = estimate_memory_usage(args.input_file)
    print("💾 Memory Estimate:")
    print(f"   File size: {mem_info['file_size_mb']:.2f} MB")
    print(f"   Est. memory needed: {mem_info['estimated_memory_mb']:.2f} MB "
          f"({mem_info['estimated_memory_gb']:.2f} GB)")
    print("")
    
    if args.action == 'sample':
        output_file = args.output if args.output.endswith('.csv') else f"{args.output}/sampled_data.csv"
        processor.create_sampled_dataset(
            args.input_file,
            output_file,
            sample_size=args.sample_size
        )
    
    elif args.action == 'split':
        processor.split_large_file(
            args.input_file,
            args.output,
            max_rows_per_file=args.sample_size
        )
    
    elif args.action == 'process':
        data, stats = processor.process_large_csv(
            args.input_file,
            output_dir=args.output,
            sample_fraction=args.sample_fraction
        )
        
        print("\n📊 Processing Statistics:")
        print(json.dumps(stats, indent=2))
        
        # Save processed data
        output_file = Path(args.output) / 'processed_data.csv'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(output_file, index=False)
        print(f"\n💾 Saved to: {output_file}")


if __name__ == "__main__":
    main()

