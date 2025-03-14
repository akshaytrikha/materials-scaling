#!/usr/bin/env python3

import argparse
import lmdb
import numpy as np
import os
import zlib
import orjson
import random
from typing import Optional, List
from data_utils import VALID_DATASETS


def merge_lmdb_databases(
    src_paths: List[str],
    output_path: str,
    entries_per_db: int = 10,
    fraction: Optional[float] = None,
    key_prefix: Optional[str] = None,
    verbose: bool = False,
    shuffle: bool = False,
    seed: Optional[int] = None,
):
    """
    Merge multiple LMDB databases into a single new database.

    Args:
        src_paths: List of paths to the source LMDB databases
        output_path: Path where the new merged database will be created
        entries_per_db: Number of entries to take from each database
        fraction: Fraction of entries to take from each database (overrides entries_per_db)
        key_prefix: Optional prefix to filter keys
        verbose: Print detailed information during merge
        shuffle: Whether to randomly select entries instead of taking them in order
        seed: Random seed for random selection
    """
    if seed is not None:
        random.seed(seed)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Check if output file already exists
    if os.path.exists(output_path):
        raise ValueError(f"Output file '{output_path}' already exists.")

    # Create new LMDB environment for output
    is_aselmdb = output_path.endswith(".aselmdb")
    env_out = lmdb.open(
        output_path,
        subdir=not is_aselmdb,
        map_size=1099511627776,  # 1TB
        meminit=False,
        map_async=True,
    )
    
    # Special keys that should be merged/copied
    special_keys = ["metadata"]
    combined_metadata = {}
    next_id = 1
    
    # Process each source database
    with env_out.begin(write=True) as txn_out:
        for db_path in src_paths:
            if verbose:
                print(f"Processing database: {db_path}")
            
            # Check if path exists
            if not os.path.exists(db_path):
                print(f"Warning: Database path '{db_path}' does not exist. Skipping.")
                continue
                
            # Handle directory with .aselmdb file
            if os.path.isdir(db_path):
                aselmdb_path = None
                for item in os.listdir(db_path):
                    if item.endswith(".aselmdb"):
                        aselmdb_path = os.path.join(db_path, item)
                        break
                
                if aselmdb_path:
                    if verbose:
                        print(f"Found ASELMDB file: {aselmdb_path}")
                    db_path = aselmdb_path
            
            # Open source database
            is_src_aselmdb = db_path.endswith(".aselmdb")
            env_src = lmdb.open(
                db_path,
                readonly=True,
                lock=False,
                subdir=not is_src_aselmdb,
                meminit=False,
                map_async=True,
            )
            
            try:
                with env_src.begin() as txn_src:
                    # Get special keys from source
                    for key_name in special_keys:
                        key_bytes = key_name.encode("ascii")
                        value = txn_src.get(key_bytes)
                        if value is not None:
                            try:
                                decoded = orjson.loads(zlib.decompress(value))
                                if key_name == "metadata":
                                    # Merge metadata 
                                    if not combined_metadata:
                                        combined_metadata = decoded
                                    else:
                                        # Merge metadata dictionaries
                                        for meta_key, meta_val in decoded.items():
                                            if meta_key not in combined_metadata:
                                                combined_metadata[meta_key] = meta_val
                                            # Could add more sophisticated merging for duplicate keys
                            except Exception as e:
                                print(f"Error decoding special key '{key_name}': {e}")
                    
                    # Determine how many entries to take
                    if fraction is not None:
                        # Count valid entries if using fraction
                        total_entries = 0
                        valid_keys = []
                        cursor = txn_src.cursor()
                        
                        for key, _ in cursor:
                            key_str = key.decode("ascii", errors="replace")
                            if key_str in ["nextid", "metadata", "deleted_ids"]:
                                continue
                            if key_prefix and not key_str.startswith(key_prefix):
                                continue
                            valid_keys.append(key)
                            total_entries += 1
                        
                        # Calculate entries to take based on fraction
                        entries_to_take = max(1, int(total_entries * fraction))
                        
                        if verbose:
                            print(f"  Database has {total_entries} entries, taking {entries_to_take} ({fraction:.1%})")
                        
                        # Randomly sample keys
                        if entries_to_take < total_entries:
                            sampled_keys = random.sample(valid_keys, entries_to_take)
                        else:
                            sampled_keys = valid_keys
                        
                        # Copy the sampled entries
                        count = 0
                        for key in sampled_keys:
                            try:
                                key_str = key.decode("ascii", errors="replace")
                                value = txn_src.get(key)
                                
                                # Try to decode as ASE format
                                try:
                                    decoded_value = orjson.loads(zlib.decompress(value))
                                    
                                    # Write to output database with new ID
                                    txn_out.put(
                                        f"{next_id}".encode("ascii"),
                                        zlib.compress(orjson.dumps(decoded_value, option=orjson.OPT_SERIALIZE_NUMPY))
                                    )
                                    
                                    if verbose:
                                        print(f"  - Copied entry with original key '{key_str}' to new ID {next_id}")
                                    
                                    next_id += 1
                                    count += 1
                                    
                                except Exception as e:
                                    if verbose:
                                        print(f"  - Skipping entry with key '{key_str}': {e}")
                                    continue
                                    
                            except Exception as e:
                                print(f"Error processing entry with key {key}: {e}")
                        
                        if verbose:
                            print(f"  Copied {count} entries from {db_path}")
                    else:
                        # Modified method: either take entries in order or randomly
                        cursor = txn_src.cursor()
                        count = 0
                        valid_keys = []
                        
                        # First collect all valid keys
                        for key, _ in cursor:
                            key_str = key.decode("ascii", errors="replace")
                            if key_str in ["nextid", "metadata", "deleted_ids"]:
                                continue
                            if key_prefix and not key_str.startswith(key_prefix):
                                continue
                            valid_keys.append(key)
                        
                        # Select keys either randomly or in order
                        if shuffle:
                            selected_keys = random.sample(valid_keys, min(entries_per_db, len(valid_keys)))
                        else:
                            selected_keys = valid_keys[:entries_per_db]
                        
                        # Process selected keys
                        for key in selected_keys:
                            try:
                                key_str = key.decode("ascii", errors="replace")
                                value = txn_src.get(key)
                                
                                # Try to decode as ASE format
                                try:
                                    decoded_value = orjson.loads(zlib.decompress(value))
                                    
                                    # Write to output database with new ID
                                    txn_out.put(
                                        f"{next_id}".encode("ascii"),
                                        zlib.compress(orjson.dumps(decoded_value, option=orjson.OPT_SERIALIZE_NUMPY))
                                    )
                                    
                                    if verbose:
                                        print(f"  - Copied entry with original key '{key_str}' to new ID {next_id}")
                                    
                                    next_id += 1
                                    count += 1
                                    
                                except Exception as e:
                                    if verbose:
                                        print(f"  - Skipping entry with key '{key_str}': {e}")
                                    continue
                                    
                            except Exception as e:
                                print(f"Error processing entry with key {key}: {e}")
            
            finally:
                env_src.close()
        
        # Write special keys to output database
        if combined_metadata:
            txn_out.put(
                "metadata".encode("ascii"),
                zlib.compress(orjson.dumps(combined_metadata, option=orjson.OPT_SERIALIZE_NUMPY))
            )
            
        # Write nextid
        txn_out.put(
            "nextid".encode("ascii"),
            zlib.compress(orjson.dumps(next_id, option=orjson.OPT_SERIALIZE_NUMPY))
        )
        
        # Write empty deleted_ids
        txn_out.put(
            "deleted_ids".encode("ascii"),
            zlib.compress(orjson.dumps([], option=orjson.OPT_SERIALIZE_NUMPY))
        )

    # Get all entries from the output database and shuffle them
    if shuffle:
        with env_out.begin(write=True) as txn_out:
            # Collect all regular entries (excluding special keys)
            entries = []
            cursor = txn_out.cursor()
            for key, value in cursor:
                key_str = key.decode("ascii", errors="replace")
                if key_str not in ["nextid", "metadata", "deleted_ids"]:
                    entries.append((int(key_str), value))
            
            # Shuffle entries
            random.shuffle(entries)
            
            # Delete old entries
            cursor = txn_out.cursor()
            for key, _ in cursor:
                key_str = key.decode("ascii", errors="replace")
                if key_str not in ["nextid", "metadata", "deleted_ids"]:
                    txn_out.delete(key)
            
            # Write shuffled entries with new sequential IDs
            new_id = 1
            for _, value in entries:
                txn_out.put(str(new_id).encode("ascii"), value)
                new_id += 1
            
            # Update nextid
            txn_out.put(
                "nextid".encode("ascii"),
                zlib.compress(orjson.dumps(new_id, option=orjson.OPT_SERIALIZE_NUMPY))
            )

    env_out.close()
    print(f"Successfully merged databases into {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Utilities for LMDB databases")
    # Dataset selection arguments
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Names of the source datasets"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (e.g., 'val', 'train')"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory name for the merged database"
    )
    parser.add_argument(
        "--datasets_base_path",
        type=str,
        default="./datasets",
        help="Base path for all datasets"
    )
    
    # Make entries_per_db and fraction mutually exclusive
    entries_group = parser.add_mutually_exclusive_group()
    entries_group.add_argument(
        "--entries-per-db",
        type=int,
        default=10,
        help="Number of entries to take from each database (default: 10)",
    )
    entries_group.add_argument(
        "--fraction",
        type=float,
        help="Fraction of entries to take from each database (0.0-1.0)",
    )
    
    parser.add_argument("--key-prefix", help="Only include keys with this prefix")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information during merge")
    parser.add_argument("--shuffle", action="store_true", default=True, help="Randomly select entries instead of taking them in order")
    parser.add_argument("--seed", type=int, default=99, help="Random seed for random selection")

    args = parser.parse_args()

    if args.datasets[0] == "all":
        args.datasets = VALID_DATASETS

    # Construct paths
    src_paths = [
        os.path.join(args.datasets_base_path, args.split, dataset, "data.aselmdb")
        for dataset in args.datasets
    ]
    output_path = os.path.join(args.datasets_base_path, args.split, args.output, "data.aselmdb")

    merge_lmdb_databases(
        src_paths, 
        output_path,
        args.entries_per_db,
        args.fraction,
        args.key_prefix,
        args.verbose,
        args.shuffle,
        args.seed
    )
