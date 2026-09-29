"""Merge inversion chunk H5 files into a single output file."""
import h5py
import glob
import argparse

def merge_chunks(chunks_dir, out_path):
    chunk_files = sorted(glob.glob(f"{chunks_dir}/inv_*.h5"))
    print(f"Found {len(chunk_files)} chunk files")

    merged = 0
    skipped = 0
    with h5py.File(out_path, "a") as out:
        existing = set(out.keys())
        for chunk_file in chunk_files:
            with h5py.File(chunk_file, "r") as src:
                for mpid in src.keys():
                    if mpid in existing:
                        skipped += 1
                        continue
                    src.copy(mpid, out)
                    existing.add(mpid)
                    merged += 1

    with h5py.File(out_path, "r") as out:
        print(f"Merged {merged} new entries, skipped {skipped} duplicates")
        print(f"Total entries in output: {len(out.keys())}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--chunks-dir", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    merge_chunks(args.chunks_dir, args.output)
