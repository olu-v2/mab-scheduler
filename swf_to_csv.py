
# =============================================================================
# swf_to_csv.py  —  Convert SWF trace files to CSV
# =============================================================================
# Usage:
#   python3 swf_to_csv.py                                  (uses default file)
#   python3 swf_to_csv.py --swf NASA-iPSC-1993-3.1-cln.swf
#   python3 swf_to_csv.py --swf myfile.swf --out jobs.csv
# =============================================================================

import csv, os, argparse

# Standard SWF column definitions (18 columns, 1-based in spec)
SWF_COLUMNS = [
    "job_id",
    "submit_time",
    "wait_time",
    "run_time",
    "allocated_processors",
    "avg_cpu_time_used",
    "used_memory",
    "requested_processors",
    "requested_time",
    "requested_memory",
    "status",
    "user_id",
    "group_id",
    "application_id",
    "queue_id",
    "partition_id",
    "preceding_job_id",
    "think_time",
]


def parse_swf_header(filepath: str) -> dict:
    """Extract key-value metadata from SWF comment header lines."""
    meta = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line.startswith(';'):
                break
            if ':' in line:
                key, _, val = line[1:].partition(':')
                meta[key.strip()] = val.strip()
    return meta


def swf_to_csv(swf_path: str, csv_path: str = None,
               max_rows: int = None) -> str:
    """
    Convert an SWF file to CSV.

    Parameters
    ----------
    swf_path : str   Path to the .swf file
    csv_path : str   Output CSV path (default: same name with .csv extension)
    max_rows : int   Maximum data rows to convert (None = all)

    Returns
    -------
    str  Path to the written CSV file
    """
    if not os.path.exists(swf_path):
        raise FileNotFoundError(f"SWF file not found: {swf_path}")

    if csv_path is None:
        base = os.path.splitext(swf_path)[0]
        csv_path = base + '.csv'

    # --- Print header metadata ---
    meta = parse_swf_header(swf_path)
    if meta:
        print("SWF File Metadata:")
        for k, v in meta.items():
            print(f"  {k}: {v}")
        print()

    # --- Convert data rows ---
    rows_written = 0
    rows_skipped = 0
    comment_lines = []

    with open(swf_path, 'r') as swf_file,          open(csv_path, 'w', newline='') as csv_file:

        writer = csv.writer(csv_file)

        # Write header row
        writer.writerow(SWF_COLUMNS)

        for raw_line in swf_file:
            line = raw_line.strip()

            # Store comment lines (metadata) but skip them for data
            if not line:
                continue
            if line.startswith(';'):
                comment_lines.append(line)
                continue

            fields = line.split()

            # Pad or trim to exactly 18 columns
            if len(fields) < 18:
                fields += ['-1'] * (18 - len(fields))
            else:
                fields = fields[:18]

            writer.writerow(fields)
            rows_written += 1

            if max_rows and rows_written >= max_rows:
                break

    print(f"Converted  : {swf_path}")
    print(f"Output CSV : {csv_path}")
    print(f"Rows written: {rows_written:,}")
    print(f"Comment lines in header: {len(comment_lines)}")
    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SWF trace to CSV")
    parser.add_argument("--swf", default="NASA-iPSC-1993-3.1-cln.swf",
                        help="Path to the .swf input file")
    parser.add_argument("--out", default=None,
                        help="Path for the output .csv file (optional)")
    parser.add_argument("--max_rows", type=int, default=None,
                        help="Max rows to convert (optional, default: all)")
    args = parser.parse_args()

    swf_to_csv(args.swf, args.out, args.max_rows)
