"""Execute the student notebooks in order and save fresh outputs.

Example: python run_notebooks.py --start 1 --end 11
Feature caches and the raw claims file must already exist; see README.md.
"""
import argparse
import os
from pathlib import Path
import time

# Avoid excessive numerical-library threads when several notebooks use trees.
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--end', type=int, default=11)
    parser.add_argument('--timeout', type=int, default=1800, help='Seconds per code cell')
    args = parser.parse_args()
    if not 1 <= args.start <= args.end <= 11:
        parser.error('Choose a range between 1 and 11.')
    root = Path(__file__).resolve().parent
    for path in sorted(root.glob('[0-9][0-9]_*.ipynb')):
        number = int(path.name[:2])
        if not args.start <= number <= args.end:
            continue
        start = time.monotonic()
        print(f'Running {path.name}', flush=True)
        notebook = nbformat.read(path, as_version=4)
        def progress(cell, cell_index, **kwargs):
            print(f'  Completed code cell {cell_index + 1}', flush=True)
            for output in cell.get('outputs', []):
                if output.get('output_type') == 'stream' and output.get('name') == 'stdout':
                    for line in output.get('text', '').splitlines():
                        if line.startswith(('Finished ', 'Choice frozen', 'Frozen warning')):
                            print('  ' + line, flush=True)
        try:
            NotebookClient(notebook, timeout=args.timeout, kernel_name='python3',
                           on_cell_executed=progress,
                           resources={'metadata': {'path': str(root)}}).execute()
        finally:
            nbformat.write(notebook, path)
        print(f'Finished in {time.monotonic() - start:.1f} seconds', flush=True)
    if args.start == 1 and args.end == 11:
        from refresh_summary import main as refresh_summary
        refresh_summary()


if __name__ == '__main__':
    main()
