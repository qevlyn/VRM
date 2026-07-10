from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description='Reconstruct a split GitHub Release asset from verified parts.')
    parser.add_argument('--manifest', required=True, help='Path to the split parts manifest JSON.')
    parser.add_argument('--output', required=True, help='Path to write the reconstructed output file.')
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    base_dir = manifest_path.parent
    output = Path(args.output).resolve()
    parts = sorted(manifest['parts'], key=lambda row: row['ordinal'])
    for part in parts:
        path = base_dir / part['filename']
        if not path.exists():
            raise SystemExit(f'missing part: {path}')
        size = path.stat().st_size
        if size != part['byte_size']:
            raise SystemExit(f'part size mismatch for {path}: expected {part["byte_size"]}, got {size}')
        digest = sha256_file(path)
        if digest != part['sha256']:
            raise SystemExit(f'part SHA-256 mismatch for {path}: expected {part["sha256"]}, got {digest}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('wb') as out:
        for part in parts:
            with (base_dir / part['filename']).open('rb') as src:
                shutil.copyfileobj(src, out, length=1024 * 1024)
    size = output.stat().st_size
    if size != manifest['original_byte_size']:
        raise SystemExit(f'reconstructed size mismatch: expected {manifest["original_byte_size"]}, got {size}')
    digest = sha256_file(output)
    if digest != manifest['original_sha256']:
        raise SystemExit(f'reconstructed SHA-256 mismatch: expected {manifest["original_sha256"]}, got {digest}')
    print(f'reconstructed {output}')
    print(f'sha256 {digest}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
