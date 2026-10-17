#!/usr/bin/env python3
"""Verify the published source bytes and syntax without loading any model."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'provenance/source_manifest.json').read_text())
for item in manifest['files']:
    path = root / item['path']
    payload = path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == item['published_sha256'], str(path)
    if path.suffix == '.py':
        ast.parse(payload.decode('utf-8'), filename=str(path))
print(json.dumps({'verified_source_files': len(manifest['files']), 'model_calls': 0}))
