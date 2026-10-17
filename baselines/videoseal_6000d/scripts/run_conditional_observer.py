#!/usr/bin/env python3
"""Portable orchestration of the captured conditional Observer method.

The original helper, packet selection, frame extraction and Observer are imported
from source/. This wrapper removes host-specific service/process management.
It has not been evaluated on GPUs as part of the source-publication task.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

SOURCE = Path(__file__).resolve().parents[1] / 'source'


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def finalize(funnel_rows, observed, results):
    """The actual coordinator's adoption rule; preserve all rows and predictions."""
    rows = []
    for ordinal, row in enumerate(funnel_rows):
        key = (row['video_id'], row['uid'])
        item = observed.get(key, {})
        prediction = str(item.get('prediction') or '')
        adopted = row['has_forced_full_video'] and prediction in {'A', 'B', 'C', 'D'}
        rows.append({'ordinal': ordinal, 'video_id': key[0], 'uid': key[1],
            'original_prediction': row['prediction'],
            'has_forced_full_video': row['has_forced_full_video'],
            'observer_prediction': prediction, 'observer_status': item.get('status'),
            'final_prediction': prediction if adopted else row['prediction'],
            'final_source': 'DirectObserver' if adopted else 'OriginalSearch',
            'search_state': results.get(key, {}).get('state', 'missing')})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parquet', required=True, type=Path)
    parser.add_argument('--search-root', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--model-path', required=True, type=Path)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    for field in ('parquet', 'search_root', 'output_root', 'model_path'):
        setattr(args, field, getattr(args, field).expanduser().resolve())
    # A separate output directory keeps the original search and failures intact.
    args.output_root.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(SOURCE), str(SOURCE / 'baselines/VideoSEAL'),
                   str(SOURCE / 'research/diagnostics'),
                   str(SOURCE / 'methods/CAUSAL_MODALITY_INSTRUMENT')]
    import pyarrow.parquet as pq
    import audit_fixed30_summary_planner_inspector as funnel_source
    import run_fixed30_inspector_pilots as helper
    observer_source = module_at('captured_original_observer',
        SOURCE / 'methods/CAUSAL_MODALITY_INSTRUMENT/diagnostics/run_observer_capacity.py')

    public_tasks = pq.read_table(args.parquet)['extra_info'].to_pylist()
    results = {}
    for line in (args.search_root / 'results.jsonl').read_text().splitlines():
        if line.strip():
            value = json.loads(line)
            key = (str(value.get('video_id') or ''), str(value.get('uid') or ''))
            results[key] = value
    if len(public_tasks) != 300:
        raise ValueError('This published baseline uses the fixed 300-row subset.')
    expected_keys = {(str(x['video_id']), str(x['qa_uid'])) for x in public_tasks}
    if len(expected_keys) != 300 or set(results) != expected_keys:
        raise ValueError('Search must contain the same complete 300 unique identities.')

    funnel_rows, baseline = [], {}
    for info in public_tasks:
        key = (str(info['video_id']), str(info['qa_uid']))
        result = results.get(key, {})
        path = funnel_source.resolve_trajectory(args.search_root, result or
            {'video_id': key[0], 'uid': key[1]})
        trajectory = json.loads(path.read_text()) if path is not None else {}
        extracted = funnel_source.audit_one(result, info, trajectory, '', {})
        row = {'video_id': key[0], 'uid': key[1], 'question': info['question'],
               'prediction': str(result.get('prediction') or ''),
               'has_forced_full_video': extracted['has_forced_full_video'],
               'inspections': extracted['inspections'],
               'trajectory_path': str(path) if path else None}
        funnel_rows.append(row)
        baseline[key] = {'video_id': key[0], 'uid': key[1], 'question': info['question'],
                         'prediction': row['prediction'], 'ground_truth': ''}
    funnel = args.output_root / 'funnel_public_only.json'
    funnel.write_text(json.dumps({'rows': funnel_rows, 'gold_present': False},
                                ensure_ascii=False, indent=2))
    protocol_path = args.output_root / 'observer_packets.json'
    subprocess.run([sys.executable,
        str(SOURCE / 'methods/CAUSAL_MODALITY_INSTRUMENT/freeze_full_observer_protocol.py'),
        '--funnel', str(funnel), '--output', str(protocol_path)],
        cwd=SOURCE, check=True)
    protocol = json.loads(protocol_path.read_text())['items']
    forced = {(row['video_id'], row['uid']) for row in funnel_rows
              if row['has_forced_full_video']}
    selected = [item for item in protocol
                if (str(item['video_id']), str(item['uid'])) in forced]
    (args.output_root / 'CONDITIONAL_OBSERVER_SELECTION.json').write_text(json.dumps({
        'fixed_rows': 300, 'forced_rows': len(forced),
        'available_local_packets': len(protocol), 'conditional_observer_calls': len(selected)},
        indent=2) + '\n')

    observed = {}
    if selected:
        temp_root = args.output_root / 'observer_frames'
        temp_root.mkdir()
        settings = SimpleNamespace(backend='transformers', model='Qwen2.5-VL-7B-Instruct',
            model_path=args.model_path, device=args.device,
            base_url='http://127.0.0.1:19405/v1', api_key='local-vllm',
            fps=2.0, frames=64, workers=1, timeout=360.0, max_tokens=2,
            representation='images', temp_root=temp_root)
        settings.observer = observer_source.TransformersObserver(
            str(settings.model_path), device=settings.device)
        tasks_by_key = helper.load_tasks(args.parquet)
        with (args.output_root / 'observer_raw.jsonl').open('x') as handle:
            for item in selected:
                key = (str(item['video_id']), str(item['uid']))
                started = time.monotonic()
                try:
                    value = observer_source.run_one(helper=helper, item=item,
                        task=tasks_by_key[key], baseline=baseline[key], args=settings)
                except Exception as exc:
                    value = {'video_id': key[0], 'uid': key[1], 'status': 'error',
                             'error': type(exc).__name__ + ': ' + str(exc)}
                value['wall_seconds'] = time.monotonic() - started
                value['actual_generated_tokens'] = None
                observed[key] = value
                handle.write(json.dumps(value, ensure_ascii=False) + '\n')
                handle.flush()
    final = finalize(funnel_rows, observed, results)
    with (args.output_root / 'FINAL_PREDICTIONS.jsonl').open('x') as handle:
        for row in final:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps({'state': 'COMPLETE', 'fixed_rows': len(final),
        'observer_calls': len(selected),
        'observer_adopted': sum(x['final_source'] == 'DirectObserver' for x in final)}))


if __name__ == '__main__':
    main()
