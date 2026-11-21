"""Evaluate JSONL LLM answers using strict exact-match and reference-based lexical overlap."""
import argparse
import json
from pathlib import Path
import re
from statistics import mean


def normalize(text):
    return ' '.join(re.findall(r'[a-z0-9]+', text.lower()))


def token_f1(reference, candidate):
    """Bag-of-words overlap; only a proxy, not factual entailment."""
    from collections import Counter
    truth, pred = Counter(normalize(reference).split()), Counter(normalize(candidate).split())
    common = sum((truth & pred).values())
    if not truth and not pred:
        return 1.0
    if not common:
        return 0.0
    precision, recall = common / sum(pred.values()), common / sum(truth.values())
    return 2 * precision * recall / (precision + recall)


def read_jsonl(path):
    rows = []
    for n, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f'Invalid JSON on line {n}: {exc}') from exc
        for key in ('id', 'question', 'reference_answer', 'model_answer', 'context'):
            if key not in item or not isinstance(item[key], str):
                raise ValueError(f'Line {n} missing string field {key!r}')
        rows.append(item)
    if not rows:
        raise ValueError('Input JSONL contains no examples')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate example ids')
    return rows


def score(rows):
    details = []
    for row in rows:
        ref, ans, context = row['reference_answer'], row['model_answer'], row['context']
        details.append({'id': row['id'],
                        'exact_match': int(normalize(ref) == normalize(ans)),
                        'token_f1': round(token_f1(ref, ans), 4),
                        'reference_in_context': bool(normalize(ref) and normalize(ref) in normalize(context)),
                        'answer_in_context': bool(normalize(ans) and normalize(ans) in normalize(context))})
    return {'n': len(details),
            'exact_match_rate': round(mean(d['exact_match'] for d in details), 4),
            'mean_token_f1': round(mean(d['token_f1'] for d in details), 4),
            'answer_contained_in_context_rate': round(mean(d['answer_in_context'] for d in details), 4),
            'examples': details}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', nargs='?', type=Path, default=Path(__file__).parent / 'sample_predictions.jsonl')
    parser.add_argument('--out', type=Path, default=Path('evaluation_report.json'))
    args = parser.parse_args()
    metrics = score(read_jsonl(args.input))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in metrics.items() if key != 'examples'}, indent=2))
