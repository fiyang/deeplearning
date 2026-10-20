"""Author's 18-field last-value preprocessing, with the existing held-out split.

We intentionally fit vocabulary on fit_train ONLY, instead of the author's
train+validation pool. Original source feature weights are unused by AITM.py.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from raw_data import RawTable, FIELDS, write_json, counts

AUTHOR_FIELDS = ['101','121','122','124','125','126','127','128','129','205','206','207','216','508','509','702','853','301']
FIELD_INDICES = np.array([FIELDS.index(f) for f in AUTHOR_FIELDS])


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def last_values(part, rows):
    bags = np.asarray(rows, dtype=np.int64)[:, None] * len(FIELDS) + FIELD_INDICES
    ends = part['offsets'][bags + 1].astype(np.int64)
    present = ends > part['offsets'][bags]
    values = np.zeros(bags.shape, dtype=np.uint32)
    values[present] = part['tokens'][ends[present] - 1] // len(FIELDS)
    return values, present


def extract(raw, path):
    common, present = last_values(raw.parts['common'], np.arange(raw.info['common']['rows']))
    result = np.lib.format.open_memmap(path, mode='w+', dtype='uint32', shape=(raw.n, len(AUTHOR_FIELDS)))
    for start in range(0, raw.n, 200000):
        end = min(start + 200000, raw.n)
        sample, _ = last_values(raw.parts['sample'], np.arange(start, end))
        refs = raw.refs[start:end]
        # Matches feat_dict.update(common_feat_dict[common_id]) in official code.
        result[start:end] = np.where(present[refs], common[refs], sample)
    result.flush()
    # Independent dictionary reconstruction from cached sparse records.
    selected = np.unique(np.r_[np.arange(min(32, raw.n)), np.random.default_rng(42).integers(raw.n, size=96)])
    for row in selected:
        expected = {}
        for name, index in [('sample', row), ('common', int(raw.refs[row]))]:
            part = raw.parts[name]
            left, right = part['offsets'][index * len(FIELDS):index * len(FIELDS) + len(FIELDS) + 1:len(FIELDS)]
            for token in part['tokens'][left:right]:
                expected[FIELDS[int(token) % len(FIELDS)]] = int(token) // len(FIELDS)
        np.testing.assert_array_equal(result[row], [expected.get(f, 0) for f in AUTHOR_FIELDS])
    print(f'DENSE_EXTRACTED rows={raw.n} fields={len(AUTHOR_FIELDS)} checked={len(selected)}', flush=True)
    return result


def fit_vocab(dense, rows, minimum=12):
    vocabularies = []
    for i, field in enumerate(AUTHOR_FIELDS):
        frequency = np.bincount(np.asarray(dense[rows, i], dtype=np.int64))
        frequency[0] = 0
        vocab = np.flatnonzero(frequency >= minimum).astype(np.uint32)
        vocabularies.append(vocab)
        print(f'VOCAB field={field} kept={len(vocab)} min_occurrences={minimum}', flush=True)
    return vocabularies


def encode(dense, vocabularies, path):
    lookups = []
    for vocabulary in vocabularies:
        lookup = np.zeros(int(vocabulary[-1]) + 1 if len(vocabulary) else 1, dtype=np.int32)
        lookup[vocabulary] = np.arange(1, len(vocabulary) + 1, dtype=np.int32)
        lookups.append(lookup)
    result = np.lib.format.open_memmap(path, mode='w+', dtype='int32', shape=dense.shape)
    unknown = np.zeros(dense.shape[1], dtype=np.int64)
    for start in range(0, len(dense), 200000):
        values = dense[start:start + 200000]
        chunk = np.zeros(values.shape, dtype=np.int32)
        for i, lookup in enumerate(lookups):
            known = values[:, i] < len(lookup)
            chunk[known, i] = lookup[values[known, i]]
        unknown += (chunk == 0).sum(0)
        result[start:start + len(chunk)] = chunk
    result.flush()
    return dict(zip(AUTHOR_FIELDS, map(int, unknown)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('../data/aliccp-full'))
    parser.add_argument('--reference-run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    train_rows = np.load(args.reference_run / 'train_rows.npy')
    validation_rows = np.load(args.reference_run / 'validation_rows.npy')
    raw = RawTable(args.data_dir / 'cache/train')
    assert len(train_rows) + len(validation_rows) == int((~((raw.labels[:, 0] == 0) & (raw.labels[:, 1] == 1))).sum())
    assert not np.intersect1d(train_rows, validation_rows).size
    dense = extract(raw, args.output / 'train.raw_ids.npy')
    # Author starts a new feature count at zero, then retains counter > 10.
    # Its actual threshold is therefore at least 12 occurrences, not 10.
    vocabularies = fit_vocab(dense, train_rows, minimum=12)
    for field, vocabulary in zip(AUTHOR_FIELDS, vocabularies):
        np.save(args.output / f'vocabulary_{field}.npy', vocabulary)
    mapping = {'train_source': encode(dense, vocabularies, args.output / 'train.ids.npy')}
    manifest = dict(fields=AUTHOR_FIELDS, vocabulary_sizes=[len(v) + 1 for v in vocabularies],
                    reference_run=str(args.reference_run.resolve()),
                    train_rows_sha256=sha256(args.reference_run / 'train_rows.npy'),
                    validation_rows_sha256=sha256(args.reference_run / 'validation_rows.npy'),
                    train=counts(raw.labels, train_rows), validation=counts(raw.labels, validation_rows),
                    train_cache=raw.info, minimum_occurrences=12,
                    weights='Ignored, as in the author single-ID dictionary preprocessing',
                    unknown='Per-field trainable index 0; no hashing',
                    vocabulary_scope='Actual training rows only; validation and test excluded')
    del raw, dense
    raw = RawTable(args.data_dir / 'cache/test')
    dense = extract(raw, args.output / 'test.raw_ids.npy')
    mapping['test_source'] = encode(dense, vocabularies, args.output / 'test.ids.npy')
    test_rows = np.flatnonzero(~((raw.labels[:, 0] == 0) & (raw.labels[:, 1] == 1))).astype(np.int32)
    np.save(args.output / 'test_rows.npy', test_rows)
    manifest.update(test=counts(raw.labels, test_rows), test_cache=raw.info, unknown_rows_by_field=mapping,
                    test_rows_sha256=sha256(args.output / 'test_rows.npy'), status='passed')
    write_json(args.output / 'manifest.json', manifest)
    print('PREPARATION_COMPLETE ' + json.dumps(manifest['vocabulary_sizes']), flush=True)


if __name__ == '__main__':
    main()
