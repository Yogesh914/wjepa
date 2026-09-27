"""WHAR raw-session data loading for Wavelet W-JEPA.

Windows are sliced on the fly from each dataset's raw ``sessions/sessions.parquet``.
Pretraining mixes the standard six-channel view across the source datasets;
fine-tuning uses all configured channels of a single benchmark.
"""
import os
from itertools import combinations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from config import WHAR_DATASET_IDS, canonical_dataset_name, get_dataset_config


_PARTICIPANT_SPLIT_CACHE = {}


def _python_scalar(value):
    """Convert NumPy participant IDs to JSON-friendly Python scalars."""
    return value.item() if isinstance(value, np.generic) else value


class WHARRawWindow_JEPA(Dataset):
    """Fixed-length windows sliced from raw session parquet."""

    def __init__(self, data_path, dataset_name, split='train', return_labels=True,
                 split_seed=42, train_ratio=0.2, window_size=500, window_stride=500,
                 channel_subset=None, use_augmentation=False, sampling_rate=None,
                 split_strategy='window'):
        self.dataset_name = canonical_dataset_name(dataset_name)
        self.split = split
        self.return_labels = return_labels
        self.window_size = int(window_size)
        self.window_stride = int(window_stride)
        self.sampling_rate = float(sampling_rate) if sampling_rate is not None else None
        if split_strategy not in {'window', 'participant'}:
            raise ValueError(f"Unknown split strategy: {split_strategy}")
        self.split_strategy = split_strategy
        if self.sampling_rate is not None and self.sampling_rate <= 0:
            raise ValueError(f"sampling_rate must be positive, got {sampling_rate}")

        if use_augmentation:
            from .augmentation import finetune_aug
            self.augmentation_fn = finetune_aug
        else:
            self.augmentation_fn = None
        self.use_augmentation = use_augmentation

        cfg = get_dataset_config(self.dataset_name, channel_subset=channel_subset)
        self.channel_names = cfg['channels']
        self.drop_activity_ids = set(cfg['drop_activity_ids'])
        self.num_channels = len(self.channel_names)

        dataset_dirname = WHAR_DATASET_IDS.get(self.dataset_name, self.dataset_name)
        if os.path.basename(os.path.normpath(data_path)) == dataset_dirname:
            self.dataset_dir = data_path
        else:
            self.dataset_dir = os.path.join(data_path, dataset_dirname)

        sessions_path = os.path.join(self.dataset_dir, 'sessions', 'sessions.parquet')
        session_df_csv_path = os.path.join(self.dataset_dir, 'metadata', 'session_df.csv')
        session_df_parquet_path = os.path.join(
            self.dataset_dir, 'metadata', 'session_df.parquet')
        if not os.path.exists(sessions_path):
            raise FileNotFoundError(f"Session parquet not found: {sessions_path}")

        if os.path.exists(session_df_csv_path):
            session_df = pd.read_csv(session_df_csv_path, index_col=0)
        elif os.path.exists(session_df_parquet_path):
            session_df = pd.read_parquet(session_df_parquet_path)
        else:
            raise FileNotFoundError(
                "Session metadata not found; expected one of: "
                f"{session_df_csv_path}, {session_df_parquet_path}"
            )
        meta = session_df.dropna(subset=['subject_id', 'activity_id']).copy()
        if self.drop_activity_ids:
            meta = meta[~meta['activity_id'].isin(self.drop_activity_ids)]

        label_values = sorted(meta['activity_id'].unique().tolist())
        self.label_map = {activity_id: idx for idx, activity_id in enumerate(label_values)}
        meta['label'] = meta['activity_id'].map(self.label_map).astype(np.int64)
        meta = meta.set_index('session_id', drop=False)
        self.num_classes = len(label_values)

        parquet_columns = self.channel_names + ['session_id']
        if self.sampling_rate is not None:
            parquet_columns.append('timestamp')
        sessions = pd.read_parquet(sessions_path, columns=parquet_columns)
        missing = [c for c in self.channel_names if c not in sessions.columns]
        if missing:
            raise ValueError(f"{self.dataset_name} is missing requested channels: {missing}")

        sessions['session_id'] = sessions['session_id'].astype(meta.index.dtype, copy=False)
        sessions = sessions[sessions['session_id'].isin(set(meta.index.tolist()))].reset_index(drop=True)

        samples, labels, subjects = [], [], []
        value_chunks = []
        value_offset = 0
        grouped = sessions.groupby('session_id', sort=False)
        for session_id, session in grouped:
            if session_id not in meta.index:
                continue

            if self.sampling_rate is None:
                session_values = session[self.channel_names].to_numpy(
                    dtype=np.float32, copy=True)
            else:
                session_values = self._resample_session(session)

            session_length = len(session_values)
            if session_length < self.window_size:
                continue
            session_meta = meta.loc[session_id]
            label = int(session_meta['label'])
            subject = session_meta['subject_id']
            # These float32 session arrays are owned by the dataset. Clean once
            # before slicing; doing the identical conversion in every window
            # repeats scans and temporary masks on every training epoch.
            value_chunks.append(np.nan_to_num(session_values, copy=False))
            last_start = session_length - self.window_size
            for start in range(0, last_start + 1, self.window_stride):
                samples.append(
                    range(
                        value_offset + start,
                        value_offset + start + self.window_size,
                    )
                )
                labels.append(label)
                subjects.append(subject)
            value_offset += session_length

        if not value_chunks:
            raise ValueError(
                f"{self.dataset_name} has no sessions with at least "
                f"{self.window_size} samples after resampling"
            )
        self.values = np.concatenate(value_chunks, axis=0)

        all_labels = np.asarray(labels, dtype=np.int64)
        all_subjects = np.asarray(subjects)
        # Keep lightweight window bounds so another split can reuse the same
        # raw/resampled values instead of rereading and resampling every session.
        self._all_group_indices = samples
        self._all_labels = all_labels
        self._all_subjects = all_subjects
        self._split_options = dict(split_seed=split_seed, train_ratio=train_ratio,
                                   split_strategy=split_strategy)
        selected = self._select_indices(all_labels, all_subjects, split=split,
                                        **self._split_options)

        self.group_indices = [samples[int(idx)] for idx in selected]
        self.labels = np.asarray([labels[int(idx)] for idx in selected], dtype=np.int64)
        self.subjects = all_subjects[selected]
        self.participants = [
            _python_scalar(value) for value in np.unique(self.subjects)
        ]

        print(f"{cfg['display_name']} {split} raw split: {len(self.group_indices)} windows, "
              f"{self.num_classes} classes, {self.num_channels} channels, "
              f"{len(self.participants)} participants, split_strategy={self.split_strategy}, "
              f"sampling_rate={self.sampling_rate or 'native'}Hz, "
              f"window={self.window_size}, stride={self.window_stride}, aug={self.use_augmentation}")

    def evaluation_split(self, split='test'):
        """An unaugmented split sharing the already prepared raw session array."""
        import copy
        dataset = copy.copy(self)
        selected = self._select_indices(self._all_labels, self._all_subjects,
                                        split=split, **self._split_options)
        dataset.split = split
        dataset.group_indices = [self._all_group_indices[int(i)] for i in selected]
        dataset.labels = self._all_labels[selected]
        dataset.subjects = self._all_subjects[selected]
        dataset.participants = [_python_scalar(v) for v in np.unique(dataset.subjects)]
        dataset.use_augmentation = False
        dataset.augmentation_fn = None
        print(f"{self.dataset_name} {split}: {len(dataset)} windows, "
              f"{len(dataset.participants)} participants (shared session values)")
        return dataset

    def _resample_session(self, session):
        """Interpolate one continuous session onto an exact fixed-rate time grid."""
        frame = session[['timestamp'] + self.channel_names].copy()
        frame['timestamp'] = pd.to_datetime(frame['timestamp'], errors='coerce')
        frame = frame.dropna(subset=['timestamp']).sort_values('timestamp')
        frame = frame.drop_duplicates(subset=['timestamp'], keep='first')
        if frame.empty:
            return np.empty((0, self.num_channels), dtype=np.float32)

        period_ns = int(round(1_000_000_000 / self.sampling_rate))
        rule = pd.Timedelta(period_ns, unit='ns')
        values = frame.set_index('timestamp')[self.channel_names]
        values = values.resample(rule).mean().interpolate(method='linear')
        return values.to_numpy(dtype=np.float32, copy=True)

    def _select_indices(self, labels, subjects, split, split_seed, train_ratio,
                        split_strategy):
        """Select a deterministic window- or participant-level train/test split."""
        if split_strategy == 'participant':
            return self._select_participant_indices(
                labels, subjects, split=split, split_seed=split_seed,
                train_ratio=train_ratio)

        rng = np.random.default_rng(split_seed)
        train_indices, test_indices = [], []
        for label in sorted(set(labels.tolist())):
            class_indices = np.flatnonzero(labels == label)
            rng.shuffle(class_indices)
            if len(class_indices) <= 1:
                n_train = len(class_indices)
            else:
                n_train = max(1, int(round(len(class_indices) * train_ratio)))
                n_train = min(n_train, len(class_indices) - 1)
            train_indices.extend(class_indices[:n_train].tolist())
            test_indices.extend(class_indices[n_train:].tolist())

        selected = np.asarray(test_indices if split in {'test', 'val'} else train_indices, dtype=np.int64)
        rng.shuffle(selected)
        return selected

    def _select_participant_indices(self, labels, subjects, split, split_seed,
                                    train_ratio):
        """Choose disjoint participants while approximating the window ratio.

        The participant count follows ``train_ratio``. Among subsets of that
        size, selection first maximizes class coverage across both splits, then
        minimizes the difference from the requested number of train windows,
        and finally minimizes class-distribution drift. Seeded ordering makes
        otherwise equivalent choices reproducible.
        """
        participants = np.unique(subjects)
        if len(participants) < 2:
            raise ValueError(
                f"Participant splitting requires at least two participants, got "
                f"{len(participants)} for {self.dataset_name}"
            )

        n_train_participants = int(round(len(participants) * train_ratio))
        n_train_participants = min(max(n_train_participants, 1), len(participants) - 1)
        num_classes = self.num_classes
        participant_indices = [np.flatnonzero(subjects == value) for value in participants]
        participant_counts = np.asarray([len(indices) for indices in participant_indices])
        participant_class_counts = np.stack([
            np.bincount(labels[indices], minlength=num_classes)
            for indices in participant_indices
        ])
        total_class_counts = participant_class_counts.sum(axis=0)
        summary = tuple(
            (
                repr(_python_scalar(participant)),
                int(participant_counts[index]),
                tuple(int(value) for value in participant_class_counts[index]),
            )
            for index, participant in enumerate(participants)
        )
        cache_key = (
            self.dataset_name, summary, n_train_participants,
            float(train_ratio), int(split_seed),
        )

        train_participant_indices = _PARTICIPANT_SPLIT_CACHE.get(cache_key)
        if train_participant_indices is None:
            rng = np.random.default_rng(split_seed)
            random_order = rng.permutation(len(participants))
            tie_rank = np.empty(len(participants), dtype=np.int64)
            tie_rank[random_order] = np.arange(len(participants))
            target_train_windows = len(labels) * train_ratio
            best_numeric_score = None
            best_tie_score = None
            best_candidate = None

            for candidate in combinations(
                    range(len(participants)), n_train_participants):
                candidate_array = np.asarray(candidate, dtype=np.int64)
                train_count = int(participant_counts[candidate_array].sum())
                train_class_counts = participant_class_counts[candidate_array].sum(axis=0)
                test_class_counts = total_class_counts - train_class_counts
                missing_classes = int(
                    np.count_nonzero(train_class_counts == 0)
                    + np.count_nonzero(test_class_counts == 0)
                )
                sample_error = abs(train_count - target_train_windows)
                train_distribution = train_class_counts / max(train_count, 1)
                overall_distribution = total_class_counts / len(labels)
                distribution_error = float(
                    np.abs(train_distribution - overall_distribution).sum())
                numeric_score = (missing_classes, sample_error, distribution_error)
                tie_score = tuple(sorted(int(tie_rank[index]) for index in candidate))
                if (best_numeric_score is None
                        or numeric_score < best_numeric_score
                        or (numeric_score == best_numeric_score
                            and tie_score < best_tie_score)):
                    best_numeric_score = numeric_score
                    best_tie_score = tie_score
                    best_candidate = candidate

            train_participant_indices = tuple(best_candidate)
            _PARTICIPANT_SPLIT_CACHE[cache_key] = train_participant_indices

        train_participants = participants[list(train_participant_indices)]
        train_mask = np.isin(subjects, train_participants)
        selected = np.flatnonzero(~train_mask if split in {'test', 'val'} else train_mask)
        rng = np.random.default_rng(split_seed)
        rng.shuffle(selected)
        return selected

    def __len__(self):
        return len(self.group_indices)

    def __getitem__(self, idx):
        window = self.group_indices[idx]
        # Windows are contiguous; a range stores two bounds instead of 500
        # int64 indices, and a slice avoids an extra advanced-indexing copy.
        rows = self.values[window.start:window.stop].copy()
        if self.use_augmentation and self.augmentation_fn is not None:
            rows = self.augmentation_fn(rows)
        x = torch.from_numpy(rows).float().transpose(0, 1)  # [C, T]
        x = x.unsqueeze(0)  # [1, C, T]
        if self.return_labels:
            return x, torch.tensor(self.labels[idx]).long()
        return x


class WHARCombinedPretrain_JEPA(Dataset):
    """Concatenates the standard six-channel windows from several source datasets."""

    def __init__(self, data_path, dataset_names, split_seed=42, train_ratio=1.0,
                 budget=None, window_size=500, window_stride=500, sampling_rate=None):
        self.dataset_names = [canonical_dataset_name(n) for n in dataset_names]
        self.budget = budget
        self.seed = split_seed
        self.epoch = 0

        self.datasets = []
        for name in self.dataset_names:
            dataset = WHARRawWindow_JEPA(
                data_path=data_path, dataset_name=name, split='train',
                return_labels=False, split_seed=split_seed, train_ratio=train_ratio,
                window_size=window_size, window_stride=window_stride,
                channel_subset='standard', sampling_rate=sampling_rate)
            if len(dataset) > 0:
                self.datasets.append(dataset)
        if not self.datasets:
            raise ValueError("No WHAR datasets were available for combined pretraining")

        self.num_channels = self.datasets[0].num_channels
        if any(d.num_channels != self.num_channels for d in self.datasets):
            counts = {d.dataset_name: d.num_channels for d in self.datasets}
            raise ValueError(f"WHAR-All datasets must share a channel count, got {counts}")
        self.total_available = sum(len(d) for d in self.datasets)
        self.indices = self._make_epoch_indices()
        print(f"WHAR-All train split: {self.total_available} source windows from "
              f"{len(self.datasets)} datasets; epoch length {len(self.indices)}")

    def _make_epoch_indices(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        if self.budget is None or self.budget >= self.total_available:
            indices = [(d_idx, s_idx) for d_idx, d in enumerate(self.datasets)
                       for s_idx in range(len(d))]
            rng.shuffle(indices)
            return indices

        budget = int(self.budget)
        per_dataset = budget // len(self.datasets)
        remainder = budget % len(self.datasets)
        indices = []
        for order_idx, dataset in enumerate(self.datasets):
            n_samples = per_dataset + int(order_idx < remainder)
            if n_samples <= 0:
                continue
            sample_indices = rng.choice(len(dataset), size=n_samples, replace=n_samples > len(dataset))
            indices.extend((order_idx, int(s)) for s in sample_indices)
        rng.shuffle(indices)
        return indices

    def resample_epoch(self):
        self.epoch += 1
        self.indices = self._make_epoch_indices()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        dataset_idx, sample_idx = self.indices[idx]
        return self.datasets[dataset_idx][sample_idx]


def load_whar_combined_dataset(data_path, batch_size, dataset_names, split_seed=42,
                               train_ratio=1.0, budget=None, window_size=500,
                               window_stride=500, num_workers=4, sampling_rate=None):
    dataset = WHARCombinedPretrain_JEPA(
        data_path=data_path, dataset_names=dataset_names, split_seed=split_seed,
        train_ratio=train_ratio, budget=budget, window_size=window_size,
        window_stride=window_stride, sampling_rate=sampling_rate)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=torch.cuda.is_available(), drop_last=False)
    return loader, dataset


def load_whar_dataset(data_path, dataset_name, batch_size, split='train', return_labels=True,
                      split_seed=42, train_ratio=0.2, window_size=500, window_stride=500,
                      num_workers=4, shuffle=True, channel_subset=None, use_augmentation=False,
                      sampling_rate=None, split_strategy='window'):
    dataset = WHARRawWindow_JEPA(
        data_path=data_path, dataset_name=dataset_name, split=split,
        return_labels=return_labels, split_seed=split_seed, train_ratio=train_ratio,
        window_size=window_size, window_stride=window_stride, channel_subset=channel_subset,
        use_augmentation=use_augmentation, sampling_rate=sampling_rate,
        split_strategy=split_strategy)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                        pin_memory=torch.cuda.is_available(), drop_last=False)
    return loader, dataset


def get_whar_dataloaders(data_path, dataset_name, batch_size, return_labels=True,
                         split_seed=42, train_ratio=0.2, window_size=500, window_stride=500,
                         num_workers=4, channel_subset=None, use_augmentation=False,
                         sampling_rate=None, split_strategy='window'):
    train_loader, train_dataset = load_whar_dataset(
        data_path=data_path, dataset_name=dataset_name, batch_size=batch_size, split='train',
        return_labels=return_labels, split_seed=split_seed, train_ratio=train_ratio,
        window_size=window_size, window_stride=window_stride, num_workers=num_workers,
        shuffle=True, channel_subset=channel_subset, use_augmentation=use_augmentation,
        sampling_rate=sampling_rate, split_strategy=split_strategy)
    test_dataset = train_dataset.evaluation_split()
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                            drop_last=False)
    return train_loader, test_loader, train_dataset, test_dataset
