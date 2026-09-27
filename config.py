"""Configuration for the Wavelet-JEPA WHAR pipeline.

A single base-size encoder is pretrained with a db4 level-4 wavelet tokenizer on the
standard six-channel (acc+gyro) view of the benchmark datasets, then fine-tuned on
each benchmark with all configured channels.
"""

# --- Encoder / predictor (base size; the only configuration used) ---
BASE_MODEL_CONFIG = {
    "embed_dim": 384,
    "depth": 8,
    "num_heads": 6,
    "mlp_ratio": 4.0,
    "predictor_embed_dim": 192,
    "predictor_depth": 4,
    "predictor_num_heads": 6,
}

# --- Wavelet tokenizer ---
WAVELET_CONFIG = {
    "wavelet": "db4",
    "wavelet_level": 4,
    "wavelet_tokens_per_level": 2,
}

WHAR_WINDOW_LENGTH = 500

# --- Default optimization hyper-parameters ---
PRETRAIN_CONFIG = {
    "batch_size": 256,
    "epochs": 50,
    "warmup_epochs": 8,
    "lr": 2e-4,
    "min_lr": 1e-6,
    "weight_decay": 0.05,
}

FINETUNE_CONFIG = {
    "batch_size": 64,
    "epochs": 70,
    "warmup_epochs": 10,
    "lr": 1e-4,
    "min_lr": 1e-6,
    "weight_decay": 0.05,
}


# --- Downstream benchmark channel layouts ---
PAMAP2_POSITIONS = ["hand", "chest", "ankle"]
PAMAP2_ACCEL_CHANNELS = [
    f"{position}_{sensor}_{axis}"
    for position in PAMAP2_POSITIONS
    for sensor in ("acc", "acc_2")
    for axis in ("x", "y", "z")
]
PAMAP2_GYRO_CHANNELS = [
    f"{position}_gyro_{axis}"
    for position in PAMAP2_POSITIONS
    for axis in ("x", "y", "z")
]
PAMAP2_STANDARD_CHANNELS = [
    "hand_acc_x", "hand_acc_y", "hand_acc_z",
    "hand_gyro_x", "hand_gyro_y", "hand_gyro_z",
]

DSADS_POSITIONS = ["T", "RA", "LA", "RL", "LL"]
DSADS_ACCEL_CHANNELS = [
    f"{position}_{axis}acc"
    for position in DSADS_POSITIONS
    for axis in ("x", "y", "z")
]
DSADS_GYRO_CHANNELS = [
    f"{position}_{axis}gyro"
    for position in DSADS_POSITIONS
    for axis in ("x", "y", "z")
]
DSADS_STANDARD_CHANNELS = [
    "RA_xacc", "RA_yacc", "RA_zacc",
    "RA_xgyro", "RA_ygyro", "RA_zgyro",
]

MHEALTH_ACCEL_CHANNELS = [
    "chest_acc_x", "chest_acc_y", "chest_acc_z",
    "lankle_acc_x", "lankle_acc_y", "lankle_acc_z",
    "rarm_acc_x", "rarm_acc_y", "rarm_acc_z",
]
MHEALTH_GYRO_CHANNELS = [
    "lankle_gyro_x", "lankle_gyro_y", "lankle_gyro_z",
    "rarm_gyro_x", "rarm_gyro_y", "rarm_gyro_z",
]
MHEALTH_STANDARD_CHANNELS = [
    "rarm_acc_x", "rarm_acc_y", "rarm_acc_z",
    "rarm_gyro_x", "rarm_gyro_y", "rarm_gyro_z",
]

REALWORLD_POSITIONS = ["chest", "forearm", "head", "shin", "thigh", "upperarm", "waist"]
REALWORLD_ACCEL_CHANNELS = [
    f"acc_{position}_{axis}"
    for position in REALWORLD_POSITIONS
    for axis in ("x", "y", "z")
]
REALWORLD_GYRO_CHANNELS = [
    f"gyr_{position}_{axis}"
    for position in REALWORLD_POSITIONS
    for axis in ("x", "y", "z")
]
REALWORLD_STANDARD_CHANNELS = [
    "acc_waist_x", "acc_waist_y", "acc_waist_z",
    "gyr_waist_x", "gyr_waist_y", "gyr_waist_z",
]

UCIHAR_ACCEL_CHANNELS = ["total_acc_x", "total_acc_y", "total_acc_z"]
UCIHAR_GYRO_CHANNELS = ["body_gyro_x", "body_gyro_y", "body_gyro_z"]
UCIHAR_STANDARD_CHANNELS = UCIHAR_ACCEL_CHANNELS + UCIHAR_GYRO_CHANNELS

USCHAD_ACCEL_CHANNELS = ["Ax", "Ay", "Az"]
USCHAD_GYRO_CHANNELS = ["GyroX", "GyroY", "GyroZ"]
USCHAD_STANDARD_CHANNELS = USCHAD_ACCEL_CHANNELS + USCHAD_GYRO_CHANNELS


def _benchmark(display_name, accel, gyro, standard, num_classes,
               drop_activity_ids, aliases=None):
    channels = accel + gyro
    return {
        "display_name": display_name,
        "aliases": aliases or [],
        "num_channels": len(channels),
        "sequence_length": WHAR_WINDOW_LENGTH,
        "num_classes": num_classes,
        "channels": channels,
        "standard_channels": standard,
        "drop_activity_ids": drop_activity_ids,
        **WAVELET_CONFIG,
    }


WHAR_BENCHMARK_CONFIGS = {
    "pamap2": _benchmark("PAMAP2", PAMAP2_ACCEL_CHANNELS, PAMAP2_GYRO_CHANNELS,
                         PAMAP2_STANDARD_CHANNELS, 12, [0]),
    "dsads": _benchmark("DSADS", DSADS_ACCEL_CHANNELS, DSADS_GYRO_CHANNELS,
                        DSADS_STANDARD_CHANNELS, 19, []),
    "mhealth": _benchmark("MHealth", MHEALTH_ACCEL_CHANNELS, MHEALTH_GYRO_CHANNELS,
                          MHEALTH_STANDARD_CHANNELS, 12, [0]),
    "realworld": _benchmark("Realworld", REALWORLD_ACCEL_CHANNELS, REALWORLD_GYRO_CHANNELS,
                            REALWORLD_STANDARD_CHANNELS, 8, [], aliases=["real_world"]),
    "uci_har": _benchmark("UCI-HAR", UCIHAR_ACCEL_CHANNELS, UCIHAR_GYRO_CHANNELS,
                          UCIHAR_STANDARD_CHANNELS, 6, [], aliases=["ucihar", "uci-har"]),
    "usc_had": _benchmark("USC-HAD", USCHAD_ACCEL_CHANNELS, USCHAD_GYRO_CHANNELS,
                          USCHAD_STANDARD_CHANNELS, 12, [], aliases=["usc-had"]),
}


# Fine-tuning targets, and the default pretraining sources (standard six channels).
BENCHMARK_DATASETS = tuple(WHAR_BENCHMARK_CONFIGS)

# whar-datasets identifiers, which are also the directory names under the data
# path, where they differ from the dataset names used here.
WHAR_DATASET_IDS = {"realworld": "real_world"}

WHAR_ALL_CONFIG = {
    "aliases": ["whar_combined", "combined_whar", "all_whar"],
    "display_name": "WHAR-All",
    "num_channels": 6,
    "sequence_length": WHAR_WINDOW_LENGTH,
    "num_classes": 0,
    "channels": ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
    "standard_channels": ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
    **WAVELET_CONFIG,
}


def canonical_dataset_name(name):
    name = name.lower().replace("-", "_")
    if name == "real_world":
        return "realworld"
    if name == "whar_all" or name in WHAR_ALL_CONFIG["aliases"]:
        return "whar_all"
    for key, cfg in WHAR_BENCHMARK_CONFIGS.items():
        if name == key or name in cfg.get("aliases", []):
            return key
    return name


def get_dataset_config(dataset, channel_subset=None):
    """Return a model/data config for ``whar_all`` or a benchmark dataset.

    ``channel_subset`` selects ``standard`` (six channels) or ``all`` (every
    configured channel, the default ``None``). The wavelet token count is derived
    from the selected channel count.
    """
    if channel_subset not in {None, "standard", "all"}:
        raise ValueError(f"Unknown channel subset: {channel_subset}")

    dataset = canonical_dataset_name(dataset)
    if dataset == "whar_all":
        cfg = dict(WHAR_ALL_CONFIG)
    elif dataset in WHAR_BENCHMARK_CONFIGS:
        cfg = dict(WHAR_BENCHMARK_CONFIGS[dataset])
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if channel_subset == "standard":
        cfg["channels"] = list(cfg["standard_channels"])
    else:
        cfg["channels"] = list(cfg["channels"])
    cfg["num_channels"] = len(cfg["channels"])
    cfg["num_patches"] = (
        (cfg["wavelet_level"] + 1) * cfg["wavelet_tokens_per_level"] * cfg["num_channels"]
    )
    return cfg
