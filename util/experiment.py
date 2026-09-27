"""Optional experiment seeding and resumable W&B reporting."""
import json
import os
from pathlib import Path
import random
import uuid

import numpy as np
import torch
import wandb

from util.checkpointing import write_json


def start_run(args, stage, default_name):
    record_path = Path(args.output_dir) / 'wandb_run.json'
    previous = json.loads(record_path.read_text()) if record_path.exists() else {}
    run_id = previous.get('id') or os.environ.get('WANDB_RUN_ID') or uuid.uuid4().hex
    run = wandb.init(
        project=args.wandb_project, entity=args.wandb_entity,
        config=vars(args), name=os.environ.get('WANDB_NAME', default_name),
        mode=args.wandb_mode, id=run_id, resume='allow',
    )
    if args.wandb_mode != 'disabled':
        run.define_metric('global_step')
        run.define_metric('step/*', step_metric='global_step')
        run.define_metric('epoch')
        for pattern in ('epoch/*', 'train/*', 'test/*'):
            run.define_metric(pattern, step_metric='epoch')
        run.config.update({'stage': stage})
        write_json(record_path, {'id': run.id, 'url': run.url,
                                'entity': run.entity, 'project': run.project})

    # Initialize training RNGs after the logging client has started. A resumed
    # checkpoint restores its saved RNG state later, as usual.
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)


def report_progress(args, latest):
    if args.wandb_mode == 'disabled':
        return
    values = {'epochs_completed': latest['epoch'],
              'complete': latest['epoch'] == args.epochs,
              'selection_criterion': 'fixed_final_epoch'}
    for name in ('accuracy', 'macro_f1', 'loss'):
        if name in latest:
            values[name] = latest[name]
    wandb.run.summary.update(values)
