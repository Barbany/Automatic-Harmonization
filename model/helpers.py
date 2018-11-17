"""Function used to create results folder and save a log of all prints."""

import os
import sys
from model.data_loader import FolderDataset


tag_params = [
    'exp'
    ]


def make_tag(params):
    def to_string(value):
        if isinstance(value, bool):
            return 'T' if value else 'F'
        elif isinstance(value, list):
            return ','.join(map(to_string, value))
        else:
            return str(value)

    return '~'.join(
        key + ':' + to_string(params[key])
        for key in tag_params
        if key not in default_params or params[key] != default_params[key]
    )


def setup_results_dir(params):
    def ensure_dir_exists(path):
        if not os.path.exists(path):
            os.makedirs(path)

    tag = make_tag(params)
    results_path = os.path.abspath(params['results_path'])
    print('results path', results_path)
    ensure_dir_exists(results_path)
    results_path = os.path.join(results_path, tag)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    elif not params['resume']:
        shutil.rmtree(results_path)
        os.makedirs(results_path)

    for subdir in ['checkpoints', 'samples']:
        ensure_dir_exists(os.path.join(results_path, subdir))

    return results_path


def load_last_checkpoint(checkpoints_path):
    checkpoints_pattern = os.path.join(
        checkpoints_path, SaverPlugin.last_pattern.format('*', '*')
    )
    checkpoint_paths = natsorted(glob(checkpoints_pattern))
    if len(checkpoint_paths) > 0:
        checkpoint_path = checkpoint_paths[-1]
        checkpoint_name = os.path.basename(checkpoint_path)
        match = re.match(
            SaverPlugin.last_pattern.format(r'(\d+)', r'(\d+)'),
            checkpoint_name
        )
        epoch = int(match.group(1))
        iteration = int(match.group(2))
        return torch.load(checkpoint_path), epoch, iteration
    else:
        return None


def tee_stdout(log_path):
    log_file = open(log_path, 'a', 1)
    stdout = sys.stdout

    class Tee:
        def write(self, string):
            log_file.write(string)
            stdout.write(string)

        def flush(self):
            log_file.flush()
            stdout.flush()

    sys.stdout = Tee()


def init_random_seed(seed, cuda):
    print('seed', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


def load_model(checkpoint_path):
    model_pattern = '.*ep{}-it{}'

    checkpoint_name = os.path.basename(checkpoint_path)
    match = re.match(
        model_pattern.format(r'(\d+)', r'(\d+)'),
        checkpoint_name
    )
    if match:
        epoch = int(match.group(1))
        iteration = int(match.group(2))
    else:
        epoch, iteration = (0, 0)
                                                                
    return torch.load(checkpoint_path), epoch, iteration


def make_data_loader(overlap_len, params):
    def data_loader(partition):
        dataset = FolderDataset(params['data_path'], params['split_by_phrase'],
                                partition, params['partitions'], params['verbose'])

        return DataLoader(dataset, batch_size=params['batch_size'], shuffle=False, drop_last=True, num_workers=2)
    return data_loader
