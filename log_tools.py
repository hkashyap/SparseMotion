# Based on https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/utils.py
import torch
from path import Path
import shutil
from collections import OrderedDict


def save_checkpoint(save_path, mfg_state, is_best, filename='checkpoint.pth.tar'):
    file_prefixes = ['mfg_net']
    states = [mfg_state]
    for (prefix, state) in zip(file_prefixes, states):
        torch.save(state, save_path/'{}_{}'.format(prefix, filename))

        if state['epoch'] % 5 == 0:
            torch.save(state, save_path / '{}_{}_{}'.format(prefix, state['epoch'], filename))

    if is_best:
        print('saving best ...')
        for prefix in file_prefixes:
            shutil.copyfile(save_path / '{}_{}'.format(prefix, filename), save_path / '{}_best.pth.tar'.format(prefix))


def save_checkpoint_ccp(save_path, mfg_state, is_best, filename='cp.pth.tar'):
    file_prefixes = ['mfg_net']
    states = [mfg_state]
    for (prefix, state) in zip(file_prefixes, states):
        torch.save(state, save_path / '{}_{}_{}'.format(prefix, state['epoch'], filename))

    if is_best:
        print('saving best ...')
        for prefix in file_prefixes:
            shutil.copyfile(save_path / '{}_{}_{}'.format(prefix, state['epoch'], filename),
                            save_path / '{}_best.pth.tar'.format(prefix))


def save_checkpoint_ccp_flow(save_path, mfg_state, is_best, filename='cp.pth.tar'):
    file_prefixes = ['flow_net']
    states = [mfg_state]
    for (prefix, state) in zip(file_prefixes, states):
        torch.save(state, save_path / '{}_{}_{}'.format(prefix, state['epoch'], filename))

    if is_best:
        print('saving best ...')
        for prefix in file_prefixes:
            shutil.copyfile(save_path / '{}_{}_{}'.format(prefix, state['epoch'], filename),
                            save_path / '{}_best.pth.tar'.format(prefix))


def save_path_formatter(args, parser):
    def is_default(key, value):
        return value == parser.get_default(key)
    args_dict = vars(args)
    data_folder_name = str(Path(args_dict['data_dir']).normpath().name)
    folder_string = [data_folder_name]
    if not is_default('epochs', args_dict['epochs']):
        folder_string.append('{}epochs'.format(args_dict['epochs']))
    keys_with_prefix = OrderedDict()
    keys_with_prefix['seq_len'] = 'seq'
    keys_with_prefix['shuffle'] = 'shuffle'
    keys_with_prefix['microbp'] = 'microbp'
    keys_with_prefix['batch_size'] = 'b'
    keys_with_prefix['lr'] = 'lr'
    keys_with_prefix['w_mft'] = 'w_mft'
    keys_with_prefix['w_mfw'] = 'w_mfw'
    keys_with_prefix['w_sparse'] = 'w_s'
    # keys_with_prefix['w_cont'] = 'w_c'
    keys_with_prefix['momentum'] = 'm'
    # keys_with_prefix['opt_algo'] = 'algo'

    for key, prefix in keys_with_prefix.items():
        value = args_dict[key]
        if not is_default(key, value):
            folder_string.append('{}{}'.format(prefix, value))
    save_path = Path('_'.join(folder_string))
    return save_path


def save_path_formatter_L0L1L2(args, parser):
    def is_default(key, value):
        return value == parser.get_default(key)
    args_dict = vars(args)
    data_folder_name = str(Path(args_dict['data_dir']).normpath().name)
    folder_string = [data_folder_name]
    if not is_default('epochs', args_dict['epochs']):
        folder_string.append('{}epochs'.format(args_dict['epochs']))
    keys_with_prefix = OrderedDict()
    keys_with_prefix['seq_len'] = 'seq'
    keys_with_prefix['shuffle'] = 'shuffle'
    keys_with_prefix['microbp'] = 'microbp'
    keys_with_prefix['batch_size'] = 'b'
    keys_with_prefix['lr'] = 'lr'
    keys_with_prefix['w_mft'] = 'w_mft'
    keys_with_prefix['w_mfw'] = 'w_mfw'
    keys_with_prefix['w_L0'] = 'wL0'
    keys_with_prefix['w_L1'] = 'wL1'
    keys_with_prefix['w_L2'] = 'wL2'
    # keys_with_prefix['w_cont'] = 'w_c'
    keys_with_prefix['momentum'] = 'm'
    # keys_with_prefix['opt_algo'] = 'algo'

    for key, prefix in keys_with_prefix.items():
        value = args_dict[key]
        if not is_default(key, value):
            folder_string.append('{}{}'.format(prefix, value))
    save_path = Path('_'.join(folder_string))
    return save_path


def save_path_formatter_direct_eval(args, parser):
    def is_default(key, value):
        return value == parser.get_default(key)
    args_dict = vars(args)
    data_folder_name = str(Path(args_dict['data_dir']).normpath().name)
    folder_string = [data_folder_name]
    keys_with_prefix = OrderedDict()
    # keys_with_prefix['seq_len'] = 'seq'
    # keys_with_prefix['algo'] = 'algo'

    for key, prefix in keys_with_prefix.items():
        value = args_dict[key]
        if not is_default(key, value):
            folder_string.append('{}{}'.format(prefix, value))
    save_path = Path('_'.join(folder_string))
    return save_path


def save_path_formatter_vkitti(args, parser):
    def is_default(key, value):
        return value == parser.get_default(key)
    args_dict = vars(args)
    data_folder_name = str(Path(args_dict['data_dir']).normpath().name)
    folder_string = [data_folder_name]
    if not is_default('epochs', args_dict['epochs']):
        folder_string.append('{}epochs'.format(args_dict['epochs']))
    keys_with_prefix = OrderedDict()
    keys_with_prefix['shuffle'] = 'shuffle'
    # keys_with_prefix['microbp'] = 'microbp'
    keys_with_prefix['batch_size'] = 'b'
    keys_with_prefix['lr'] = 'lr'
    keys_with_prefix['w_mft'] = 'w_mft'
    keys_with_prefix['w_mfw'] = 'w_mfw'
    keys_with_prefix['w_sparse'] = 'w_s'
    keys_with_prefix['momentum'] = 'm'
    keys_with_prefix['opt_algo'] = 'algo'

    for key, prefix in keys_with_prefix.items():
        value = args_dict[key]
        if not is_default(key, value):
            folder_string.append('{}{}'.format(prefix, value))
    save_path = Path('_'.join(folder_string))
    return save_path


def save_path_formatter_sintel(args, parser):
    def is_default(key, value):
        return value == parser.get_default(key)
    args_dict = vars(args)
    data_folder_name = str(Path(args_dict['data_dir']).normpath().name)
    folder_string = [data_folder_name]
    if not is_default('epochs', args_dict['epochs']):
        folder_string.append('{}epochs'.format(args_dict['epochs']))
    keys_with_prefix = OrderedDict()
    keys_with_prefix['shuffle'] = 'shuffle'
    # keys_with_prefix['microbp'] = 'microbp'
    keys_with_prefix['batch_size'] = 'b'
    keys_with_prefix['lr'] = 'lr'
    keys_with_prefix['w_mft'] = 'w_mft'
    keys_with_prefix['w_mfw'] = 'w_mfw'
    keys_with_prefix['w_sparse'] = 'w_s'
    keys_with_prefix['momentum'] = 'm'
    keys_with_prefix['weight_decay'] = 'wd'
    keys_with_prefix['train_on'] = 'ft'
    keys_with_prefix['opt_algo'] = 'algo'

    for key, prefix in keys_with_prefix.items():
        value = args_dict[key]
        if not is_default(key, value):
            folder_string.append('{}{}'.format(prefix, value))
    save_path = Path('_'.join(folder_string))
    return save_path


def save_path_formatter_tum(args, parser):
    def is_default(key, value):
        return value == parser.get_default(key)
    args_dict = vars(args)
    data_folder_name = str(Path(args_dict['data_dir']).normpath().name)
    #folder_string = [data_folder_name]
    folder_string = [args.seq_name]
    if not is_default('epochs', args_dict['epochs']):
        folder_string.append('{}epochs'.format(args_dict['epochs']))
    keys_with_prefix = OrderedDict()
    keys_with_prefix['shuffle'] = 'shuffle'
    # keys_with_prefix['microbp'] = 'microbp'
    keys_with_prefix['batch_size'] = 'b'
    keys_with_prefix['lr'] = 'lr'
    #keys_with_prefix['seq_name'] = 'seq'
    keys_with_prefix['w_mft'] = 'w_mft'
    keys_with_prefix['scale_loss'] = 'scaleL'
    keys_with_prefix['w_mfw'] = 'w_mfw'
    keys_with_prefix['w_sparse'] = 'w_s'
    keys_with_prefix['train_flow'] = 'trF'
    keys_with_prefix['momentum'] = 'm'
    keys_with_prefix['opt_algo'] = 'algo'

    for key, prefix in keys_with_prefix.items():
        value = args_dict[key]
        if not is_default(key, value):
            folder_string.append('{}{}'.format(prefix, value))
    save_path = Path('_'.join(folder_string))
    return save_path

def save_path_formatter_tum_direct_eval(args, parser):
    def is_default(key, value):
        return value == parser.get_default(key)
    args_dict = vars(args)
    #data_folder_name = str(Path(args_dict['data_dir']).normpath().name)
    #folder_string = [data_folder_name]
    folder_string = [args.seq_name]
    keys_with_prefix = OrderedDict()
    keys_with_prefix['algo'] = 'algo'

    for key, prefix in keys_with_prefix.items():
        value = args_dict[key]
        if not is_default(key, value):
            folder_string.append('{}{}'.format(prefix, value))
    save_path = Path('_'.join(folder_string))
    return save_path


def save_path_formatter_demo(args, parser):
    def is_default(key, value):
        return value == parser.get_default(key)
    args_dict = vars(args)
    data_folder_name = str(Path(args_dict['data_dir']).normpath().name)
    folder_string = [data_folder_name]
    keys_with_prefix = OrderedDict()
    keys_with_prefix['seq_len'] = 'seq'

    for key, prefix in keys_with_prefix.items():
        value = args_dict[key]
        if not is_default(key, value):
            folder_string.append('{}{}'.format(prefix, value))
    save_path = Path('_'.join(folder_string))
    return save_path