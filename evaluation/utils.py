import random
import numpy as np
import yaml
from argparse import Action
import os
from datetime import datetime


def setup_seeds(seed=42):
    import torch
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def get_config(config_file, config_name):
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)
        config = config.get(config_name, None)
        return config

def get_result_dir(model_name, dataset_name, time=None, output_root="./outputs"):
    dir_list = os.listdir(os.path.join(output_root, model_name))
    dir_list = [dir_name for dir_name in dir_list if dir_name[:-16] == dataset_name]
    closest_dir = None
    if time:
        given_time = datetime.strptime(time, "%Y%m%d-%H%M%S")
        for dir_name in dir_list:
            try:
                dir_time = dir_name.split('_')[-1]
                dir_time = datetime.strptime(dir_time, "%Y%m%d-%H%M%S")
            except ValueError:
                continue
            if dir_time > given_time:
                if closest_dir is None or dir_time < closest_time:
                    closest_dir = dir_name
                    closest_time = dir_time
    if closest_dir is None or time is None:
        closest_dir = max(dir_list, key=lambda f: datetime.strptime(f.split('_')[-1], "%Y%m%d-%H%M%S"))
    closest_dir = os.path.join(output_root, model_name, closest_dir)
    return closest_dir, closest_dir[-15:]





class DictAction(Action):
    """
    argparse action to split an argument into KEY=VALUE form
    on the first = and append to a dictionary. List options can
    be passed as comma separated values, i.e 'KEY=V1,V2,V3', or with explicit
    brackets, i.e. 'KEY=[V1,V2,V3]'. It also support nested brackets to build
    list/tuple values. e.g. 'KEY=[(V1,V2),(V3,V4)]'
    """

    @staticmethod
    def _parse_int_float_bool(val):
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        if val.lower() in ['true', 'false']:
            return True if val.lower() == 'true' else False
        if val == 'None':
            return None
        return val

    @staticmethod
    def _parse_iterable(val):
        """Parse iterable values in the string.

        All elements inside '()' or '[]' are treated as iterable values.

        Args:
            val (str): Value string.

        Returns:
            list | tuple: The expanded list or tuple from the string.

        Examples:
            >>> DictAction._parse_iterable('1,2,3')
            [1, 2, 3]
            >>> DictAction._parse_iterable('[a, b, c]')
            ['a', 'b', 'c']
            >>> DictAction._parse_iterable('[(1, 2, 3), [a, b], c]')
            [(1, 2, 3), ['a', 'b'], 'c']
        """

        def find_next_comma(string):
            """Find the position of next comma in the string.

            If no ',' is found in the string, return the string length. All
            chars inside '()' and '[]' are treated as one element and thus ','
            inside these brackets are ignored.
            """
            assert (string.count('(') == string.count(')')) and (
                    string.count('[') == string.count(']')), \
                f'Imbalanced brackets exist in {string}'
            end = len(string)
            for idx, char in enumerate(string):
                pre = string[:idx]
                # The string before this ',' is balanced
                if ((char == ',') and (pre.count('(') == pre.count(')'))
                        and (pre.count('[') == pre.count(']'))):
                    end = idx
                    break
            return end

        # Strip ' and " characters and replace whitespace.
        # val = val.strip('\'\"').replace(' ', '')
        # val = val.strip('\'\"')#.replace(' ', '')
        is_tuple = False
        if val.startswith('(') and val.endswith(')'):
            is_tuple = True
            val = val[1:-1]
        elif val.startswith('[') and val.endswith(']'):
            val = val[1:-1]
        elif ',' not in val:
            # val is a single value
            return DictAction._parse_int_float_bool(val)

        values = []
        while len(val) > 0:
            comma_idx = find_next_comma(val)
            element = DictAction._parse_iterable(val[:comma_idx])
            values.append(element)
            val = val[comma_idx + 1:]
        if is_tuple:
            values = tuple(values)
        return values

    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        print(values)
        for kv in values:
            key, val = kv.split('=', maxsplit=1)
            options[key] = self._parse_iterable(val)
        setattr(namespace, self.dest, options)


def merge_from_dict(cfg_dict, options, allow_list_keys=True):
    """Merge list into cfg_dict.

    Merge the dict parsed by MultipleKVAction into this cfg.

    Examples:
        >>> options = {'model.backbone.depth': 50,
        ...            'model.backbone.with_cp':True}
        >>> cfg = Config(dict(model=dict(backbone=dict(type='ResNet'))))
        >>> cfg.merge_from_dict(options)
        >>> cfg_dict = super(Config, self).__getattribute__('_cfg_dict')
        >>> assert cfg_dict == dict(
        ...     model=dict(backbone=dict(depth=50, with_cp=True)))

        >>> # Merge list element
        >>> cfg = Config(dict(pipeline=[
        ...     dict(type='LoadImage'), dict(type='LoadAnnotations')]))
        >>> options = dict(pipeline={'0': dict(type='SelfLoadImage')})
        >>> cfg.merge_from_dict(options, allow_list_keys=True)
        >>> cfg_dict = super(Config, self).__getattribute__('_cfg_dict')
        >>> assert cfg_dict == dict(pipeline=[
        ...     dict(type='SelfLoadImage'), dict(type='LoadAnnotations')])

    Args:
        options (dict): dict of configs to merge from.
        allow_list_keys (bool): If True, int string keys (e.g. '0', '1')
            are allowed in ``options`` and will replace the element of the
            corresponding index in the config if the config is a list.
            Default: True.
    """
    option_cfg_dict = {}
    for full_key, v in options.items():
        d = option_cfg_dict
        key_list = full_key.split('.')
        for subkey in key_list[:-1]:
            d.setdefault(subkey, {})
            d = d[subkey]
        subkey = key_list[-1]
        d[subkey] = v
    cfg_dict = merge_a_into_b(option_cfg_dict, cfg_dict, allow_list_keys=allow_list_keys)
    return cfg_dict

def merge_a_into_b(a, b, allow_list_keys=False):
    b = b.copy()
    for k, v in a.items():
        if allow_list_keys and k.isdigit() and isinstance(b, list):
            k = int(k)
            if len(b) <= k:
                raise KeyError(f'Index {k} exceeds the length of list {b}')
            b[k] = merge_a_into_b(v, b[k], allow_list_keys)
        elif isinstance(v, dict):
            if k in b and not v.pop('_delete_', False):
                allowed_types = (dict, list) if allow_list_keys else dict
                if not isinstance(b[k], allowed_types):
                    raise TypeError(
                        f'{k}={v} in child config cannot inherit from '
                        f'base because {k} is a dict in the child config '
                        f'but is of type {type(b[k])} in base config. '
                        f'You may set `{DELETE_KEY}=True` to ignore the '
                        f'base config.')
                b[k] = merge_a_into_b(v, b[k], allow_list_keys)
            else:
                b[k] = v
        else:
            b[k] = v
    return b