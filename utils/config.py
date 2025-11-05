import os
from typing import Any, Optional
import yaml
import json
import copy
import argparse

import logging
logger = logging.getLogger(__name__)

class Config(object):
    """
    Global config object. 
    It automatically loads from a hierarchy of config files and turns the keys to the 
    class attributes. 
    """
    def __init__(self, load=True, cfg_dict=None, path: Optional[str]="configs/base.yaml"):
        """
        Args: 
            load (bool): whether or not yaml is needed to be loaded.
            cfg_dict (dict): dictionary of configs to be updated into the attributes
        """
        if load:
            if path is None:
                args = self._parse_args()
                self.cfg_file = args.cfg_file
            else:
                self.cfg_file = path

            print("Loading config from {}.".format(self.cfg_file))
            self.need_initialization = True
            cfg_base = self._initialize_cfg()
            cfg_dict = self._load_yaml(self.cfg_file)
            cfg_dict = self._merge_cfg_from_base(cfg_base, cfg_dict)
            self.cfg_dict = cfg_dict

        self.update_dict(cfg_dict)

    def _parse_args(self):
        """
        Wrapper for argument parser. 
        """
        parser = argparse.ArgumentParser(
            description="Argparser for configuring [code base name to think of] codebase"
        )
        parser.add_argument(
            "--cfg",
            dest="cfg_file",
            help="Path to the configuration file",
            default="configs/base.yaml"
        )
        return parser.parse_args()

    def _path_join(self, path_list):
        """
        Join a list of paths.
        Args:
            path_list (list): list of paths.
        """
        path = ""
        for p in path_list:
            path+= p + '/'
        return path[:-1]

    def _initialize_cfg(self):
        """
        When loading config for the first time, base config is required to be read.
        """
        cfg = None
        base_path = './configs/base.yaml'
        if self.need_initialization:
            self.need_initialization = False
            if os.path.exists(base_path):

                with open(base_path, 'r') as f:
                    cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)
            else:
                logging.warning("Base config file not found at {}. Skipping base config loading.".format(base_path))
        return cfg
    
    def _load_yaml(self, cfg_file, file_name=""):
        """
        Load the specified yaml file.
        Args:
            args: parsed args by `self._parse_args`.
            file_name (str): the file name to be read from if specified.
        """
        assert cfg_file is not None
        if not file_name == "": # reading from base file
            with open(file_name, 'r') as f:
                cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)
        else: # reading from top file
            with open(cfg_file, 'r') as f:
                cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)
                file_name = cfg_file

        if "_BASE_RUN" not in cfg.keys() and "_BASE_MODEL" not in cfg.keys() and "_BASE" not in cfg.keys():
            # return cfg if the base file is being accessed
            return cfg

        if "_BASE" in cfg.keys():
            # load the base file of the current config file
            if cfg["_BASE"][1] == '.':
                prev_count = cfg["_BASE"].count('..')
                cfg_base_file = self._path_join(file_name.split('/')[:(-1-cfg["_BASE"].count('..'))] + cfg["_BASE"].split('/')[prev_count:])
            else:
                cfg_base_file = cfg["_BASE"].replace(
                    "./", 
                    cfg_file.replace(cfg_file.split('/')[-1], "")
                )
            cfg_base = self._load_yaml(cfg_file, cfg_base_file)
            cfg = self._merge_cfg_from_base(cfg_base, cfg)
        else:
            # load the base run and the base model file of the current config file
            if "_BASE_RUN" in cfg.keys():
                if cfg["_BASE_RUN"][1] == '.':
                    prev_count = cfg["_BASE_RUN"].count('..')
                    cfg_base_file = self._path_join(file_name.split('/')[:(-1-prev_count)] + cfg["_BASE_RUN"].split('/')[prev_count:])
                else:
                    cfg_base_file = cfg["_BASE_RUN"].replace(
                        "./", 
                        cfg_file.replace(cfg_file.split('/')[-1], "")
                    )
                cfg_base = self._load_yaml(cfg_file, cfg_base_file)
                cfg = self._merge_cfg_from_base(cfg_base, cfg, preserve_base=True)
            if "_BASE_MODEL" in cfg.keys():
                if cfg["_BASE_MODEL"][1] == '.':
                    prev_count = cfg["_BASE_MODEL"].count('..')
                    cfg_base_file = self._path_join(file_name.split('/')[:(-1-cfg["_BASE_MODEL"].count('..'))] + cfg["_BASE_MODEL"].split('/')[prev_count:])
                else:
                    cfg_base_file = cfg["_BASE_MODEL"].replace(
                        "./", 
                        cfg_file.replace(cfg_file.split('/')[-1], "")
                    )
                cfg_base = self._load_yaml(cfg_file, cfg_base_file)
                cfg = self._merge_cfg_from_base(cfg_base, cfg)
        cfg = self._merge_cfg_from_command(cfg_file, cfg)
        return cfg
    
    def _merge_cfg_from_base(self, cfg_base, cfg_new, preserve_base=False):
        """
        Replace the attributes in the base config by the values in the coming config, 
        unless preserve base is set to True.
        Args:
            cfg_base (dict): the base config.
            cfg_new (dict): the coming config to be merged with the base config.
            preserve_base (bool): if true, the keys and the values in the cfg_new will 
                not replace the keys and the values in the cfg_base, if they exist in 
                cfg_base. When the keys and the values are not present in the cfg_base,
                then they are filled into the cfg_base.
        """
        if cfg_base is None:
            return cfg_new
        for k,v in cfg_new.items():
            if k in cfg_base.keys():
                if isinstance(v, dict):
                    self._merge_cfg_from_base(cfg_base[k], v)
                else:
                    cfg_base[k] = v
            else:
                if "BASE" not in k or preserve_base:
                    cfg_base[k] = v
        return cfg_base

    def _merge_cfg_from_command(self, args, cfg):
        """
        Merge cfg from command. Currently only support depth of four. 
        E.g. VIDEO.BACKBONE.BRANCH.XXXX. is an attribute with depth of four.
        Args:
            args : the command in which the overriding attributes are set.
            cfg (dict): the loaded cfg from files.
        """
        assert len(args.opts) % 2 == 0, 'Override list {} has odd length: {}.'.format(
            args.opts, len(args.opts)
        )
        keys = args.opts[0::2]
        vals = args.opts[1::2]

        # maximum supported depth 3
        for idx, key in enumerate(keys):
            key_split = key.split('.')
            assert len(key_split) <= 4, 'Key depth error. \nMaximum depth: 3\n Get depth: {}'.format(
                len(key_split)
            )
            assert key_split[0] in cfg.keys(), 'Non-existant key: {}.'.format(
                key_split[0]
            )
            if len(key_split) == 2:
                assert key_split[1] in cfg[key_split[0]].keys(), 'Non-existant key: {}.'.format(
                    key
                )
            elif len(key_split) == 3:
                assert key_split[1] in cfg[key_split[0]].keys(), 'Non-existant key: {}.'.format(
                    key
                )
                assert key_split[2] in cfg[key_split[0]][key_split[1]].keys(), 'Non-existant key: {}.'.format(
                    key
                )
            elif len(key_split) == 4:
                assert key_split[1] in cfg[key_split[0]].keys(), 'Non-existant key: {}.'.format(
                    key
                )
                assert key_split[2] in cfg[key_split[0]][key_split[1]].keys(), 'Non-existant key: {}.'.format(
                    key
                )
                assert key_split[3] in cfg[key_split[0]][key_split[1]][key_split[2]].keys(), 'Non-existant key: {}.'.format(
                    key
                )


            if len(key_split) == 1:
                cfg[key_split[0]] = vals[idx]
            elif len(key_split) == 2:
                cfg[key_split[0]][key_split[1]] = vals[idx]
            elif len(key_split) == 3:
                cfg[key_split[0]][key_split[1]][key_split[2]] = vals[idx]
            elif len(key_split) == 4:
                cfg[key_split[0]][key_split[1]][key_split[2]][key_split[3]] = vals[idx]
            
        return cfg
    
    def update_dict(self, cfg_dict):
        """
        Set the dict to be attributes of the config recurrently.
        Args:
            cfg_dict (dict): the dictionary to be set as the attribute of the current 
                config class.
        """
        def recur(key, elem):
            if type(elem) is dict:
                return key, Config(load=False, cfg_dict=elem)
            else:
                if type(elem) is str and elem[1:3]=="e-":
                    elem = float(elem)
                return key, elem
        
        dic = dict(recur(k, v) for k, v in cfg_dict.items())
        self.__dict__.update(dic)
    
    def get_args(self):
        """
        Returns the read arguments.
        """
        return self.cfg_file
    
    def __getattr__(self, name) -> Any:
        """
        Return None for any missing attribute instead of raising AttributeError.
        This makes accessing cfg.NON_EXISTANT_KEY return None.
        """
        return None
    
    def __repr__(self):
        return "{}\n".format(self.dump())
            
    def dump(self):
        return json.dumps(self.cfg_dict, indent=2)

    def deep_copy(self):
        return copy.deepcopy(self)
    
if __name__ == '__main__':
    # debug
    logging.basicConfig(level=logging.INFO)
    cfg = Config(load=True)
    logging.info(cfg.dump())
    logging.info(cfg.TOKENIZER.MIN_LEN)
