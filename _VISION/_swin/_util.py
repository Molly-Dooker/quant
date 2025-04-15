import traceback, ipdb
import sys
from optimum.quanto import (
    qfloat8,
    qint4,
    qint8,
)
import torch
import collections
def update_statedict(state_dict : collections.OrderedDict):
    state_dict_temp={}
    for key in state_dict.keys():
        if not key.endswith('attn.qkv.weight'): continue
        target_module= key[:-11]
        q_bias = state_dict[f'{target_module}.q_bias']
        v_bias = state_dict[f'{target_module}.v_bias']
        qkv_bias = torch.cat((q_bias, torch.zeros_like(v_bias, requires_grad=False),v_bias))
        state_dict_temp[f'{target_module}.qkv.bias']=qkv_bias
    state_dict.update(state_dict_temp)
    return state_dict


def transform(data_batch, processor):
    IMAGE = data_batch["image"]
    IMAGE = [image.convert('RGB') for image in IMAGE]
    inputs = processor(IMAGE, return_tensors="pt")
    inputs["labels"] = data_batch["label"]
    return inputs

def keyword_to_itype(k):
    return {"none": None, "int4": qint4, "int8": qint8, "float8": qfloat8}[k]

def ipdb_sys_excepthook():
    """
    When called this function will set up the system exception hook.
    This hook throws one into an ipdb breakpoint if and where a system
    exception occurs in one's run.

    E.g.
    >>> ipdb_sys_excepthook()
    """


    def info(type, value, tb):
        """
        System excepthook that includes an ipdb breakpoint.
        """
        if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        # we are in interactive mode or we don't have a tty-like
        # device, so we call the default hook
            sys.__excepthook__(type, value, tb)
        else:
            # we are NOT in interactive mode, print the exception...
            traceback.print_exception(type, value, tb)
            print
            # ...then start the debugger in post-mortem mode.
            # pdb.pm() # deprecated
            ipdb.post_mortem(tb) # more "modern"
    sys.excepthook = info