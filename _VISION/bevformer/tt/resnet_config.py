import ttnn
from bos_metal import config, op

def sharded_conv2d_config(x=3, y=3, shard='height'):
    map_shard = {
        "height": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        "width": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        "block": ttnn.TensorMemoryLayout.BLOCK_SHARDED
    }
    return op.Conv2dConfig(
        core_grid=ttnn.CoreRangeSet({
            ttnn.CoreRange((0, 0), (x, y))
        }),
        shard_layout=map_shard[shard]
    )


resnet50_config_v55_4x5 = {
    "layer1": {
        "0": {
            "conv1": 
            {
                "init_config": op.InitConfig(
                    input_config=op.InputConfig(
                        sharded_to_interleaved=True,    
                    ),
                )
            },
            "downsample": {
                "0": {
                        "init_config": op.InitConfig(
                        input_config=op.InputConfig(
                            sharded_to_interleaved=True,    
                        ),
                    )
                }
            }
        },
        "1": {
            "conv3": {
                "config": sharded_conv2d_config(x=3, y=3, shard='height')
            },
        },
        "2": {
            "conv3": {
                "config": sharded_conv2d_config(x=3, y=3, shard='height')
            },
        },

    },
    "layer2": {
        "0": {
            "conv3": {
                "config": sharded_conv2d_config(x=3, y=3, shard='height')
            },
            "downsample": {
                "0": {
                    "config": sharded_conv2d_config(x=3, y=3, shard='height')
                }
            }
        },
        "1": {
            "conv3": {
                "config": sharded_conv2d_config(x=3, y=3, shard='height')
            },
        },
        "2": {
            "conv3": {
                "config": sharded_conv2d_config(x=3, y=3, shard='height')
            },
        },
        "3": {
            "conv3": {
                "config": sharded_conv2d_config(x=3, y=3, shard='height')
            },
        }
    },
    "layer3": {
        "0": {
            "conv3": {
                "config": sharded_conv2d_config(x=3, y=3, shard='height')
            },
            "downsample": {
                "0": {
                    "config": sharded_conv2d_config(x=3, y=3, shard='height')
                }
            },
            "add" : {
                "init_config": op.InitConfig(
                    input_config=[
                        op.InputConfig(memory_config=ttnn.DRAM_MEMORY_CONFIG), 
                        op.InputConfig(memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    ],
                ),
            }
        },
        "1": {
            "conv3": {
                "config": sharded_conv2d_config(x=3, y=3, shard='height')
            },
        },
        "2": {
            "conv3": {
                "config": sharded_conv2d_config(x=3, y=3, shard='height')
            },
        },
        "3": {
            "conv3": {
                "config": sharded_conv2d_config(x=3, y=3, shard='height')
            },
        },
        "4": {
            "conv3": {
                "config": sharded_conv2d_config(x=3, y=3, shard='height')
            },
        },
        "5": {
            "conv3": {
                "config": sharded_conv2d_config(x=3, y=3, shard='height')
            },
        }
    },
    "layer4": {
        "0": {
            "conv3": {
                "config": sharded_conv2d_config(x=3, y=3, shard='block')
            },
            "downsample": {
                "0": {
                    "config": sharded_conv2d_config(x=3, y=3, shard='block')
                }
            }
        },
        "1": {
            "conv3": {
                "config": sharded_conv2d_config(x=3, y=3, shard='block')
            },
        },
        "2": {
            "conv3": {
                "config": sharded_conv2d_config(x=3, y=3, shard='block')
            },
        },
    }
}
 
