import ttnn
from bos_metal import config, op
from bos_metal.operations.conv.conv2d_config import Conv2dConfig



bevformer_config_v55_4x5 = {
    "img_backbone": {
        "conv1": {
              "conv" : {"config" : {
                    "act_block_h_override" : 32*1,
                    "input_channels_alignment" : 8*2,
                }
            }
        },
        "layer1": {
            "0": { # 0
                "conv1": {
                        "conv" : {"config" : {
                        "act_block_h_override" : 32*1,
                        }
                    }
                },
                "conv2": {
                        "conv" : {"config" : {
                        "act_block_h_override" : 32*3,
                        }
                    }
                },
                "conv3": {
                        "conv" : {"config" : {
                        "act_block_h_override" : 32*7,
                        }
                    }
                },
                "downsample": {
                    "0" : {
                            "conv" : {"config" : {
                            "act_block_h_override" : 32*7,
                            }
                        }
                    }
                }
            },
             "1": { # 1
                "conv1": {
                        "conv" : {"config" : {
                        "act_block_h_override" : 32*7,
                        }
                    }
                },
                "conv2": {
                        "conv" : {"config" : {
                        "act_block_h_override" : 32*3,
                        }
                    }
                },
                "conv3": {
                        "conv" : {"config" : {
                        "act_block_h_override" : 32*7,
                        }
                    }
                }
            },
            "2": { # 2
                "conv1": {
                        "conv" : {"config" : {
                        "act_block_h_override" : 32*2,
                        }
                    }
                },
                "conv2": {
                        "conv" : {"config" : {
                        "act_block_h_override" : 32*3,
                        }
                    }
                },
                "conv3": {
                        "conv" : {"config" : {
                        "act_block_h_override" : 32*7,
                        }
                    }
                },
            }
        },
        "layer2": {
            "0": { # 3
                "conv1": {
                        "conv" : {"config" : {
                        "act_block_h_override" : 32*7,
                        }
                    }
                },
                "conv2": {
                        "conv" : {"config" : {
                        "act_block_h_override" : 32*1,
                        }
                    }
                },
                "conv3": {
                        "conv" : {"config" : {
                        "act_block_h_override" : 32*7,
                        }
                    }
                },
                "downsample": {
                    "0" : {
                            "conv" : {"config" : {
                            "act_block_h_override" : 32*1
                            }
                        }
                    }
                }
            },
             "1": { # 4
                "conv1": {
                        "conv" : {"config" : {
                        "act_block_h_override" : 32*7,
                        }
                    }
                },
                "conv2": {
                        "conv" : {"config" : {
                        "act_block_h_override" : 32*1,
                        }
                    }
                },
                "conv3": {
                        "conv" : {"config" : {
                        "act_block_h_override" : 32*7,
                        }
                    }
                },
            },
           "2": { # 5
                "conv1": {
                        "conv" : {"config" : {
                        "act_block_h_override" : 32*7,
                        }
                    }
                },
                "conv2": {
                        "conv" : {"config" : {
                        "act_block_h_override" : 32*1,
                        }
                    }
                },
                "conv3": {
                        "conv" : {"config" : {
                        "act_block_h_override" : 32*7,
                        }
                    }
                }
            },
            "3": { # 6
                "conv1": {
                        "conv" : {"config" : {
                        "act_block_h_override" : 32*7,
                        }
                    }
                },
                "conv2": {
                        "conv" : {"config" : {
                        "act_block_h_override" : 32*1,
                        }
                    }
                },
                "conv3": {
                        "conv" : {"config" : {
                        "act_block_h_override" : 32*7,
                        }
                    }
                }
            },
        },
        "layer3": {
            "0": { #7 
                 "conv1": {
                        "conv" : {"config" : {
                        "act_block_h_override" : 32*7,
                        }
                    }
                },
                "conv2": {
                        "conv" : {"config" : {
                        "act_block_h_override" : 32*1, #5  #3 
                        }
                    }
                },
                "conv3": {
                        "conv" : {"config" : {
                        "act_block_h_override" : 32*2,
                        }
                    }
                },
                "downsample": {
                    "0" : {
                            "conv" : {"config" : {
                            "act_block_h_override" : 32*1,
                            }
                        }
                    }
                }
            },
            "1": { #8
                 "conv1": {
                        "conv" : {"config" : {
                        "act_block_h_override" : 32*4,
                        }
                    }
                },
                "conv2": {
                        "conv" : {"config" : {
                        "act_block_h_override" : 32*1,
                        }
                    }
                },
                "conv3": {
                        "conv" : {"config" : {
                        "act_block_h_override" : 32*4,
                        }
                    }
                }
            },
            "2": { #9
                 "conv1": {
                        "conv" : {"config" : {
                        "act_block_h_override" : 32*4,
                        }
                    }
                },
                "conv2": {
                        "conv" : {"config" : {
                        "act_block_h_override" : 32*1,
                        }
                    }
                },
                "conv3": {
                        "conv" : {"config" : {
                        "act_block_h_override" : 32*4,
                        }
                    }
                }
            },
            "3": { #10
                 "conv1": {
                        "conv" : {"config" : {
                        "act_block_h_override" : 32*4,
                        }
                    }
                },
                "conv2": {
                        "conv" : {"config" : {
                        "act_block_h_override" : 32*1,
                        }
                    }
                },
                "conv3": {
                        "conv" : {"config" : {
                        "act_block_h_override" : 32*4,
                        }
                    }
                }
            },
            "4": { #11
                 "conv1": {
                        "conv" : {"config" : {
                        "act_block_h_override" : 32*4,
                        }
                    }
                },
                "conv2": {
                        "conv" : {"config" : {
                        "act_block_h_override" : 32*1,
                        }
                    }
                },
                "conv3": {
                        "conv" : {"config" : {
                        "act_block_h_override" : 32*4,
                        }
                    }
                }
            },
            "5": { #12
                 "conv1": {
                        "conv" : {"config" : {
                        "act_block_h_override" : 32*4,
                        }
                    }
                },
                "conv2": {
                        "conv" : {"config" : {
                        "act_block_h_override" : 32*1,
                        }
                    }
                },
                "conv3": {
                        "conv" : {"config" : {
                        "act_block_h_override" : 32*4,
                        }
                    }
                }
            }
        },
        "layer4": {
            "0": { # 13
                "conv1": {
                        "conv" : {"config" : {
                        "act_block_h_override" : 32*1,
                        # "dtype" : ttnn.bfloat16
                        }
                    }
                },
                "conv2": {
                        "conv" : {"config" : {
                        "act_block_h_override" : 32*1,
                        # "dtype" : ttnn.bfloat16
                        }
                    }
                },
                "conv3": {
                        "conv" : {"config" : {
                        "act_block_h_override" : 32*1,
                        # "dtype" : ttnn.bfloat16
                        }
                    }
                },
                "downsample": {
                    "0" : {
                            "conv" : {"config" : {
                            "act_block_h_override" : 32*1,
                            # "dtype" : ttnn.bfloat16
                            }
                        }
                    }
                }
            },
             "1": { # 14
                "conv1": {
                        "conv" : {"config" : {
                        "act_block_h_override" : 32*2,
                        }
                    }
                },
                "conv2": {
                        "conv" : {"config" : {
                        "act_block_h_override" : 32*1,
                        }
                    }
                },
                "conv3": {
                        "conv" : {"config" : {
                        "act_block_h_override" : 32*1,
                        }
                    }
                }
            },
            "2": { # 15
                "conv1": {
                        "conv" : {"config" : {
                        "act_block_h_override" : 32*2,
                        }
                    }
                },
                "conv2": {
                        "conv" : {"config" : {
                        "act_block_h_override" : 32*1,
                        }
                    }
                },
                "conv3": {
                        "conv" : {"config" : {
                        "act_block_h_override" : 32*1,
                        }
                    }
                },
            }
        },
    },

    "img_neck" :{
        "lateral_convs": {
            "0":{
                "conv": {
                    "config" : {
                        "act_block_h_override" : 32*2,
                    }
                }
            }
        },
        "fpn_convs":{
            "0":{
                "conv": {
                     "config" : {
                        "act_block_h_override" : 32*1,
                    }
                }
            }
        }
    },
    "pts_bbox_head": {
        "transformer" :{
            "encoder": {
                "layers" :{
                    "0": {
                        "norms": {
                            # "init_config": op.InitConfig(
                            #                     input_config=op.InputConfig(
                            #                     sharded_to_interleaved=True,
                            #                     layout=ttnn.ROW_MAJOR_LAYOUT,
                            #                     memory_config=ttnn.L1_MEMORY_CONFIG
                            #                     )
                            #                 ),
                            # "config": sharded_conv2d_config(x=3, y=4, shard='height')
                            },
        
                    },
                },
                    # "1": {
                    #     "norm": {
                           
                    #     }
                    # },
            }
        },
    }

        # {
        #         #         "init_config": op.InitConfig(
        #         #                             output_config=op.OutputConfig(
        #         #                             reallocate=True,  
        #         #                             layout=ttnn.ROW_MAJOR_LAYOUT,
        #         #                             memory_config=ttnn.DRAM_MEMORY_CONFIG
        #         #                             )
        #         #                         ),
        #         #         "config": sharded_conv2d_config(x=3, y=4, shard='height')
        #         #         },
        # }
    # }
}
