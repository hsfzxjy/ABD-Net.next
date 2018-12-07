#!/usr/bin/env python3

import json

arg_list = [
    {
        "epoch": 40,
        "env": {},
        "criterion": "xent",
        "fixbase": 0,
        "fix_custom_loss": False,
        "switch_loss": 0,
        "arch": 'densenet121',
        "log_dir": "multi_test_log/densenet121_40_0_xent"
    },
    {
        "epoch": 40,
        "env": {},
        "criterion": "xent",
        "fixbase": 10,
        "fix_custom_loss": False,
        "switch_loss": 0,
        "arch": 'densenet121_fc512',
        "log_dir": "multi_test_log/densenet121_fc512_40_10_xent"
    },

    {
        "epoch": 40,
        "env": {},
        "criterion": "xent",
        "fixbase": 10,
        "fix_custom_loss": False,
        "switch_loss": 0,
        "arch": 'densenet121',
        "log_dir": "multi_test_log/densenet121_40_10_xent"
    },
    {
        "epoch": 40,
        "env": {},
        "criterion": "xent",
        "fixbase": 0,
        "fix_custom_loss": False,
        "switch_loss": 0,
        "arch": 'densenet121_fc512',
        "log_dir": "multi_test_log/densenet121_fc512_40_0_xent"
    },

]

for beta in [f'{base}e-{exp}' for exp in [6, 7, 8, 9] for base in [1, 5, 7]]:
    pass
    # constrain weights
    # arg_list.extend([
    #     {
    #         "epoch": 40,
    #         "env": {"beta": beta, "constraint_weights": "1"},
    #         "criterion": "lowrank",
    #         "fixbase": 0,
    #         "fix_custom_loss": False,
    #         "switch_loss": 15,
    #         "arch": 'densenet121',
    #         "log_dir": f"multi_test_log/densenet121_40_0_lowrank_beta_{beta}_sl15_cw"
    #     },
    #     {
    #         "epoch": 40,
    #         "env": {"beta": beta, "constraint_weights": "1"},
    #         "criterion": "lowrank",
    #         "fixbase": 10,
    #         "fix_custom_loss": True,
    #         "switch_loss": 15,
    #         "arch": 'densenet121_fc512',
    #         "log_dir": f"multi_test_log/densenet121_fc512_40_10_lowrank_beta_{beta}_sl15_fcl_cw"
    #     },
    # ])

    # arg_list.extend([
    #     {
    #         "epoch": 40,
    #         "env": {"beta": beta},
    #         "criterion": "lowrank",
    #         "fixbase": 0,
    #         "fix_custom_loss": False,
    #         "switch_loss": 15,
    #         "arch": 'densenet121',
    #         "log_dir": f"multi_test_log/densenet121_40_0_lowrank_beta_{beta}_sl15"
    #     },
    #     {
    #         "epoch": 40,
    #         "env": {"beta": beta},
    #         "criterion": "lowrank",
    #         "fixbase": 10,
    #         "fix_custom_loss": True,
    #         "switch_loss": 15,
    #         "arch": 'densenet121_fc512',
    #         "log_dir": f"multi_test_log/densenet121_fc512_40_10_lowrank_beta_{beta}_sl15_fcl"
    #     },
    # ])

    # Singular
    # constrain weights
    # arg_list.extend([
    #     {
    #         "epoch": 40,
    #         "env": {"beta": beta, "constraint_weights": "1"},
    #         "criterion": "singular",
    #         "fixbase": 0,
    #         "fix_custom_loss": False,
    #         "switch_loss": 15,
    #         "arch": 'densenet121',
    #         "log_dir": f"multi_test_log/densenet121_40_0_singular_beta_{beta}_sl15_cw"
    #     },
    #     {
    #         "epoch": 40,
    #         "env": {"beta": beta, "constraint_weights": "1"},
    #         "criterion": "singular",
    #         "fixbase": 10,
    #         "fix_custom_loss": True,
    #         "switch_loss": 15,
    #         "arch": 'densenet121_fc512',
    #         "log_dir": f"multi_test_log/densenet121_fc512_40_10_singular_beta_{beta}_sl15_fcl_cw"
    #     },
    # ])

    # arg_list.extend([
    #     {
    #         "epoch": 40,
    #         "env": {"beta": beta},
    #         "criterion": "singular",
    #         "fixbase": 0,
    #         "fix_custom_loss": False,
    #         "switch_loss": 15,
    #         "arch": 'densenet121',
    #         "log_dir": f"multi_test_log/densenet121_40_0_singular_beta_{beta}_sl15"
    #     },
    #     {
    #         "epoch": 40,
    #         "env": {"beta": beta},
    #         "criterion": "singular",
    #         "fixbase": 10,
    #         "fix_custom_loss": True,
    #         "switch_loss": 15,
    #         "arch": 'densenet121_fc512',
    #         "log_dir": f"multi_test_log/densenet121_fc512_40_10_singular_beta_{beta}_sl15_fcl"
    #     },
    # ])

print(json.dumps(arg_list, indent=2))
