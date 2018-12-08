#!/usr/bin/env python3

import json
arg_list = []

for beta in [f'{base}e-{exp}' for exp in [5, 6] for base in [1]]:
    for reg in ['svmo', 'svdo', 'so']:
        for fixbase in [0, 10]:
            arg_list.extend([
                {
                    "epoch": 40,
                    "env": {"beta": beta},
                    "criterion": "xent",
                    "regularizer": reg,
                    "fixbase": fixbase,
                    "fix_custom_loss": False,
                    "switch_loss": 15,
                    "arch": 'densenet121_fc512',
                    "log_dir": f"multi_test_log/densenet121_fc512_40_{fixbase}_xent_reg_{reg}_beta_{beta}_sl15"
                },

            ])

print(json.dumps(arg_list, indent=2))
