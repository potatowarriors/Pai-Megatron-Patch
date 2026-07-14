"""DiLoCo entry point: installs the outer-loop patch, then runs pretrain_alpha.

Launch each node as an independent single-node torchrun (no WORLD_SIZE=2);
cross-node sync is handled entirely by diloco_patch via private Gloo pairs.
See diloco_patch.py for the env contract and launch_diloco.sh for usage.
"""
import os
import runpy
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import diloco_patch  # noqa: E402

diloco_patch.install()
runpy.run_path(os.path.join(_HERE, "pretrain_alpha.py"), run_name="__main__")
