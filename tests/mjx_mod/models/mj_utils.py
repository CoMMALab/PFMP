import mujoco
import os
from typing import Tuple


def load_mj(name: str) -> Tuple[mujoco.MjModel, mujoco.MjData]:
    cwd = os.getcwd()
    os.chdir("tests/mjx_mod/assets")
    with open(f"{name}.xml", 'r') as f:
        xml = f.read()
    
    mj_model = mujoco.MjModel.from_xml_string(xml)
    mj_data = mujoco.MjData(mj_model)

    key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
    os.chdir(cwd)
    return (mj_model, mj_data)
