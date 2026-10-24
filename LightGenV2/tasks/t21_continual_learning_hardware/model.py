"""Native 8-um deployment candidate; original T16 graph stays untouched."""
from LightGenV2.tasks.t20_reconfigurable_rgb_sar.hardware_aware import NativeOEOOptics


def build_model(architecture):
    # 4*230 + 3*32 = 1016 native pixels. One optical expert array,
    # one global phase, plus an independent router for MoE only.
    return NativeOEOOptics(architecture,expert_grid=4,expert_size=230,gap=32,
                           border=20,routing_temperature=1.25,readout='linear')
