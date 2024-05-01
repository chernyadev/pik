import numpy as np
import quaternion

from pik.pik import Pik

xml_file = "universal_robots_ur5e/ur5e.xml"
attachment_site = "attachment_site"
actuator_names = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]
reference_pose = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0])

pik = Pik(
    xml_file,
    site_name=attachment_site,
    actuator_names=actuator_names,
    reference_pose=reference_pose,
)

pos = np.array([0.5, 0, 0.5])
ori = quaternion.from_euler_angles(0, np.pi, 0)
ori = quaternion.as_float_array(ori)
pik.ik(pos, ori)
