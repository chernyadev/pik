import numpy as np

from pik.pik import Pik

xml_file = "franka_emika_panda/panda_nohand.xml"
attachment_site = "attachment_site"
actuator_names = [
    "actuator1",
    "actuator2",
    "actuator3",
    "actuator4",
    "actuator5",
    "actuator6",
    "actuator7",
]
reference_pose = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])

pik = Pik(
    xml_file,
    site_name=attachment_site,
    actuator_names=actuator_names,
    reference_pose=reference_pose,
)
pik.solve(np.array([0.5, 0, 0.5]), np.array([0, 1, 0, 0]))
