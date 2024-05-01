"""Mujoco Pseudo IK solver."""
from typing import Optional

import mujoco
import numpy as np


class Pik:
    """Mujoco Pseudo IK solver."""

    DEFAULT_KN_POS = 10.0
    DEFAULT_KN_ROT = 5.0
    DT = 0.002
    INTEGRATION_DT = 0.5
    DAMPING = 1e-2

    def __init__(
        self,
        xml_path: str,
        site_name: str,
        actuator_names: list[str],
        reference_pose: Optional[np.ndarray] = None,
        kn: Optional[np.ndarray] = None,
        ik_steps: int = 2,
    ):
        """Init."""
        self._ik_steps = ik_steps

        self._gravity_compensation: bool = True
        self._max_angvel = np.pi * 2

        self._model = mujoco.MjModel.from_xml_path(xml_path)
        self._data = mujoco.MjData(self._model)

        self._model.body_gravcomp[:] = float(self._gravity_compensation)
        self._model.opt.timestep = self.DT

        self._site_id = self._model.site(site_name).id
        self._actuator_ids = []
        self._dof_ids = []
        for name in actuator_names:
            actuator = self._model.actuator(name)
            self._actuator_ids.append(actuator.id)
            self._dof_ids.append(actuator.trnid[0])

        # Nullspace P gain.
        self._Kn = self._default_kn(len(actuator_names)) if kn is None else kn

        # Initial joint configuration
        self._q0 = (
            np.zeros_like(actuator_names) if reference_pose is None else reference_pose
        )

        # Pre-allocate numpy arrays.
        self._jac = np.zeros((6, self._model.nv))
        self._diag = self.DAMPING * np.eye(self._model.nv)
        self._eye = np.eye(self._model.nv)
        self._twist = np.zeros(6)
        self._site_quat = np.zeros(4)
        self._site_quat_conj = np.zeros(4)
        self._error_quat = np.zeros(4)

    def ik(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
        reference_pose: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Solve IK."""
        q0 = self._q0 if reference_pose is None else reference_pose
        self._set_pose(q0)

        for _ in range(self._ik_steps):
            dx = position - self._data.site(self._site_id).xpos
            self._twist[:3] = dx / self.INTEGRATION_DT
            mujoco.mju_mat2Quat(self._site_quat, self._data.site(self._site_id).xmat)
            mujoco.mju_negQuat(self._site_quat_conj, self._site_quat)
            mujoco.mju_mulQuat(self._error_quat, orientation, self._site_quat_conj)
            mujoco.mju_quat2Vel(self._twist[3:], self._error_quat, 1.0)
            self._twist[3:] *= self.INTEGRATION_DT

            mujoco.mj_jacSite(
                self._model, self._data, self._jac[:3], self._jac[3:], self._site_id
            )
            dq = np.linalg.solve(
                self._jac.T @ self._jac + self._diag, self._jac.T @ self._twist
            )
            dq += (self._eye - np.linalg.pinv(self._jac) @ self._jac) @ (
                self._Kn * (q0 - self._data.qpos[self._dof_ids])
            )

            dq_abs_max = np.abs(dq).max()
            if dq_abs_max > self._max_angvel:
                dq *= self._max_angvel / dq_abs_max

            q = self._data.qpos.copy()
            mujoco.mj_integratePos(self._model, q, dq, self.INTEGRATION_DT)
            np.clip(q, *self._model.jnt_range.T, out=q)

            self._data.ctrl[self._actuator_ids] = q[self._dof_ids]
            mujoco.mj_step(self._model, self._data)
        return q[self._dof_ids].copy()

    def fk(self, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Solve Forward Kinematics."""
        self._set_pose(pose)
        position = self._data.site(self._site_id).xpos
        orientation = np.zeros(4)
        mujoco.mju_mat2Quat(orientation, self._data.site(self._site_id).xmat)
        return position, orientation

    def _default_kn(self, joints_count: int):
        kn = np.repeat(self.DEFAULT_KN_POS, joints_count)
        if joints_count >= 6:
            kn[-3:] = self.DEFAULT_KN_ROT
        return kn

    def _set_pose(self, pose: np.ndarray):
        self._data.ctrl[self._actuator_ids] = pose
        self._data.qpos[self._dof_ids] = pose
        self._data.qvel[self._dof_ids] *= 0
        self._data.qacc[self._dof_ids] *= 0
        mujoco.mj_step(self._model, self._data)
