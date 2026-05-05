"""DOF freezing task implementation."""

from __future__ import annotations

import mujoco
import numpy as np

from ..configuration import Configuration
from ..exceptions import TaskDefinitionError
from .task import Task


class DofFreezingTask(Task):
    """Freeze specific degrees of freedom to zero velocity.

    This task is typically used as an equality constraint to prevent specific
    joints from moving. It enforces zero velocity on the selected DOFs.

    Attributes:
        dof_indices: List of DOF indices to freeze (zero velocity).

    Example:

    .. code-block:: python

        # Freeze specific DOFs by index.
        dof_freezing_task = DofFreezingTask(
            model=model,
            dof_indices=[0, 1, 2]  # Freeze first 3 DOFs
        )

        # Use as equality constraint in IK solver.
        v = solve_ik(
            configuration=configuration,
            tasks=[frame_task, com_task],
            constraints=[dof_freezing_task],  # Enforce exactly
            dt=dt,
            solver="proxqp",
        )

        # Freeze specific joints by name.
        joint_names = ["shoulder_pan", "shoulder_lift"]
        dof_indices = []
        for joint_name in joint_names:
            joint_id = model.joint(joint_name).id
            dof_adr = model.jnt_dofadr[joint_id]
            dof_indices.append(dof_adr)

        dof_freezing_task = DofFreezingTask(model=model, dof_indices=dof_indices)
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        dof_indices: list[int],
        gain: float = 1.0,
    ):
        """Initialize the DOF freezing task.

        Args:
            model: MuJoCo model.
            dof_indices: List of DOF indices to freeze.
            gain: Task gain (typically 1.0 for equality constraints).
        """
        if not dof_indices:
            raise TaskDefinitionError(
                f"{self.__class__.__name__} requires at least one DOF index."
            )

        # Check that all DOF indices are valid.
        for dof_idx in dof_indices:
            if dof_idx < 0 or dof_idx >= model.nv:
                raise TaskDefinitionError(
                    f"DOF index {dof_idx} is out of range [0, {model.nv})."
                )

        # Check for duplicates.
        if len(dof_indices) != len(set(dof_indices)):
            raise TaskDefinitionError(f"Duplicate DOF indices found: {dof_indices}.")

        self.dof_indices = sorted(dof_indices)
        self.nv = model.nv
        self.k = len(dof_indices)

        super().__init__(
            cost=np.ones(self.k),
            gain=gain,
            lm_damping=0.0,
        )

        # Cache error and Jacobian since they don't depend on configuration.
        self._error = np.zeros(self.k)
        self._jacobian = np.zeros((self.k, self.nv))
        for i, dof_idx in enumerate(self.dof_indices):
            self._jacobian[i, dof_idx] = 1.0

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        """Compute the DOF freezing task error.

        The error is always zero since we're constraining velocity, not position.
        When used as an equality constraint with zero error, this enforces
        Δq[dof] = 0 for each frozen DOF.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Zero vector of shape :math:`(k,)` where :math:`k` is the number
            of frozen DOFs.
        """
        return self._error

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        """Compute the DOF freezing task Jacobian.

        The Jacobian is a matrix with one row per frozen DOF, where each row
        is a standard basis vector (row of the identity matrix) selecting the
        corresponding DOF.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Jacobian matrix of shape :math:`(k, n_v)` where :math:`k` is the
            number of frozen DOFs and :math:`n_v` is the number of velocity DOFs.
        """
        return self._jacobian
