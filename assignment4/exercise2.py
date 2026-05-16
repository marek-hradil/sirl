import equinox as eqx
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx
from scipy.optimize import minimize

UR5E_XML = "./ur5e.xml"


class UR5E(eqx.Module):
    model: mjx.Model
    data: mjx.Data
    mj_model: mujoco.MjModel = eqx.field(static=True)
    mj_data: mujoco.MjData = eqx.field(static=True)
    site_id: int = eqx.field(static=True)
    renderer: mujoco.Renderer = eqx.field(static=True)

    @classmethod
    def from_xml(cls, path: str):
        mj_model = mujoco.MjModel.from_xml_path(path)
        mj_data = mujoco.MjData(mj_model)
        return cls(
            model=mjx.put_model(mj_model),
            data=mjx.put_data(mj_model, mj_data),
            mj_model=mj_model,
            mj_data=mj_data,
            site_id=mj_model.site("attachment_site").id,
            renderer=mujoco.Renderer(mj_model),
        )

    def sync(self, q):
        self.mj_data.qpos[:6] = np.array(q)
        mujoco.mj_forward(self.mj_model, self.mj_data)

    @eqx.filter_jit
    def forward(self, q):
        d = mjx.fwd_position(self.model, self.data.replace(qpos=q))
        return d.site_xpos[self.site_id], d.site_xmat[self.site_id]

    @eqx.filter_jit
    def pose_err(self, q, target_pos, target_mat):
        pos, mat = self.forward(q)
        return jnp.sum((target_pos - pos) ** 2) + jnp.sum((target_mat - mat) ** 2)

    def compute_target_q(self, target_pos, target_mat, init_guess):
        grad_fn = eqx.filter_jit(eqx.filter_grad(self.pose_err))
        result = minimize(
            self.pose_err,
            x0=init_guess,
            args=(target_pos, target_mat),
            method="L-BFGS-B",
            jac=grad_fn,
        )
        return jnp.array(result.x)


model = UR5E.from_xml(UR5E_XML)


def test():
    test_q = jnp.deg2rad(jnp.array([-90.0, -60.0, 90.0, -30.0, 0.0, 0.0]))
    pos, rot = model.forward(test_q)
    print("start pos:", pos)
    print("start rot:", rot)

    target_pos = pos + jnp.array([0.0, 0.0, 0.2])
    target_rot = rot

    target_q = model.compute_target_q(target_pos, target_rot, init_guess=test_q)
    res_pos, res_rot = model.forward(target_q)
    print("solved pos:", res_pos)
    print("solved rot:", res_rot)
    print("pos error:", jnp.linalg.norm(res_pos - target_pos))


if __name__ == "__main__":
    test()
