import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("threelinks.xml")
data  = mujoco.MjData(model)

mujoco.viewer.launch(model, data)  # opens GUI, blocks until closed