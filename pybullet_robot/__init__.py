from importlib.metadata import PackageNotFoundError, version

from .bullet_robot import BulletRobot
from .pybullet_ik_interface import PybulletIKInterface

try:
    __version__ = version("pybullet_robot")
except PackageNotFoundError:
    # package not installed (e.g. running from a source checkout)
    __version__ = "0.0.0"

__all__ = ["BulletRobot", "PybulletIKInterface", "__version__"]
