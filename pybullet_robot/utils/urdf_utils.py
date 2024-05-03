"""Utils related to urdf files."""

from typing import List
import logging
import importlib
from robot_descriptions._descriptions import DESCRIPTIONS as _AR_DESCRIPTIONS


AWESOME_ROBOTS: List[str] = _AR_DESCRIPTIONS.keys()
"""List of all available robots from the Awesome Robots List that can be used."""


def get_urdf_from_awesome_robot_descriptions(
    robot_description_pkg_name: str,
) -> str:
    """Get robot urdf for the specified robot description package.

    The specified package should be from the list of awesome robot descriptions
    (https://github.com/robot-descriptions/robot_descriptions.py/tree/main?tab=readme-ov-file#descriptions).

    The list of available robot descriptions can also be viewed in the variable `AWESOME_ROBOTS`
    imported from `utils.urdf_utils`.

    Downloads description package for the specified robot and caches it locally (only needs
    downloadin once).

    Args:
        robot_description_pkg_name (str): The package name as specified in the Awesome Robot
            Descriptions list.

    Returns:
        str: Path to the robot's urdf file.
    """
    try:
        return importlib.import_module(
            f"robot_descriptions.{robot_description_pkg_name}"
        ).URDF_PATH
    except AttributeError as e:
        msg = f"URDF has not been provided for robot {robot_description_pkg_name}."
        logging.error(msg)
        raise AttributeError(msg) from e
    except (ModuleNotFoundError, KeyError) as e:
        msg = (
            f"No description package called {robot_description_pkg_name} in"
            " Awesome Robot Descriptions list. Use on of the description names from "
            "https://github.com/robot-descriptions/robot_descriptions.py/tree/main?tab=readme-ov-file#descriptions."
        )
        logging.error(msg)
        raise ModuleNotFoundError(msg) from e
