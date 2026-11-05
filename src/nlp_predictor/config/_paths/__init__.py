# LEAVE THIS AS IT IS IF YOU DONT'T KNOW WHAT IT DOES
# We don't want to expose internal classes used only
# for config loading to config's scope

from .paths import Paths

__all__ = ["Paths"]
