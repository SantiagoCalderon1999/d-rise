# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Module for creating explanations for vision models."""

from .version import name, version
from .DRISE_runner import get_saliency_map

__name__ = name
__version__ = version
