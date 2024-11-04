"""Test module loading."""

import unittest

from steer.version import get_version


class TestModyke(unittest.TestCase):
    """Test modules load."""

    def test_version_type(self):
        """Test the version is a string.

        This is only meant to be an example test.
        """
        from steer.llm import Heuristic
        from steer.utils import get_rxn_img

        assert True
