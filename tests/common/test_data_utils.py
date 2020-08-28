import re

from hypothesis import given
from hypothesis.strategies import booleans, text

from declutr.common.data_utils import sanitize


class TestDataUtils:
    @given(text=text(), lowercase=booleans())
    def test_sanitize(self, text: str, lowercase: bool) -> None:
        sanitized_text = sanitize(text, lowercase=lowercase)

        # There should be no cases of multiple spaces or tabs
        assert re.search(r"[ ]{2,}", sanitized_text) is None
        assert "\t" not in sanitized_text
        # The beginning and end of the string should be stripped of whitespace
        assert not sanitized_text.startswith(("\n", " "))
        assert not sanitized_text.endswith(("\n", " "))
        # Sometimes, hypothesis generates text that cannot be lowercased (like latin characters).
        # We don't particularly care about this, and it breaks this check.
        # Only run if the generated text can be lowercased.
        if lowercase and text.lower().islower():
            assert all(not char.isupper() for char in sanitized_text)
