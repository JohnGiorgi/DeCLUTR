import re

from hypothesis import given
from hypothesis.strategies import text

from declutr.common.data_utils import sanitize


class TestDataUtils:
    @given(text=text())
    def test_sanitize(self, text: str):
        sanitized_text = sanitize(text)

        # There should be no cases of multiple spaces or tabs
        assert re.search(r"[ ]{2,}", sanitized_text) is None
        assert "\t" not in sanitized_text
        # The beginning and end of the string should be stripped of whitespace
        assert not sanitized_text.startswith(("\n", " "))
        assert not sanitized_text.endswith(("\n", " "))
