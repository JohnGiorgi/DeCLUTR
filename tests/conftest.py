import pytest

from declutr.encoder import Encoder


@pytest.fixture
def declutr_small():
    return Encoder("declutr-small")
