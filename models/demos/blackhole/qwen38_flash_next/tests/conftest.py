# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import re
from contextlib import contextmanager

import pytest


@contextmanager
def _expect_error(expected_exception, match=None):
    try:
        yield
    except expected_exception as exception:
        if match is not None and re.search(match, str(exception)) is None:
            raise AssertionError(f"Exception message did not match {match!r}: {exception}") from exception
    else:
        raise AssertionError(f"Expected {expected_exception} to be raised")


@pytest.fixture
def expect_error():
    return _expect_error
