import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from index_cache import create_index_for_chunks


class IndexCacheTests(unittest.TestCase):
    def test_indexes_are_batched_and_reused_from_disk(self):
        chunks = [f"chunk {index}" for index in range(35)]
        batch_sizes = []

        def embedder(batch):
            batch_sizes.append(len(batch))
            return np.ones((len(batch), 384), dtype="float32")

        with tempfile.TemporaryDirectory() as cache_directory:
            with patch("index_cache.INDEX_CACHE_DIR", Path(cache_directory)):
                index = create_index_for_chunks(chunks, embedder)
                self.assertEqual(index.ntotal, len(chunks))
                self.assertEqual(batch_sizes, [32, 3])

                reused_index = create_index_for_chunks(
                    chunks,
                    lambda _batch: self.fail("cached index should avoid embedding"),
                )

        self.assertEqual(reused_index.ntotal, len(chunks))


if __name__ == "__main__":
    unittest.main()
