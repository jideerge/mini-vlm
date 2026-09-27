import unittest
from scripts.audit_spatial import relation_options


class SpatialAuditTests(unittest.TestCase):
    def test_single_pair_is_clear(self):
        self.assertEqual(relation_options([
            {"name": "dog", "cx": 20}, {"name": "person", "cx": 80}],
            "dog", "person"), ({"左边"}, 1, 1))

    def test_multiple_pairs_can_contradict(self):
        self.assertEqual(relation_options([
            {"name": "dog", "cx": 20}, {"name": "dog", "cx": 90},
            {"name": "person", "cx": 50}], "dog", "person"),
            ({"左边", "右边"}, 2, 1))

    def test_equal_centers_do_not_define_left_or_right(self):
        self.assertEqual(relation_options([
            {"name": "dog", "cx": 50}, {"name": "person", "cx": 50}],
            "dog", "person"), (set(), 1, 1))


if __name__ == "__main__":
    unittest.main()
