import unittest
from evaluation.metrics import score_content, summarize_content


class ContentMetricsTests(unittest.TestCase):
    def score(self, task, ref, generated, **meta):
        return score_content({"task": task, "answer": ref, "meta": meta}, generated)

    def test_count_equivalent_numbers(self):
        for text in ("图中有二只猫。", "2", "图中有两只猫。图中有两只猫。"):
            self.assertTrue(self.score("counting", "图中有两个猫。", text,
                                       count=2, **{"class": "cat"})["correct"])
        self.assertTrue(self.score("counting", "图中有十二个人。", "12", count=12)["correct"])

    def test_count_wrong_and_contradictory(self):
        self.assertEqual(self.score("counting", "图中有两个猫。", "图中有三个猫。", count=2)["status"], "wrong")
        for text in ("两只猫，三只猫", "可能有两只猫", "有两只狗", "猫", "没有两只猫"):
            self.assertFalse(self.score("counting", "图中有两个猫。", text,
                                        count=2, **{"class": "cat"})["parsed"])

    def test_numeric_punctuation_cannot_merge_conflicting_values(self):
        for text in ("1，2", "-12", "12.0"):
            self.assertFalse(self.score("counting", "12", text, count=12)["parsed"])
        self.assertTrue(self.score("counting", "2", "2。2。", count=2)["correct"])

    def test_existence_polarity(self):
        for text in ("没有", "否", "图中没有猫。"):
            self.assertTrue(self.score("existence", "图中没有猫。", text, label=0,
                                       **{"class": "cat"})["correct"])
        self.assertEqual(self.score("existence", "图中没有猫。", "有猫", label=0)["status"], "wrong")
        for text in ("有猫，没有猫", "也许有猫", "没有狗", "猫"):
            self.assertFalse(self.score("existence", "图中没有猫。", text, label=0,
                                        **{"class": "cat"})["parsed"])

    def test_spatial_role_and_contradiction(self):
        for text in ("左侧", "狗在人的左边。", "左边左边"):
            self.assertTrue(self.score("spatial", "狗在人的左边。", text,
                                       subject="dog", other="person", answer="左边")["correct"])
        for text in ("人在狗的左边", "狗不在人的左边", "左边还是右边", "左边右边"):
            self.assertFalse(self.score("spatial", "狗在人的左边。", text,
                                        subject="dog", other="person", answer="左边")["parsed"])

    def test_listing_order_duplicates_and_partial_match(self):
        meta = {"classes": ["cat", "dog"]}
        self.assertTrue(self.score("listing", "图中有猫、狗。", "狗和猫，猫", **meta)["correct"])
        score = self.score("listing", "图中有猫、狗。", "猫和人", **meta)
        self.assertEqual(score["status"], "wrong")
        self.assertEqual(score["f1"], 0.5)
        for text in ("没有猫和狗", "猫和狗和大象", "可能有猫和狗"):
            self.assertFalse(self.score("listing", "图中有猫、狗。", text, **meta)["parsed"])

    def test_attribute_and_repetition_are_separate(self):
        score = self.score("attribute", "图片主要是猫。", "图片主要是猫。主要是猫。",
                           **{"class": "cat"})
        self.assertTrue(score["correct_nonexact"])
        self.assertTrue(score["reference_prefix_extra"])
        self.assertFalse(self.score("attribute", "图片主要是猫。", "主要是猫和狗",
                                    **{"class": "cat"})["parsed"])
        repeated = self.score("attribute", "图片主要是猫。", "猫。猫。", **{"class": "cat"})
        self.assertTrue(repeated["repetition_detected"])
        self.assertTrue(repeated["correct"])

    def test_positive_quantified_objects(self):
        for text in ("图片主要是一个人。", "图片主要是两个人。"):
            self.assertTrue(self.score("attribute", "图片主要是人。", text, **{"class":"person"})["correct"])
        self.assertTrue(self.score("existence", "有火车", "是的，图中有一个火车。", label=1, **{"class":"train"})["correct"])
        self.assertTrue(self.score("listing", "猫和狗", "图中包括两只猫和一只狗。", classes=["cat","dog"])["correct"])

    def test_quantifier_normalization_remains_conservative(self):
        for text in ("图中有零只猫", "图中有0只猫", "图中有-1只猫", "图中有1.5只猫", "图中没有两只猫", "可能有一只猫", "图中有一只猫和一只大象", "图中有一只猫和一只狗"):
            self.assertFalse(self.score("existence", "有猫", text, label=1, **{"class":"cat"})["parsed"],text)
        self.assertFalse(self.score("attribute", "人", "图片主要是一个人和一只狗", **{"class":"person"})["parsed"])
        self.assertEqual(self.score("counting", "一只猫", "两只猫", count=1, **{"class":"cat"})["status"],"wrong")

    def test_summary_counts_unparsed_as_incorrect(self):
        scores = [self.score("counting", "1", x, count=1) for x in ("1", "2", "不知道")]
        summary = summarize_content(scores)
        self.assertAlmostEqual(summary["correct"], 1/3)
        self.assertAlmostEqual(summary["parsed"], 2/3)
        self.assertEqual(summary["status_counts"], {"correct": 1, "wrong": 1, "unparsed": 1})


if __name__ == "__main__":
    unittest.main()
