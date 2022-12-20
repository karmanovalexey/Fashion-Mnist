import configparser
import os
import unittest
import pandas as pd
import sys

sys.path.insert(1, os.path.join(os.getcwd(), "src"))

from train import Trainer

config = configparser.ConfigParser()
config.read("config.ini")


class TestMultiModel(unittest.TestCase):

    def setUp(self) -> None:
        self.trainer = Trainer()

    def test_log_reg(self):
        self.assertEqual(self.trainer.train('logreg'), True)
        self.assertGreater(self.trainer.logreg(benchmark=True) > 0.1)

    def test_svm(self):
        self.assertEqual(self.trainer.train('svm'), True)
        self.assertGreater(self.trainer.svm(benchmark=True) > 0.1)


if __name__ == "__main__":
    unittest.main()
