import unittest
from pathlib import Path

from classifier import NearestCentroidClassifier, accuracy, load_csv, train

DATA = Path(__file__).parents[1] / "data" / "tracks.csv"


class MusicGenreTests(unittest.TestCase):
    def test_dataset_loads(self):
        rows, labels = load_csv(DATA)
        self.assertEqual(len(rows), 16)
        self.assertEqual(set(labels), {"jazz", "pop", "rock", "classical"})

    def test_classifier_learns_genre_profiles(self):
        model, score = train(DATA)
        self.assertGreaterEqual(score, 0.65)
        self.assertEqual(model.predict_one([128, 0.82, 0.86, 0.08]), "pop")
        self.assertEqual(model.predict_one([78, 0.38, 0.28, 0.82]), "classical")

    def test_prediction_count(self):
        rows, _ = load_csv(DATA)
        model, _ = train(DATA)
        self.assertEqual(len(model.predict(rows)), 16)

    def test_invalid_input_rejected(self):
        with self.assertRaises(ValueError):
            NearestCentroidClassifier().fit([], [])

    def test_accuracy(self):
        self.assertEqual(accuracy(["jazz", "rock"], ["jazz", "jazz"]), 0.5)


if __name__ == "__main__":
    unittest.main()
