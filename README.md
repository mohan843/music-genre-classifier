# Music Genre Classifier

An audio-machine-learning mini-project that classifies tracks as jazz, pop, rock, or classical from tempo, danceability, energy, and acousticness features.

## Overview

Music discovery systems often use acoustic descriptors alongside audio embeddings. This project demonstrates an interpretable nearest-centroid baseline that maps compact numerical features to broad genre profiles.

## Problem statement

Given acoustic features for a track, predict the closest genre profile for exploratory tagging and recommendation experiments.

## Features

- Nearest-centroid classification from scratch.
- Acoustic feature profiles for four genres.
- CSV sample data, CLI inference, and tests.
- Standard-library-only implementation.

## Technology stack

- Python 3.10+
- Audio ML and feature-based classification
- Euclidean distance and centroid profiles
- CSV, `argparse`, and `pathlib`
- `unittest` for automated testing

## Installation

```bash
git clone https://github.com/mohan843/music-genre-classifier.git
cd music-genre-classifier
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

No third-party runtime dependencies are required.

## Execution

```bash
python classifier.py --tempo-bpm 128 --danceability 0.82 --energy 0.86 --acousticness 0.08
```

## Testing

```bash
python -m unittest discover -s tests -v
```

## Sample output

```text
Training accuracy: 68.8%
Predicted music genre: pop
```

## Limitations and responsible use

The dataset is synthetic and genres are broad cultural categories rather than objective truths. Real systems need diverse catalogs, audio embeddings, temporal features, artist context, evaluation for novelty and diversity, and careful handling of listener data.

## Future improvements

- Extract features directly from audio files.
- Compare k-nearest neighbors, SVMs, and neural embeddings.
- Add multilabel tags and subgenre classification.
- Evaluate catalog diversity and recommendation quality.
- Build an interactive audio-tagging dashboard.
