# Author:
#   Vihaan Sahu <pteroisvolitans12@gmail.com>


import sys
import sklearn.linear_model

# Compatibility fix for old scikit-learn pickle files on Python 3.13+
# This maps the old module path to the new one so pickle.load() works.
if 'sklearn.linear_model.logistic' not in sys.modules:
    sys.modules['sklearn.linear_model.logistic'] = sklearn.linear_model

    
import json
import pickle
import glob
import numpy as np
from pathlib import Path
from scipy.sparse import hstack
from sklearn.feature_extraction.text import HashingVectorizer
from rct_reviewer import get_data_path
from rct_reviewer.ml.classifier import MiniClassifier
import logging

log = logging.getLogger(__name__)


DEFAULT_CALIBRATION = {
    "thresholds": {
        "svm": {"balanced": 0.0, "precise": 0.5, "sensitive": -0.5},
        "svm_cnn": {"balanced": 0.0, "precise": 0.5, "sensitive": -0.5}
    },
    "scales": {
        "svm": {"mean": 0.0, "std": 1.0, "weight": 1.0},
        "cnn": {"mean": 0.0, "std": 1.0, "weight": 1.0},
        "ptyp": {"mean": 0.0, "std": 1.0}
    }
}

class RCTRobot:
    def __init__(self):
        log.info("Loading RCT models...")
        
        # Load SVM
        self.svm_clf = MiniClassifier('rct/rct_svm_weights.npz')
        self.svm_vectorizer = HashingVectorizer(binary=False, ngram_range=(1, 1), stop_words='english')
        
        # Load calibration JSON
        calib_path = get_data_path('rct/rct_model_calibration.json')
        if calib_path.exists():
            try:
                with open(calib_path, 'r') as f:
                    self.constants = json.load(f)
            except Exception as e:
                log.warning(f"Failed to load calibration JSON: {e}. Using defaults.")
                self.constants = DEFAULT_CALIBRATION
        else:
            log.warning("rct_model_calibration.json not found. Using defaults.")
            self.constants = DEFAULT_CALIBRATION
        
        # Load pickle calibration files (may fail due to sklearn version mismatch)
        self.calibration = {}
        pkl_files = [
            ('svm_cnn', 'rct/svm_cnn_calibration.pck'),
            ('svm_cnn_ptyp', 'rct/svm_cnn_ptyp_calibration.pck')
        ]
        for key, fname in pkl_files:
            fpath = get_data_path(fname)
            if fpath.exists():
                try:
                    with open(fpath, 'rb') as f:
                        self.calibration[key] = pickle.load(f)
                    log.info(f"Loaded calibration: {fname}")
                except Exception as e:
                    log.warning(f"Could not load {fname} (sklearn version mismatch?): {e}")
                    log.warning("Probability calibration will use sigmoid fallback.")

        # Try loading CNN (TensorFlow)
        self.cnn_models = []
        self.vocab_map = None
        self._load_cnn()

    def _load_cnn(self):
        try:
            import tensorflow as tf
            tf.get_logger().setLevel('ERROR')
            from tensorflow.keras.models import load_model
            
            cnn_files = glob.glob(str(get_data_path('rct/*.h5')))
            if cnn_files:
                log.info(f"Loading {len(cnn_files)} CNN models...")
                self.cnn_models = [load_model(f) for f in cnn_files]
                
                vocab_path = get_data_path('rct/cnn_vocab_map.pck')
                if vocab_path.exists():
                    with open(vocab_path, 'rb') as f:
                        self.vocab_map = pickle.load(f)
        except ImportError:
            log.warning("TensorFlow not found. Using SVM-only mode.")
        except Exception as e:
            log.warning(f"CNN Load Error: {e}. Using SVM-only mode.")

    def predict(self, title: str, abstract: str):
        text_ab = f"{title}\n\n{abstract}"
        
        X_ti = self.svm_vectorizer.transform([title])
        X_ab = self.svm_vectorizer.transform([text_ab])
        X_svm = hstack([X_ab, X_ti])
        svm_score = self.svm_clf.decision_function(X_svm)[0]
        
        scales = self.constants['scales']['svm']
        score = (svm_score - scales['mean']) / scales['std']
        
        threshold = self.constants['thresholds']['svm']['balanced']
        
        # Probability (sigmoid fallback if calibration models didn't load)
        prob = float(1.0 / (1.0 + np.exp(-score)))
        
        return {
            "is_rct": bool(score >= threshold),
            "score": float(score),
            "probability": prob,
            "model": "svm"
        }