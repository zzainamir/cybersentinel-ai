"""
Preprocessing pipeline for CIC-IDS2017.

Pipeline order (order matters — do not rearrange):
  A. Create labels (binary and multi-class)
  B. Impute NaN values
  C. Remove useless features (zero variance + high correlation)
  D. Split into train / val / test     <-- MUST happen before scaling and SMOTE
  E. Scale features                    <-- Fit on train only to prevent leakage
  F. Create model-specific training sets

Output files (saved to data/features/):
  X_train.parquet          SMOTE-balanced, for LSTM binary classifier
  X_train_benign.parquet   Benign-only, for Isolation Forest and Autoencoder
  X_train_full.parquet     Full unbalanced training set, for Random Forest
  y_binary_train.parquet   Binary labels matching X_train (SMOTE'd)
  y_multi_train.parquet    Multi-class labels matching X_train_full
  X_val / X_test           Unmodified — reflects real-world class proportions
  y_binary_val/test
  y_multi_val/test
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import json
import yaml
import joblib
import logging

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling   import SMOTE
from imblearn.under_sampling  import RandomUnderSampler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class CICIDSPreprocessor:
    """
    Full preprocessing pipeline for CIC-IDS2017.

    Usage:
        preprocessor = CICIDSPreprocessor()
        splits = preprocessor.fit_transform(df)
        preprocessor.save_splits(splits)
        preprocessor.save_artifacts()
    """

    def __init__(self, config_path: str = 'configs/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.random_state  = self.config['project']['random_state']
        self.test_size     = self.config['training']['test_size']
        self.val_size      = self.config['training']['val_size']
        self.features_dir  = Path(self.config['data']['features_dir'])
        self.processed_dir = Path(self.config['data']['processed_dir'])

        # Fitted on training data — saved for use during inference (Phase 4)
        self.scaler           = StandardScaler()
        self.label_encoder    = LabelEncoder()
        self.selected_features = None

    # ── Public interface ─────────────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame) -> dict:
        """Run the full pipeline and return a dict of all splits."""
        logger.info("=" * 55)
        logger.info("  PREPROCESSING PIPELINE STARTING")
        logger.info("=" * 55)

        X, y_binary, y_multi = self._create_labels(df)
        X                    = self._impute_nan(X)
        X, features          = self._select_features(X)
        self.selected_features = features
        logger.info(f"Features after selection: {len(features)}")

        splits = self._split(X, y_binary, y_multi)
        splits = self._scale(splits)
        splits = self._create_model_sets(splits)

        logger.info("=" * 55)
        logger.info("  PIPELINE COMPLETE")
        logger.info("=" * 55)
        return splits

    def save_splits(self, splits: dict) -> None:
        """Save all splits to Parquet files in data/features/."""
        self.features_dir.mkdir(parents=True, exist_ok=True)

        for name, data in splits.items():
            if data is None:
                continue
            path = self.features_dir / f'{name}.parquet'
            if isinstance(data, np.ndarray):
                if data.ndim == 1:
                    pd.DataFrame({'value': data}).to_parquet(path, index=False)
                else:
                    pd.DataFrame(data).to_parquet(path, index=False)
            logger.info(f"  Saved  {name:<35}  shape: {data.shape}")

    def save_artifacts(self, output_dir: str = 'models/saved') -> None:
        """Save fitted scaler, label encoder, and feature list for inference."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler,             out / 'scaler.pkl')
        joblib.dump(self.label_encoder,      out / 'label_encoder.pkl')
        joblib.dump(self.selected_features,  out / 'selected_features.pkl')
        logger.info(f"Artifacts saved to {output_dir}/")

    # ── Private pipeline steps ───────────────────────────────────────────────

    def _create_labels(self, df):
        """
        Create two label formats:
          y_binary : 0 = benign, 1 = attack
          y_multi  : integer per attack type (0=Benign, 1=DDoS, 2=DoS, ...)
        """
        y_binary = (df['Label'] != 'BENIGN').astype(int).values
        y_multi  = self.label_encoder.fit_transform(df['Label'].values)

        drop_cols = [c for c in ['Label', 'source_day'] if c in df.columns]
        X = df.drop(columns=drop_cols).select_dtypes(include=[np.number])

        logger.info(f"Binary:  {Counter(y_binary)}")
        logger.info(f"Classes: {df['Label'].nunique()} unique attack types")
        return X, y_binary, y_multi

    def _impute_nan(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fill NaN values with the column median.
        Median is preferred over mean for network data because many features
        are heavily right-skewed (a few huge values distort the mean).
        """
        nan_cols = X.isnull().sum()
        nan_cols = nan_cols[nan_cols > 0]
        if len(nan_cols) > 0:
            logger.info(f"Imputing {len(nan_cols)} columns with median...")
            for col in nan_cols.index:
                X[col] = X[col].fillna(X[col].median())
        else:
            logger.info("No NaN values to impute")
        return X

    def _select_features(self, X: pd.DataFrame):
        """
        Two-step feature reduction:

        Step 1 — Remove near-zero-variance features.
          Features that are almost the same value for every row carry
          no discriminative power for a classifier.

        Step 2 — Remove highly correlated features (r > 0.98).
          When two features are nearly identical, one is redundant.
          Removing it speeds up training without losing information.
        """
        # Step 1: variance threshold
        var_sel = VarianceThreshold(threshold=0.01)
        X_var   = var_sel.fit_transform(X)
        kept    = X.columns[var_sel.get_support()].tolist()
        logger.info(f"Removed {len(X.columns) - len(kept)} zero-variance features")

        # Step 2: correlation filter
        df_tmp   = pd.DataFrame(X_var, columns=kept)
        corr_mat = df_tmp.corr().abs()
        upper    = corr_mat.where(
            np.triu(np.ones(corr_mat.shape), k=1).astype(bool)
        )
        to_drop  = [c for c in upper.columns if any(upper[c] > 0.98)]
        df_tmp.drop(columns=to_drop, inplace=True)
        logger.info(f"Removed {len(to_drop)} highly correlated features (r > 0.98)")

        return df_tmp.values, df_tmp.columns.tolist()

    def _split(self, X, y_binary, y_multi) -> dict:
        """
        Stratified 70 / 15 / 15 split.
        Stratified means each split has the same benign:attack ratio as the full
        dataset — without this, you could get a test set with no attacks at all.
        """
        temp = self.test_size + self.val_size
        val_ratio = self.val_size / temp

        X_tr, X_tmp, yb_tr, yb_tmp, ym_tr, ym_tmp = train_test_split(
            X, y_binary, y_multi,
            test_size=temp,
            random_state=self.random_state,
            stratify=y_binary
        )
        X_val, X_test, yb_val, yb_test, ym_val, ym_test = train_test_split(
            X_tmp, yb_tmp, ym_tmp,
            test_size=(1 - val_ratio),
            random_state=self.random_state,
            stratify=yb_tmp
        )
        logger.info(
            f"Split → Train: {len(X_tr):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}"
        )
        return {
            'X_train':        X_tr,   'y_binary_train': yb_tr,  'y_multi_train': ym_tr,
            'X_val':          X_val,  'y_binary_val':   yb_val, 'y_multi_val':   ym_val,
            'X_test':         X_test, 'y_binary_test':  yb_test,'y_multi_test':  ym_test,
        }

    def _scale(self, splits: dict) -> dict:
        """
        StandardScaler: subtracts mean and divides by std deviation so every
        feature ends up with mean=0 and std=1.

        CRITICAL: fit (learn mean and std) ONLY on the training set.
        Then apply those same values to val and test.
        If you fit on the full dataset, the model has indirectly seen the test
        set statistics — this is called data leakage and inflates your results.
        """
        splits['X_train'] = self.scaler.fit_transform(splits['X_train'])
        splits['X_val']   = self.scaler.transform(splits['X_val'])
        splits['X_test']  = self.scaler.transform(splits['X_test'])
        logger.info("Scaling complete — StandardScaler fitted on training set only")
        return splits

    def _create_model_sets(self, splits: dict) -> dict:
        """
        Different models need different training data formats:

          Isolation Forest  → benign-only (learns what 'normal' looks like)
          Autoencoder       → benign-only (learns to reconstruct normal traffic)
          Random Forest     → full imbalanced data + class_weight='balanced'
          LSTM              → SMOTE-balanced binary data

        This method produces a separate array for each use case.
        """
        X      = splits['X_train']
        y_bin  = splits['y_binary_train']
        y_mult = splits['y_multi_train']

        # ── Benign-only for Isolation Forest and Autoencoder ──────────────
        benign_mask = (y_bin == 0)
        splits['X_train_benign'] = X[benign_mask]
        logger.info(f"Benign-only set: {splits['X_train_benign'].shape[0]:,} samples")

        # ── Full unbalanced set + multi-class labels for Random Forest ────
        splits['X_train_full']   = X
        splits['y_multi_train_full'] = y_mult
        logger.info(f"Full training set (RF): {X.shape[0]:,} samples")

        # ── SMOTE-balanced binary set for LSTM ───────────────────────────
        # Step 1: undersample benign to 3:1 ratio (makes SMOTE feasible in RAM)
        attack_n       = Counter(y_bin)[1]
        target_benign  = min(attack_n * 3, Counter(y_bin)[0])
        undersampler   = RandomUnderSampler(
            sampling_strategy={0: target_benign, 1: attack_n},
            random_state=self.random_state
        )
        X_under, y_under = undersampler.fit_resample(X, y_bin)
        logger.info(f"After undersampling: {Counter(y_under)}")

        # Step 2: SMOTE to reach 1:1 balance
        smote = SMOTE(random_state=self.random_state, k_neighbors=5)
        X_bal, y_bal = smote.fit_resample(X_under, y_under)
        logger.info(f"After SMOTE:         {Counter(y_bal)}")

        splits['X_train']        = X_bal
        splits['y_binary_train'] = y_bal
        return splits