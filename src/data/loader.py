"""
DataLoader for CIC-IDS2017 dataset.

Loads all 8 CSV files, fixes known data quirks, and returns a single
clean DataFrame ready for the preprocessing pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class CICIDSLoader:
    """
    Loads and initially cleans the CIC-IDS2017 dataset.

    Usage:
        loader = CICIDSLoader()
        df = loader.load()
    """

    # These columns are unique identifiers, not network behaviour features.
    # Keeping them would let the model memorise IP addresses instead of
    # learning actual attack patterns — a form of data leakage.
    COLUMNS_TO_DROP = [
        'Flow ID', 'Source IP', 'Source Port',
        'Destination IP', 'Destination Port', 'Timestamp',
    ]

    def __init__(self, config_path: str = 'configs/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.data_dir     = Path(self.config['data']['raw_dir'])
        self.random_state = self.config['project']['random_state']

    def load(self) -> pd.DataFrame:
        """
        Load all available CSV files from data/raw/ into a combined DataFrame.
        Files not yet downloaded are skipped with a warning.
        """
        files = self.config['data']['files']
        dfs   = []

        for day_key, fname in files.items():
            fpath = self.data_dir / fname
            if not fpath.exists():
                logger.warning(f"Not found: {fname}  —  skipping")
                continue

            df_day = pd.read_csv(fpath, encoding='utf-8', low_memory=False)
            df_day['source_day'] = day_key
            dfs.append(df_day)
            logger.info(
                f"Loaded  {fname.split('WorkingHours')[0].rstrip('-'):<38}"
                f"  {len(df_day):>9,} rows"
            )

        if not dfs:
            raise FileNotFoundError(
                f"No dataset files found in {self.data_dir.resolve()}.\n"
                "Make sure the CSV files from CIC-IDS2017 are in data/raw/."
            )

        df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined: {len(df):,} rows  x  {len(df.columns)} columns")
        return self._initial_clean(df)

    # ── Private methods ──────────────────────────────────────────────────────

    def _initial_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fix the four known quirks of the CIC-IDS2017 CSV export:

        Quirk 1: All column names have a leading space  e.g. ' Flow Duration'
        Quirk 2: Label column may be ' Label' (with leading space)
        Quirk 3: Label values have leading/trailing spaces
        Quirk 4: Flow Bytes/s and Flow Packets/s contain infinity (division by zero
                 when flow duration is 0). These must become NaN before modelling.
        """
        # Fix 1 & 2
        df.columns = df.columns.str.strip()
        if ' Label' in df.columns:
            df.rename(columns={' Label': 'Label'}, inplace=True)

        # Fix 3
        df['Label'] = df['Label'].str.strip()

        # Fix 4
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_count    = np.isinf(df[numeric_cols]).sum().sum()
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        logger.info(f"Replaced {inf_count:,} infinity values with NaN")

        # Drop identifier columns
        to_drop = [c for c in self.COLUMNS_TO_DROP if c in df.columns]
        df.drop(columns=to_drop, inplace=True)

        # Remove exact duplicate rows
        before = len(df)
        df.drop_duplicates(inplace=True)
        dropped = before - len(df)
        if dropped > 0:
            logger.info(f"Removed {dropped:,} duplicate rows")

        logger.info("Initial cleaning complete")
        return df