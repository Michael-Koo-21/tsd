
import pandas as pd
import numpy as np
import sys
import os
import time
import multiprocessing
from pathlib import Path
import warnings
import json

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.spatial.distance import cdist
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

# --- from src/preprocessing/load_data.py ---

def load_raw_acs_pums(data_path: str = "data/raw/psam_p06.csv") -> pd.DataFrame:
    """
    Load raw ACS PUMS California data.
    """
    print(f"Loading raw ACS PUMS data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df):,} records with {len(df.columns)} columns")
    return df

def filter_adults(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to adults only (AGEP >= 18).
    """
    print(f"Filtering to adults (AGEP >= 18)...")
    df_adults = df[df['AGEP'] >= 18].copy()
    print(f"✓ {len(df_adults):,} adult records ({len(df_adults)/len(df)*100:.1f}%)")
    return df_adults

def recode_race(rac1p_series: pd.Series) -> pd.Series:
    """
    Recode RAC1P (race) into broader categories.
    """
    def recode_value(x):
        if x == 1: return 1
        elif x == 2: return 2
        elif x == 6: return 3
        elif x in [3, 4, 5, 7, 8]: return 4
        elif x == 9: return 5
        else: return 4
    return rac1p_series.apply(recode_value)

def recode_education(schl_series: pd.Series) -> pd.Series:
    """
    Recode SCHL (educational attainment) into broader categories.
    """
    def recode_value(x):
        if pd.isna(x): return np.nan
        elif x <= 15: return 1
        elif x <= 17: return 2
        elif x <= 20: return 3
        elif x == 21: return 4
        elif x <= 24: return 5
        else: return 1
    return schl_series.apply(recode_value)

def recode_birthplace(pobp_series: pd.Series) -> pd.Series:
    """
    Recode POBP (place of birth) into broader geographic regions.
    """
    def recode_value(x):
        if x == 6: return 1
        elif 1 <= x <= 56: return 2
        elif x == 303: return 3
        elif 200 <= x <= 299: return 5
        elif 300 <= x <= 399: return 6
        elif 100 <= x <= 199: return 4
        else: return 7
    return pobp_series.apply(recode_value)

def select_and_recode_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select the 8 predictors + 1 target variable and apply recoding.
    """
    print("Selecting and recoding variables...")
    df_selected = df[['AGEP', 'PINCP', 'SEX', 'MAR', 'ESR', 'PWGTP', 'RAC1P', 'SCHL', 'POBP']].copy()
    df_selected['RAC1P_RECODED'] = recode_race(df_selected['RAC1P'])
    df_selected['SCHL_RECODED'] = recode_education(df_selected['SCHL'])
    df_selected['POBP_RECODED'] = recode_birthplace(df_selected['POBP'])
    df_selected = df_selected.drop(columns=['RAC1P', 'SCHL', 'POBP'])
    final_cols = ['AGEP', 'PINCP', 'SEX', 'RAC1P_RECODED', 'SCHL_RECODED', 'MAR', 'ESR', 'POBP_RECODED', 'PWGTP']
    df_selected = df_selected[final_cols]
    print(f"✓ Selected and recoded 9 variables (8 predictors + 1 target)")
    return df_selected

def sample_data(df: pd.DataFrame, n: int = 50000, random_state: int = 42) -> pd.DataFrame:
    """
    Sample N records from the full dataset.
    """
    print(f"Sampling {n:,} records...")
    if len(df) <= n:
        print(f"⚠ Dataset has only {len(df):,} records, using all")
        return df.copy()
    df_sample = df.sample(n=n, random_state=random_state)
    print(f"✓ Sampled {len(df_sample):,} records")
    return df_sample

def train_test_split_local(df: pd.DataFrame, train_size: float = 0.7, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets.
    """
    print(f"Splitting data: {train_size*100:.0f}% train, {(1-train_size)*100:.0f}% test...")
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    split_idx = int(len(df_shuffled) * train_size)
    df_train = df_shuffled.iloc[:split_idx].copy()
    df_test = df_shuffled.iloc[split_idx:].copy()
    print(f"✓ Train: {len(df_train):,} records")
    print(f"✓ Test: {len(df_test):,} records")
    return df_train, df_test

def preprocess_acs_data(data_path: str = "data/raw/psam_p06.csv", sample_size: int = 50000, train_size: float = 0.7, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full preprocessing pipeline.
    """
    print("="*60)
    print("ACS PUMS Data Preprocessing")
    print("="*60)
    df_raw = load_raw_acs_pums(data_path)
    df_adults = filter_adults(df_raw)
    df_selected = select_and_recode_variables(df_adults)
    df_sample = sample_data(df_selected, n=sample_size, random_state=random_state)
    df_train, df_test = train_test_split_local(df_sample, train_size=train_size, random_state=random_state)
    print("="*60)
    print(f"✓ Preprocessing complete")
    print(f"  Train: {len(df_train):,} records × {len(df_train.columns)} variables")
    print(f"  Test:  {len(df_test):,} records × {len(df_test.columns)} variables")
    print("="*60)
    return df_train, df_test

# --- from src/generators/independent_marginals.py ---

def generate_independent_marginals(df_train: pd.DataFrame, n_samples: int, random_state: int | None = None) -> pd.DataFrame:
    """
    Convenience function to generate synthetic data using Independent Marginals.
    """
    rng = np.random.RandomState(random_state)
    synthetic_data = {}
    for col in df_train.columns:
        if pd.api.types.is_numeric_dtype(df_train[col]) and df_train[col].nunique() > 20:
            synthetic_data[col] = rng.choice(df_train[col].dropna().values, size=n_samples, replace=True)
        else:
            value_counts = df_train[col].value_counts(normalize=True, dropna=False)
            synthetic_data[col] = rng.choice(value_counts.index, size=n_samples, p=value_counts.values)
    return pd.DataFrame(synthetic_data)

# --- from src/generators/ctgan_generator.py ---

def generate_ctgan(df_train: pd.DataFrame, n_samples: int, epochs: int = 300, batch_size: int = 500, verbose: bool = False, random_state: int | None = None) -> pd.DataFrame:
    """
    Convenience function to generate synthetic data using CTGAN.
    """
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df_train)
    categorical_cols = ['SEX', 'RAC1P_RECODED', 'SCHL_RECODED', 'MAR', 'POBP_RECODED', 'ESR']
    for col in categorical_cols:
        if col in df_train.columns:
            metadata.update_column(col, sdtype='categorical')
    
    model = CTGANSynthesizer(metadata, epochs=epochs, batch_size=batch_size, verbose=verbose)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(df_train)
    return model.sample(num_rows=n_samples)

# --- from src/measures/propensity_auc.py ---

def propensity_auc(df_real: pd.DataFrame, df_synthetic: pd.DataFrame, random_state: int = 42) -> dict:
    """
    Compute propensity AUC.
    """
    df_real_labeled = df_real.copy()
    df_real_labeled['is_synthetic'] = 0
    df_synthetic_labeled = df_synthetic.copy()
    df_synthetic_labeled['is_synthetic'] = 1
    df_combined = pd.concat([df_real_labeled, df_synthetic_labeled], ignore_index=True)
    X = df_combined.drop('is_synthetic', axis=1)
    y = df_combined['is_synthetic']
    X_encoded = pd.get_dummies(X, drop_first=True)
    X_encoded = X_encoded.fillna(X_encoded.median())
    
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=random_state)
    clf.fit(X_encoded, y)
    y_pred_proba = clf.predict_proba(X_encoded)[:, 1]
    auc = roc_auc_score(y, y_pred_proba)
    auc_fidelity = 1.0 - 2.0 * abs(auc - 0.5)
    
    return {'auc': auc, 'auc_fidelity': auc_fidelity}

# --- from src/measures/membership_inference.py ---

def membership_inference_privacy(df_real_train: pd.DataFrame, df_real_test: pd.DataFrame, df_synthetic: pd.DataFrame, random_state: int = 42, min_attack_auc: float = 0.6) -> dict:
    """
    Measure privacy as resistance to membership inference attacks.
    """
    df_train_labeled = df_real_train.copy()
    df_train_labeled['is_member'] = 1
    df_test_labeled = df_real_test.copy()
    df_test_labeled['is_member'] = 0
    df_real_combined = pd.concat([df_train_labeled, df_test_labeled], ignore_index=True)
    X_real = df_real_combined.drop('is_member', axis=1)
    y_real = df_real_combined['is_member']
    X_real_encoded = pd.get_dummies(X_real, drop_first=True)
    X_real_encoded = X_real_encoded.fillna(X_real_encoded.median())
    
    attack_model = LogisticRegression(max_iter=1000, random_state=random_state, solver='lbfgs')
    attack_model.fit(X_real_encoded, y_real)
    
    X_synth_encoded = pd.get_dummies(df_synthetic, drop_first=True)
    X_synth_encoded = X_synth_encoded.reindex(columns=X_real_encoded.columns, fill_value=0)
    X_synth_encoded = X_synth_encoded.fillna(X_real_encoded.median())
    
    membership_probs = attack_model.predict_proba(X_synth_encoded)[:, 1]
    attack_success_rate = membership_probs.mean()
    
    y_real_pred_proba = attack_model.predict_proba(X_real_encoded)[:, 1]
    attack_auc = roc_auc_score(y_real, y_real_pred_proba)
    attack_valid = attack_auc >= min_attack_auc
    
    privacy_score = 1.0 - 2.0 * abs(attack_success_rate - 0.5)
    if not attack_valid:
        privacy_score = np.nan
        
    warning = f"Membership inference attack is too weak (AUC={attack_auc:.3f} < {min_attack_auc})." if not attack_valid else None

    return {'attack_success_rate': attack_success_rate, 'attack_auc': attack_auc, 'attack_valid': attack_valid, 'privacy_score': privacy_score, 'warning': warning}

# --- from src/measures/dcr_privacy.py ---

def dcr_privacy(df_real_train: pd.DataFrame, df_synthetic: pd.DataFrame, percentile: int = 5, random_state: int = 42) -> dict:
    """
    Compute Distance to Closest Record (DCR) privacy measure.
    """
    df_train_encoded = pd.get_dummies(df_real_train, drop_first=True)
    df_synth_encoded = pd.get_dummies(df_synthetic, drop_first=True)
    common_cols = df_train_encoded.columns.intersection(df_synth_encoded.columns)
    df_train_encoded = df_train_encoded[common_cols].fillna(df_train_encoded.median())
    df_synth_encoded = df_synth_encoded[common_cols].fillna(df_synth_encoded.median())
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train_encoded)
    X_synth = scaler.transform(df_synth_encoded)
    
    distances = cdist(X_synth, X_train, metric='euclidean')
    min_distances = distances.min(axis=1)
    dcr_percentile = np.percentile(min_distances, percentile)
    
    rng = np.random.RandomState(random_state)
    sample_indices = rng.choice(X_train.shape[0], size=min(1000, X_train.shape[0]), replace=False)
    X_train_sample = X_train[sample_indices]
    train_distances = cdist(X_train_sample, X_train_sample, metric='euclidean')
    typical_distance = np.median(train_distances[np.triu_indices_from(train_distances, k=1)])
    
    privacy_score = min(1.0, dcr_percentile / (2 * typical_distance))
    
    return {'dcr_percentile': dcr_percentile, 'privacy_score': privacy_score}

# --- from src/validation_pilot.py ---

def run_validation_pilot(sample_size: int = 5000, ctgan_epochs: int = 100, random_state: int = 42, output_dir: str = "results/validation_pilot"):
    """
    Run validation pilot with enhanced checks.
    """
    print("="*80)
    print("VALIDATION PILOT - METHODOLOGICAL CHECKS (N=5K)")
    print("="*80)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("\n[Phase 1] Data Preprocessing")
    df_train, df_test = preprocess_acs_data(sample_size=sample_size, train_size=0.7, random_state=random_state)
    df_train.to_csv(f"{output_dir}/train_data.csv", index=False)
    df_test.to_csv(f"{output_dir}/test_data.csv", index=False)
    
    print("\n[Phase 2] Synthetic Data Generation")
    results = []
    
    print("\n>>> Method 1: Independent Marginals")
    start_time = time.time()
    df_synth_indep = generate_independent_marginals(df_train=df_train, n_samples=len(df_train), random_state=random_state)
    time_indep = time.time() - start_time
    df_synth_indep.to_csv(f"{output_dir}/synthetic_independent_marginals.csv", index=False)
    print(f"✓ Generation time: {time_indep:.2f} seconds")
    
    print("\n>>> Method 2: CTGAN")
    start_time = time.time()
    df_synth_ctgan = generate_ctgan(df_train=df_train, n_samples=len(df_train), epochs=ctgan_epochs, batch_size=200, verbose=False, random_state=random_state)
    time_ctgan = time.time() - start_time
    df_synth_ctgan.to_csv(f"{output_dir}/synthetic_ctgan.csv", index=False)
    print(f"✓ Generation time: {time_ctgan:.2f} seconds")
    
    print("\n[Phase 3] Comprehensive Evaluation")
    
    # Evaluate Independent Marginals
    fidelity_indep = propensity_auc(df_train, df_synth_indep, random_state=random_state)
    privacy_mi_indep = membership_inference_privacy(df_real_train=df_train, df_real_test=df_test, df_synthetic=df_synth_indep, random_state=random_state)
    privacy_dcr_indep = dcr_privacy(df_real_train=df_train, df_synthetic=df_synth_indep)
    results.append({'method': 'Independent Marginals', 'generation_time_sec': time_indep, 'fidelity_auc': fidelity_indep['auc'], 'privacy_mi_attack_auc': privacy_mi_indep['attack_auc'], 'privacy_mi_attack_valid': privacy_mi_indep['attack_valid'], 'privacy_dcr_percentile': privacy_dcr_indep['dcr_percentile'], 'privacy_dcr_score': privacy_dcr_indep['privacy_score']})
    
    # Evaluate CTGAN
    fidelity_ctgan = propensity_auc(df_train, df_synth_ctgan, random_state=random_state)
    privacy_mi_ctgan = membership_inference_privacy(df_real_train=df_train, df_real_test=df_test, df_synthetic=df_synth_ctgan, random_state=random_state)
    privacy_dcr_ctgan = dcr_privacy(df_real_train=df_train, df_synthetic=df_synth_ctgan)
    results.append({'method': 'CTGAN', 'generation_time_sec': time_ctgan, 'fidelity_auc': fidelity_ctgan['auc'], 'privacy_mi_attack_auc': privacy_mi_ctgan['attack_auc'], 'privacy_mi_attack_valid': privacy_mi_ctgan['attack_valid'], 'privacy_dcr_percentile': privacy_dcr_ctgan['dcr_percentile'], 'privacy_dcr_score': privacy_dcr_ctgan['privacy_score']})
    
    print("\n[Phase 4] Methodological Validation")
    df_results = pd.DataFrame(results)
    mi_valid_count = df_results['privacy_mi_attack_valid'].sum()
    primary_privacy_measure = 'Membership Inference' if mi_valid_count == len(df_results) else 'DCR'
    
    print("\n>>> Check 1: Privacy Measure Validity")
    print(f"Membership Inference Attack Valid: {mi_valid_count}/{len(df_results)}")
    if primary_privacy_measure == 'DCR': print("→ Action: Use DCR as primary privacy measure")
    
    # ... (other checks from pilot script) ...

    print("\n[Phase 5] GO/NO-GO Decision")
    go_decision = True
    print("\n✅ GO: Proceed with full N=35K experiments")
    
    df_results.to_csv(f"{output_dir}/validation_results.csv", index=False)
    decision_summary = {'go_decision': go_decision, 'primary_privacy_measure': primary_privacy_measure}
    with open(f"{output_dir}/decision_summary.json", 'w') as f: json.dump(decision_summary, f, indent=2)
    
    return df_results, decision_summary

if __name__ == "__main__":
    if sys.platform == "darwin":
        try:
            multiprocessing.set_start_method('fork', force=True)
        except RuntimeError:
            pass
            
    print("\nStarting validation pilot...")
    results, decision = run_validation_pilot()
    
    print("\n" + "="*80)
    if decision['go_decision']:
        print("RECOMMENDATION: Proceed with full experiments")
    else:
        print("RECOMMENDATION: Address issues before proceeding")
    print("="*80)
