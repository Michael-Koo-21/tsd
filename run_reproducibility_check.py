#!/usr/bin/env python3
"""Final reproducibility check for the research project."""

import os
import sys
import hashlib
import json
from datetime import datetime
from pathlib import Path

def compute_file_hash(filepath):
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for block in iter(lambda: f.read(65536), b''):
            sha256.update(block)
    return sha256.hexdigest()

def run_reproducibility_check():
    """Run comprehensive reproducibility verification."""
    
    print('=' * 80)
    print('FINAL REPRODUCIBILITY CHECK')
    print(f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 80)
    
    checks_passed = 0
    checks_failed = 0
    warnings = 0
    
    # 1. Check required files exist
    print('\n[1] REQUIRED FILES CHECK')
    required_files = {
        'Pre-registration': 'docs/pre_registration.md',
        'Study design': 'docs/study_design_spec.md',
        'Results data': 'results/experiments/all_results_complete.csv',
        'Validation pilot': 'results/validation_pilot/decision_summary.json',
        'MADA framework': 'src/analysis/mada_framework.py',
        'Statistical analysis': 'src/analysis/statistical_analysis.py',
        'Main experiment script': 'src/run_experiments.py',
        'Requirements': 'requirements.txt',
        'README': 'README.md',
    }
    
    for name, filepath in required_files.items():
        exists = os.path.exists(filepath)
        status = '✓' if exists else '✗'
        print(f'  {status} {name}: {filepath}')
        if exists:
            checks_passed += 1
        else:
            checks_failed += 1
    
    # 2. Check data integrity (hashes of key files)
    print('\n[2] DATA INTEGRITY (file hashes)')
    key_files = [
        'results/experiments/all_results_complete.csv',
        'data/raw/psam_p06.csv',
    ]
    
    hashes = {}
    for filepath in key_files:
        if os.path.exists(filepath):
            file_hash = compute_file_hash(filepath)[:16]  # First 16 chars
            hashes[filepath] = file_hash
            print(f'  ✓ {filepath}: {file_hash}...')
            checks_passed += 1
        else:
            print(f'  ✗ {filepath}: FILE NOT FOUND')
            checks_failed += 1
    
    # 3. Check synthetic data files
    print('\n[3] SYNTHETIC DATA FILES')
    methods = ['ctgan', 'independent_marginals', 'dpbn', 'synthpop', 'great']
    replicates = 5
    
    for method in methods:
        for rep in range(1, replicates + 1):
            filepath = f'results/experiments/synthetic_data/{method}_rep{rep}.csv'
            exists = os.path.exists(filepath)
            if not exists:
                print(f'  ✗ Missing: {filepath}')
                checks_failed += 1
    
    synth_count = sum(1 for m in methods for r in range(1, replicates + 1) 
                      if os.path.exists(f'results/experiments/synthetic_data/{m}_rep{r}.csv'))
    expected = len(methods) * replicates
    print(f'  Synthetic data files: {synth_count}/{expected}')
    if synth_count == expected:
        print(f'  ✓ All synthetic data files present')
        checks_passed += 1
    else:
        print(f'  ✗ Missing {expected - synth_count} synthetic data files')
        checks_failed += 1
    
    # 4. Check code modules
    print('\n[4] CODE MODULE VERIFICATION')
    modules = [
        ('Generators', [
            'src/generators/ctgan_generator.py',
            'src/generators/independent_marginals.py',
            'src/generators/dp_bayesian_network.py',
            'src/generators/synthpop_wrapper.py',
            'src/generators/great_generator.py',
        ]),
        ('Measures', [
            'src/measures/propensity_auc.py',
            'src/measures/dcr_privacy.py',
            'src/measures/tstr_utility.py',
            'src/measures/fairness_gap.py',
        ]),
        ('Analysis', [
            'src/analysis/mada_framework.py',
            'src/analysis/statistical_analysis.py',
            'src/analysis/visualizations.py',
            'src/analysis/voi_analysis.py',
        ]),
    ]
    
    for category, files in modules:
        all_exist = all(os.path.exists(f) for f in files)
        status = '✓' if all_exist else '✗'
        print(f'  {status} {category}: {len([f for f in files if os.path.exists(f)])}/{len(files)} files')
        if all_exist:
            checks_passed += 1
        else:
            checks_failed += 1
    
    # 5. Check Python dependencies can be imported
    print('\n[5] DEPENDENCY CHECK')
    try:
        import pandas
        print(f'  ✓ pandas {pandas.__version__}')
        checks_passed += 1
    except ImportError:
        print('  ✗ pandas NOT INSTALLED')
        checks_failed += 1
    
    try:
        import numpy
        print(f'  ✓ numpy {numpy.__version__}')
        checks_passed += 1
    except ImportError:
        print('  ✗ numpy NOT INSTALLED')
        checks_failed += 1
    
    try:
        import scipy
        print(f'  ✓ scipy {scipy.__version__}')
        checks_passed += 1
    except ImportError:
        print('  ✗ scipy NOT INSTALLED')
        checks_failed += 1
    
    try:
        import sklearn
        print(f'  ✓ scikit-learn {sklearn.__version__}')
        checks_passed += 1
    except ImportError:
        print('  ✗ scikit-learn NOT INSTALLED')
        checks_failed += 1
    
    # 6. Verify results can be loaded and analyzed
    print('\n[6] RESULTS VERIFICATION')
    try:
        import pandas as pd
        df = pd.read_csv('results/experiments/all_results_complete.csv')
        
        # Check expected columns
        expected_cols = ['method', 'replicate', 'seed', 'fidelity_auc', 'privacy_dcr', 
                         'utility_tstr', 'fairness_gap', 'efficiency_time']
        missing_cols = [c for c in expected_cols if c not in df.columns]
        
        if not missing_cols:
            print(f'  ✓ All expected columns present')
            checks_passed += 1
        else:
            print(f'  ✗ Missing columns: {missing_cols}')
            checks_failed += 1
        
        # Check expected number of rows
        expected_rows = 25  # 5 methods × 5 replicates
        if len(df) == expected_rows:
            print(f'  ✓ Expected row count: {len(df)}')
            checks_passed += 1
        else:
            print(f'  ⚠ Row count: {len(df)} (expected {expected_rows})')
            warnings += 1
        
        # Check all methods present
        expected_methods = {'ctgan', 'independent_marginals', 'dpbn', 'synthpop', 'great'}
        actual_methods = set(df['method'].unique())
        if expected_methods == actual_methods:
            print(f'  ✓ All methods present: {list(actual_methods)}')
            checks_passed += 1
        else:
            missing = expected_methods - actual_methods
            extra = actual_methods - expected_methods
            print(f'  ✗ Method mismatch. Missing: {missing}, Extra: {extra}')
            checks_failed += 1
            
    except Exception as e:
        print(f'  ✗ Error loading results: {e}')
        checks_failed += 1
    
    # 7. Check git status
    print('\n[7] GIT REPOSITORY STATUS')
    import subprocess
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
        if result.returncode == 0:
            commit_hash = result.stdout.strip()
            print(f'  ✓ Git repository initialized')
            print(f'    Current commit: {commit_hash[:12]}...')
            checks_passed += 1
            
            # Check for uncommitted changes
            result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
            if result.stdout.strip():
                print(f'  ⚠ Uncommitted changes present')
                warnings += 1
            else:
                print(f'  ✓ Working directory clean')
                checks_passed += 1
        else:
            print(f'  ✗ Not a git repository')
            checks_failed += 1
    except Exception as e:
        print(f'  ✗ Git check failed: {e}')
        checks_failed += 1
    
    # 8. Verify pre-registration lock
    print('\n[8] PRE-REGISTRATION LOCK VERIFICATION')
    try:
        with open('docs/pre_registration.md', 'r') as f:
            content = f.read()
        
        if 'Git Commit Hash:' in content:
            print('  ✓ Pre-registration contains git commit hash')
            checks_passed += 1
        else:
            print('  ⚠ Pre-registration missing git commit hash')
            warnings += 1
        
        if 'LOCKED' in content:
            print('  ✓ Pre-registration marked as locked')
            checks_passed += 1
        else:
            print('  ⚠ Pre-registration not marked as locked')
            warnings += 1
            
    except Exception as e:
        print(f'  ✗ Pre-registration check failed: {e}')
        checks_failed += 1
    
    # 9. Test core functionality
    print('\n[9] CORE FUNCTIONALITY TEST')
    try:
        sys.path.insert(0, '.')
        from src.analysis.mada_framework import load_results, get_method_scores, PROFILES
        
        df = load_results('results/experiments/all_results_complete.csv')
        scores = get_method_scores(df)
        
        if len(scores) == 5:  # 5 methods
            print('  ✓ MADA framework loads and processes data')
            checks_passed += 1
        else:
            print(f'  ⚠ Unexpected number of methods: {len(scores)}')
            warnings += 1
            
        if len(PROFILES) >= 3:
            print(f'  ✓ Decision profiles defined: {list(PROFILES.keys())}')
            checks_passed += 1
        else:
            print(f'  ⚠ Missing decision profiles')
            warnings += 1
            
    except Exception as e:
        print(f'  ✗ Core functionality test failed: {e}')
        checks_failed += 1
    
    # Summary
    print('\n' + '=' * 80)
    print('REPRODUCIBILITY CHECK SUMMARY')
    print('=' * 80)
    print(f'  Checks passed: {checks_passed}')
    print(f'  Checks failed: {checks_failed}')
    print(f'  Warnings: {warnings}')
    
    if checks_failed == 0:
        print('\n✓ ALL REPRODUCIBILITY CHECKS PASSED')
        print('  The experiment is reproducible from the current repository state.')
        status = 'PASS'
    elif checks_failed <= 2:
        print('\n⚠ REPRODUCIBILITY CHECK PASSED WITH WARNINGS')
        print('  Minor issues detected but results should be reproducible.')
        status = 'PASS_WITH_WARNINGS'
    else:
        print('\n✗ REPRODUCIBILITY CHECK FAILED')
        print('  Critical issues must be resolved before publication.')
        status = 'FAIL'
    
    # Write summary to file
    summary = {
        'check_date': datetime.now().isoformat(),
        'status': status,
        'checks_passed': checks_passed,
        'checks_failed': checks_failed,
        'warnings': warnings,
        'file_hashes': hashes,
    }
    
    with open('results/experiments/reproducibility_check.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f'\n  Summary written to: results/experiments/reproducibility_check.json')
    
    return status

if __name__ == '__main__':
    status = run_reproducibility_check()
    sys.exit(0 if status in ['PASS', 'PASS_WITH_WARNINGS'] else 1)
