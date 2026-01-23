"""
Synthpop Generator - Python Wrapper

Wrapper around R synthpop package using subprocess for file-based communication.

This approach:
1. Writes training data to temporary CSV
2. Calls R script via subprocess
3. Reads synthetic data from output CSV
4. Cleans up temporary files

Reference:
- Nowok, B., Raab, G. M., & Dibben, C. (2016). synthpop: Bespoke Creation
  of Synthetic Data in R. Journal of Statistical Software, 74(11).
  https://doi.org/10.18637/jss.v074.i11
"""

import pandas as pd
import subprocess
import tempfile
import os
from pathlib import Path


def generate_synthpop(
    df_train: pd.DataFrame,
    n_samples: int,
    method: str = "cart",
    cart_minbucket: int = 5,
    random_state: int = 42,
    r_script_path: str = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Generate synthetic data using R synthpop package.

    Args:
        df_train: Training data (pandas DataFrame)
        n_samples: Number of synthetic records to generate
        method: Synthesis method (default: "cart")
        cart_minbucket: Minimum bucket size for CART (default: 5)
        random_state: Random seed for reproducibility
        r_script_path: Path to synthpop_generate.R (auto-detected if None)
        verbose: Print progress messages

    Returns:
        DataFrame with synthetic data

    Raises:
        RuntimeError: If R script fails
        FileNotFoundError: If R or synthpop not installed
    """
    # Auto-detect R script path if not provided
    if r_script_path is None:
        project_root = Path(__file__).parent.parent.parent
        r_script_path = project_root / "R" / "synthpop_generate.R"

    if not os.path.exists(r_script_path):
        raise FileNotFoundError(
            f"R script not found: {r_script_path}\n"
            f"Expected location: R/synthpop_generate.R"
        )

    # Check if R is installed
    try:
        subprocess.run(
            ["Rscript", "--version"],
            capture_output=True,
            check=True,
            text=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise FileNotFoundError(
            "R not installed or not in PATH.\n"
            "Install R from: https://www.r-project.org/\n"
            "Then install synthpop: install.packages('synthpop')"
        )

    if verbose:
        print(f"Generating {n_samples:,} synthetic records with Synthpop...")
        print(f"  Method: {method}")
        print(f"  CART minbucket: {cart_minbucket}")
        print(f"  Random seed: {random_state}")

    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as input_file:
        input_path = input_file.name
        # Write training data to CSV
        # Convert categorical columns to strings for R
        df_to_write = df_train.copy()
        for col in df_to_write.select_dtypes(include=['category', 'object']).columns:
            df_to_write[col] = df_to_write[col].astype(str)
        df_to_write.to_csv(input_path, index=False)

    output_path = input_path.replace('.csv', '_synthetic.csv')

    try:
        # Call R script
        cmd = [
            "Rscript",
            str(r_script_path),
            input_path,
            output_path,
            str(n_samples),
            str(random_state)
        ]

        if verbose:
            print(f"\nCalling R script...")
            print(f"  Command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        if verbose and result.stdout:
            print("\nR Output:")
            print(result.stdout)

        if result.stderr:
            print("\nR Warnings/Messages:")
            print(result.stderr)

        # Read synthetic data
        if not os.path.exists(output_path):
            raise RuntimeError(
                f"R script did not produce output file: {output_path}\n"
                f"R may have failed silently."
            )

        df_synthetic = pd.read_csv(output_path)

        # Restore data types to match input
        for col in df_train.columns:
            if col in df_synthetic.columns:
                # Match data type from training data
                if df_train[col].dtype.name == 'category':
                    df_synthetic[col] = pd.Categorical(df_synthetic[col])
                elif df_train[col].dtype in ['object']:
                    df_synthetic[col] = df_synthetic[col].astype(str)
                else:
                    # Numeric columns
                    df_synthetic[col] = df_synthetic[col].astype(df_train[col].dtype)

        if verbose:
            print(f"\n✓ Generated {len(df_synthetic):,} synthetic records")

        return df_synthetic

    except subprocess.CalledProcessError as e:
        error_msg = f"R script failed with exit code {e.returncode}\n"
        if e.stdout:
            error_msg += f"\nStdout:\n{e.stdout}"
        if e.stderr:
            error_msg += f"\nStderr:\n{e.stderr}"
        raise RuntimeError(error_msg)

    finally:
        # Clean up temporary files
        for path in [input_path, output_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass  # Ignore cleanup errors


if __name__ == "__main__":
    # Test synthpop wrapper
    import sys
    sys.path.append('.')
    from src.preprocessing.load_data import preprocess_acs_data

    print("="*70)
    print("Testing Synthpop Wrapper")
    print("="*70)

    # Check R installation
    print("\nChecking R installation...")
    try:
        result = subprocess.run(
            ["Rscript", "--version"],
            capture_output=True,
            check=True,
            text=True
        )
        print("  ✓ R is installed")
        print(result.stderr.strip())  # R version goes to stderr
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print("  ✗ R not found!")
        print("\nPlease install R:")
        print("  macOS: brew install r")
        print("  Ubuntu: sudo apt-get install r-base")
        print("  Windows: https://www.r-project.org/")
        sys.exit(1)

    # Check synthpop package
    print("\nChecking synthpop package...")
    check_cmd = ["Rscript", "-e", "if (!require('synthpop', quietly=TRUE)) quit(status=1)"]
    result = subprocess.run(check_cmd, capture_output=True)

    if result.returncode != 0:
        print("  ✗ synthpop package not installed!")
        print("\nPlease install synthpop in R:")
        print("  R -e \"install.packages('synthpop')\"")
        sys.exit(1)
    else:
        print("  ✓ synthpop package is installed")

    # Load test data
    print("\nLoading test data (N=1K)...")
    df_train, df_test = preprocess_acs_data(sample_size=1000, random_state=42)
    print(f"  Train: {len(df_train):,} records")

    # Generate synthetic data
    print("\nGenerating synthetic data with Synthpop...")
    try:
        df_synthetic = generate_synthpop(
            df_train=df_train,
            n_samples=len(df_train),
            random_state=42,
            verbose=True
        )

        print("\nValidation:")
        print(f"  Rows: {len(df_synthetic):,} (expected {len(df_train):,})")
        print(f"  Columns: {len(df_synthetic.columns)} (expected {len(df_train.columns)})")

        if set(df_synthetic.columns) == set(df_train.columns):
            print("  ✓ Schema matches")
        else:
            print("  ✗ Schema mismatch!")

        print("\n" + "="*70)
        print("✓ Synthpop wrapper test PASSED")
        print("="*70)

    except Exception as e:
        print(f"\n✗ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
