#!/usr/bin/env python3
"""
Example script to test parameter tracking with different methods.
"""

import subprocess
import sys

def run_experiment(method, dataset="seq-cifar10", n_epochs=1, timeout=600):
    """Run a simple experiment to test parameter tracking."""
    cmd = [
        sys.executable, "main.py",
        "--model", method,
        "--dataset", dataset,
        "--n_epochs", str(n_epochs),
        "--batch_size", "32",
        "--lr", "0.001",
        "--non_verbose", "1"
    ]
    
    print(f"🧪 Testing {method} with parameter tracking...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        
        if result.returncode == 0:
            print(f"✅ {method} test successful")
            # Look for parameter analysis in output
            if "Parameter Analysis" in result.stdout or "Parameter Analysis" in result.stderr:
                print("✅ Parameter tracking is working!")
            else:
                print("⚠️  Parameter tracking output not found in logs")
        else:
            print(f"❌ {method} test failed:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {method} test timed out after {timeout} seconds")
    except Exception as e:
        print(f"❌ Error running {method}: {e}")

if __name__ == "__main__":
    print("🔬 Testing Parameter Tracking Integration")
    print("=" * 50)
    
    # Test configurations
    tests = {
        "l2p": {"dataset": "seq-cifar10-224", "n_epochs": 1},
        "dualprompt": {"dataset": "seq-cifar10-224", "n_epochs": 1},
        "coda_prompt": {"dataset": "seq-cifar10", "n_epochs": 2}
    }
    
    for method, config in tests.items():
        run_experiment(method, **config)
        print()
    
    print("🎉 Testing complete!")
    print("Check the output above for parameter analysis information.")