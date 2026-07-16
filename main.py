import subprocess
import sys

def run_all():
    print("=" * 60)
    print("PROJECT 2: LEAST SQUARES VS REGRESSION")
    print("=" * 60)
    
    print("\n[1/5] Running Preprocessing...")
    subprocess.run([sys.executable, 'src/preprocess.py'])
    
    print("\n[2/5] Running Least Squares...")
    subprocess.run([sys.executable, 'src/least_squares.py'])
    
    print("\n[3/5] Running Ridge Regression...")
    subprocess.run([sys.executable, 'src/ridge_regression.py'])
    
    print("\n[4/5] Running Lasso Regression...")
    subprocess.run([sys.executable, 'src/lasso_regression.py'])
    
    print("\n[5/5] Running Stability Analysis...")
    subprocess.run([sys.executable, 'src/stability_analysis.py'])
    
    print("\n[6/6] Creating Visualizations...")
    subprocess.run([sys.executable, 'src/visualize.py'])
    
    print("\n" + "=" * 60)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("Results saved in 'results/' folder")
    print("Figures saved in 'results/figures/' folder")
    print("=" * 60)

if __name__ == "__main__":
    run_all()