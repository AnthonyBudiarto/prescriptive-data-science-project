"""
run_all.py
----------
Runs the full pipeline end-to-end:
  Public Data -> Descriptive -> Predictive -> Prescriptive -> Optimal Decision
"""
import os
import runpy

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = [
    "01_fetch_data.py",
    "02_descriptive.py",
    "03_predictive.py",
    "04_prescriptive.py",
    "05_optimal_decision.py",
]

if __name__ == "__main__":
    for s in STAGES:
        print("\n" + "=" * 70)
        print(f"RUNNING: {s}")
        print("=" * 70)
        runpy.run_path(os.path.join(HERE, s), run_name="__main__")
    print("\nPipeline complete. See output/ for all tables and figures.")
