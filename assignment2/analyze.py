import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--file", default="trajectory.csv")
args = parser.parse_args()

df = pd.read_csv(args.file)

duration = df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]
hz = len(df) / duration if duration > 0 else 0

print(f"File:    {args.file}")
print(f"Points:  {len(df)}")
print(f"Duration:{duration:.2f} s  (~{hz:.0f} Hz)")
print()
print(f"{'axis':<6} {'min':>10} {'max':>10} {'range':>10} {'mean':>10}")
print("-" * 52)
for ax in ["x", "y", "z"]:
    lo, hi, mean = df[ax].min(), df[ax].max(), df[ax].mean()
    print(f"{ax:<6} {lo:>10.4f} {hi:>10.4f} {hi-lo:>10.4f} {mean:>10.4f}  m")
