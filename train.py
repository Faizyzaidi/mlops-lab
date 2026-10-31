"""Lab 1 - Task 2: reproducible decision tree.
Runs the training script 3 separate times (fresh Python processes)
and asserts that the accuracy is exactly identical every time."""
import subprocess
import sys

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

SEED = 42


def train():
    X, y = load_iris(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    model = DecisionTreeClassifier(max_depth=3, random_state=SEED).fit(Xtr, ytr)
    return accuracy_score(yte, model.predict(Xte))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        print(f"{train():.10f}")
    else:
        accs = []
        for i in range(3):
            out = subprocess.run([sys.executable, __file__, "child"],
                                 capture_output=True, text=True, check=True).stdout.strip()
            print(f"Execution {i + 1}: accuracy = {out}")
            accs.append(out)
        assert len(set(accs)) == 1, f"Results differ across runs: {accs}"
        print("PASS: all three executions produced the identical accuracy", accs[0])
