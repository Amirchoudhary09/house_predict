import argparse
import pickle
import sys
from pathlib import Path

try:
    import joblib
except Exception:
    joblib = None


def summarize(obj, name="root", indent=0):
    pad = "  " * indent
    t = type(obj)
    try:
        typename = f"{t.__module__}.{t.__name__}"
    except Exception:
        typename = str(t)
    out = [f"{pad}{name}: {typename}"]

    # common quick introspection
    if hasattr(obj, "shape"):
        try:
            out.append(f"{pad}  shape: {obj.shape}")
        except Exception:
            pass
    if hasattr(obj, "get_params"):
        try:
            params = obj.get_params()
            out.append(f"{pad}  get_params: {list(params.keys())[:10]}{('...' if len(params)>10 else '')}")
        except Exception:
            pass
    if isinstance(obj, dict):
        out.append(f"{pad}  dict keys: {list(obj.keys())}")
        for k, v in list(obj.items())[:20]:
            out.extend(summarize(v, name=f"key[{repr(k)}]", indent=indent + 2))
    elif isinstance(obj, (list, tuple)):
        out.append(f"{pad}  length: {len(obj)}")
        for i, v in enumerate(obj[:20]):
            out.extend(summarize(v, name=f"[{i}]", indent=indent + 2))
    else:
        # try attributes for common ML objects
        if hasattr(obj, "coef_"):
            try:
                out.append(f"{pad}  has coef_ (len): {len(obj.coef_)}")
            except Exception:
                pass
        if hasattr(obj, "classes_"):
            try:
                out.append(f"{pad}  has classes_ (len): {len(obj.classes_)}")
            except Exception:
                pass

    return out


def load_pickle(path: Path):
    # try joblib first (works for sklearn dump too)
    if joblib is not None:
        try:
            return joblib.load(path)
        except Exception:
            pass
    # fallback to pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    p = argparse.ArgumentParser(description="Inspect a .pkl / joblib file and print a quick summary")
    p.add_argument("file", help="Path to .pkl or joblib file")
    args = p.parse_args()
    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(2)

    try:
        obj = load_pickle(path)
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        sys.exit(3)

    lines = summarize(obj)
    print(f"Summary for: {path}\n")
    for L in lines:
        print(L)


if __name__ == "__main__":
    main()
