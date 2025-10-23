# io_generation_info.py
import os, json, time
from typing import Optional, Dict, Any
import torch

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_history(history: torch.Tensor, out_dir: str) -> str:
    path = os.path.join(out_dir, "history.pt")
    torch.save({"history": history}, path)
    return path

def load_history(out_dir: str, map_location: str = "cpu") -> torch.Tensor:
    path = os.path.join(out_dir, "history.pt")
    obj = torch.load(path, map_location=map_location, weights_only=True)
    return obj["history"]

def save_logits(logits: torch.Tensor, out_dir: str) -> str:
    path = os.path.join(out_dir, "logits.pt")
    torch.save({"logits": logits}, path)
    return path

def load_logits(out_dir: str, map_location: str = "cpu") -> torch.Tensor:
    path = os.path.join(out_dir, "logits.pt")
    obj = torch.load(path, map_location=map_location, weights_only=True)
    return obj["logits"]

def save_confidence_t_0(confidence_t_0: torch.Tensor, out_dir: str) -> str:
    path = os.path.join(out_dir, "confidence_t_0.pt")
    torch.save({"confidence_t_0": confidence_t_0}, path)
    return path

def load_confidence_t_0(out_dir: str, map_location: str = "cpu") -> torch.Tensor:
    path = os.path.join(out_dir, "confidence_t_0.pt")
    obj = torch.load(path, map_location=map_location, weights_only=True)
    return obj["confidence_t_0"]

def save_marginals(marginals: Dict[str, torch.Tensor], out_dir: str) -> str:
    path = os.path.join(out_dir, "marginals.pt")
    # store tensors on CPU for portability
    cpu_marginals = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in marginals.items()}
    torch.save(cpu_marginals, path)
    return path

def load_marginals(out_dir: str, map_location: str = "cpu") -> Dict[str, torch.Tensor]:
    path = os.path.join(out_dir, "marginals.pt")
    obj = torch.load(path, map_location=map_location, weights_only=True)
    return obj

def save_change_events_table(change_events: list[list[dict]], out_dir: str) -> str:
    """
    Flattens to a table with columns:
    sample, step, position, old_value, new_value, event_type, model_confidence
    """
    try:
        import pandas as pd
        rows = []
        for sample_idx, events in enumerate(change_events):
            for ev in events:
                rows.append({
                    "sample": sample_idx,
                    "step": ev["step"],
                    "position": ev["position"],
                    "old_value": ev["old_value"],
                    "new_value": ev["new_value"],
                    "event_type": ev["event_type"],
                    "old_value_confidence": ev.get("old_value_confidence", None),
                    "model_confidence": ev["model_confidence"],
                    "model_confidence_after": ev.get("model_confidence_after", None),
                })
        df = pd.DataFrame(rows)
        path_parquet = os.path.join(out_dir, "change_events.parquet")
        try:
            df.to_parquet(path_parquet, index=False)  # requires pyarrow or fastparquet
            return path_parquet
        except Exception:
            # Fallback to CSV if parquet back-end not available
            path_csv = os.path.join(out_dir, "change_events.csv")
            df.to_csv(path_csv, index=False)
            return path_csv
    except ImportError:
        # No pandas at all — store as JSONL (one event per line)
        path_jsonl = os.path.join(out_dir, "change_events.jsonl")
        with open(path_jsonl, "w") as f:
            for sample_idx, events in enumerate(change_events):
                for ev in events:
                    rec = {"sample": sample_idx, **ev}
                    f.write(json.dumps(rec) + "\n")
        return path_jsonl

def load_change_events_table(out_dir: str):
    """
    Returns a pandas DataFrame if available (parquet/csv), else a list of JSON dicts.
    """
    import os
    parquet = os.path.join(out_dir, "change_events.parquet")
    csv = os.path.join(out_dir, "change_events.csv")
    jsonl = os.path.join(out_dir, "change_events.jsonl")

    if os.path.exists(parquet):
        import pandas as pd
        return pd.read_parquet(parquet)
    if os.path.exists(csv):
        import pandas as pd
        return pd.read_csv(csv)
    if os.path.exists(jsonl):
        with open(jsonl, "r") as f:
            return [json.loads(line) for line in f]
    raise FileNotFoundError("No change events file found in out_dir.")

def save_forward_calls(forward_calls: Dict[str, Any], out_dir: str) -> str:
    path = os.path.join(out_dir, "forward_calls.pt")
    torch.save(forward_calls, path)
    return path

def load_forward_calls(out_dir: str, map_location: str = "cpu") -> Dict[str, Any]:
    path = os.path.join(out_dir, "forward_calls.pt")
    obj = torch.load(path, map_location=map_location, weights_only=True)
    return obj

def save_prune_correct(prune_correct: list[Dict[str, Any]], out_dir: str) -> str:
    path = os.path.join(out_dir, "prune_correct.pt")
    torch.save(prune_correct, path)
    return path

def load_prune_correct(out_dir: str, map_location: str = "cpu") -> list[Dict[str, Any]]:
    path = os.path.join(out_dir, "prune_correct.pt")
    obj = torch.load(path, map_location=map_location, weights_only=True)
    return obj

def save_meta(meta: Dict[str, Any], out_dir: str) -> str:
    path = os.path.join(out_dir, "meta.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    return path

def load_meta(out_dir: str) -> Dict[str, Any]:
    path = os.path.join(out_dir, "meta.json")
    with open(path, "r") as f:
        return json.load(f)
