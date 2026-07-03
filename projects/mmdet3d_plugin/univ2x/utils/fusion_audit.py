import atexit
import json
import os
import threading
import time

import numpy as np
import torch


_LOCK = threading.Lock()
_EVENT_COUNT = 0


def enabled():
    return os.environ.get('UNIV2X_FUSION_AUDIT', '').lower() in (
        '1', 'true', 'yes', 'on')


def _rank_zero():
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
    except Exception:
        pass
    return int(os.environ.get('RANK', '0')) == 0


def _audit_path():
    return os.environ.get(
        'UNIV2X_FUSION_AUDIT_PATH',
        os.path.join(os.getcwd(), 'univ2x_fusion_audit.jsonl'))


def _wandb_enabled():
    return os.environ.get('UNIV2X_FUSION_AUDIT_WANDB', '').lower() in (
        '1', 'true', 'yes', 'on')


def _to_scalar(value):
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        if value.numel() == 1:
            return value.detach().cpu().item()
        return None
    return None


def _sanitize_metrics(metrics):
    out = {}
    for key, value in (metrics or {}).items():
        scalar = _to_scalar(value)
        if scalar is not None:
            out[str(key)] = scalar
    return out


def _iter_shift_records(physical_shift):
    if not physical_shift:
        return []
    if isinstance(physical_shift, (list, tuple)):
        records = []
        for item in physical_shift:
            records.extend(_iter_shift_records(item))
        return records
    if not isinstance(physical_shift, dict):
        return []
    if physical_shift.get('shifts'):
        records = []
        for item in physical_shift.get('shifts', []):
            records.extend(_iter_shift_records(item))
        if records:
            return records
    return [physical_shift]


def shift_summary(physical_shift):
    records = _iter_shift_records(physical_shift)
    if not records:
        return {
            'shift_present': 0,
            'shift_count': 0,
            'shift_names': 'none',
        }

    names = []
    severity = 0.0
    timestamp_delay_s = 0.0
    effective_frame_delay = 0.0
    yaw_abs_deg = 0.0
    translation_m = 0.0
    keep_ratio_min = 1.0
    drop_probability = 0.0
    label_preserving = 0

    for shift in records:
        name = str(shift.get('name', 'unknown'))
        if name not in names:
            names.append(name)
        severity = max(severity, float(shift.get('severity', 0.0)))
        timestamp_delay_s = max(
            timestamp_delay_s, abs(float(shift.get('timestamp_delay_s', 0.0))))
        effective_frame_delay = max(
            effective_frame_delay, abs(float(shift.get('effective_frame_delay', 0.0))))
        yaw_abs_deg = max(yaw_abs_deg, abs(float(shift.get('yaw_deg', 0.0))))
        trans = shift.get('translation_m', None)
        if trans is not None:
            try:
                translation_m = max(
                    translation_m,
                    sum(float(v) ** 2 for v in trans[:3]) ** 0.5)
            except Exception:
                pass
        if 'keep_ratio' in shift:
            keep_ratio_min = min(keep_ratio_min, float(shift.get('keep_ratio', 1.0)))
        drop_probability = max(
            drop_probability, float(shift.get('drop_probability', 0.0)))
        label_preserving = int(label_preserving or shift.get('label_preserving', False))

    return {
        'shift_present': 1,
        'shift_count': len(records),
        'shift_names': ','.join(names),
        'shift_severity_max': severity,
        'shift_timestamp_delay_s_max': timestamp_delay_s,
        'shift_effective_frame_delay_max': effective_frame_delay,
        'shift_yaw_abs_deg_max': yaw_abs_deg,
        'shift_translation_m_max': translation_m,
        'shift_keep_ratio_min': keep_ratio_min,
        'shift_drop_probability_max': drop_probability,
        'shift_label_preserving': label_preserving,
    }


def tensor_stats(prefix, value):
    if value is None:
        return {}
    try:
        if torch.is_tensor(value):
            vals = value.detach().float().reshape(-1)
            vals = vals[torch.isfinite(vals)]
            if vals.numel() == 0:
                return {prefix + '_count': 0}
            vals_sorted, _ = torch.sort(vals)
            n = int(vals_sorted.numel())

            def pct(q):
                idx = int(round((n - 1) * q))
                return vals_sorted[idx].item()

            return {
                prefix + '_count': n,
                prefix + '_mean': vals.mean().item(),
                prefix + '_min': vals_sorted[0].item(),
                prefix + '_p50': pct(0.50),
                prefix + '_p90': pct(0.90),
                prefix + '_max': vals_sorted[-1].item(),
            }
        vals = np.asarray(value, dtype=np.float32).reshape(-1)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return {prefix + '_count': 0}
        return {
            prefix + '_count': int(vals.size),
            prefix + '_mean': float(vals.mean()),
            prefix + '_min': float(vals.min()),
            prefix + '_p50': float(np.percentile(vals, 50)),
            prefix + '_p90': float(np.percentile(vals, 90)),
            prefix + '_max': float(vals.max()),
        }
    except Exception:
        return {}


def emit(component, event, metrics=None, physical_shift=None):
    if not enabled() or not _rank_zero():
        return

    global _EVENT_COUNT
    max_events = int(os.environ.get('UNIV2X_FUSION_AUDIT_MAX_EVENTS', '0') or 0)
    with _LOCK:
        if max_events > 0 and _EVENT_COUNT >= max_events:
            return
        _EVENT_COUNT += 1
        record = {
            'time': time.time(),
            'component': component,
            'event': event,
            'event_index': _EVENT_COUNT,
        }
        record.update(shift_summary(physical_shift))
        record.update(_sanitize_metrics(metrics))

        path = _audit_path()
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, 'a') as f:
            f.write(json.dumps(record, sort_keys=True) + '\n')

    if _wandb_enabled():
        _log_wandb(component, event, record)


def _log_wandb(component, event, record):
    try:
        import wandb
        if wandb.run is None:
            return
        prefix = 'fusion_audit/{}/{}/'.format(component, event)
        payload = {}
        for key, value in record.items():
            if isinstance(value, (int, float, bool)) and key not in ('time',):
                payload[prefix + key] = value
        if payload:
            wandb.log(payload)
    except Exception:
        return


def _log_artifact():
    if not enabled() or not _wandb_enabled() or not _rank_zero():
        return
    path = _audit_path()
    if not os.path.exists(path):
        return
    try:
        import wandb
        if wandb.run is None:
            return
        artifact = wandb.Artifact('univ2x-fusion-audit', type='fusion-audit')
        artifact.add_file(path)
        wandb.log_artifact(artifact)
    except Exception:
        return


atexit.register(_log_artifact)
