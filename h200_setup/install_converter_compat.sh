#!/usr/bin/env bash
set -euo pipefail
PY=${PY:-python}

${PY} - <<"PY"
from pathlib import Path
import site
sp = Path(site.getsitepackages()[0])
base = sp / "mmdet3d" / "core" / "bbox"
base.mkdir(parents=True, exist_ok=True)
(base.parent / "__init__.py").write_text("\n")
(base / "__init__.py").write_text("\n")
(base / "box_np_ops.py").write_text("from mmdet3d.structures.bbox_3d.utils import points_cam2img\n")
print("installed mmdet3d.core.bbox.box_np_ops shim at {}".format(base / "box_np_ops.py"))
PY

# Add MMCV 2.x aliases needed by the legacy SPD converter.
${PY} - <<"PY"
from pathlib import Path
import site
sp = Path(site.getsitepackages()[0])
p = sp / "sitecustomize.py"
block = """
# H200 UniMM-V2X converter compatibility for MMCV 2.x.
try:
    import mmcv
    import mmengine
    for _name in ("dump", "load", "mkdir_or_exist", "track_iter_progress"):
        if not hasattr(mmcv, _name) and hasattr(mmengine, _name):
            setattr(mmcv, _name, getattr(mmengine, _name))
except Exception:
    pass
""".lstrip()
old = p.read_text() if p.exists() else ""
marker = "# H200 UniMM-V2X converter compatibility for MMCV 2.x."
if marker not in old:
    p.write_text(old + ("\n" if old and not old.endswith("\n") else "") + block)
print("installed mmcv compatibility aliases via {}".format(p))
PY
