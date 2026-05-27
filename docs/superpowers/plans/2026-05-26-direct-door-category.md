# Direct Door Category — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 8 "direct" door categories (LeverDoor/RoundDoor direct_pull/push, Cabinet/Fridge/Window/Safe direct_pull) where the handle is fixed at zero angle, enabling the robot to pull/push without rotating the handle.

**Architecture:** Asset-driven design — URDF `joint_2` limits set to `[0, 0]` makes the handle immovable. The DP3 expert generator detects "direct" task names and explicitly skips the handle-rotation phase in stage 2. Original UniDoorManip simulation code requires zero changes because the `open_door_flag`/`open_door_stage` checks naturally evaluate to True when handle range is zero.

**Tech Stack:** Python 3.8, xml.etree, shutil, Isaac Gym, PyTorch, Zarr, Bash

---

## File Structure

| File | Responsibility |
|---|---|
| `/tmp/generate_direct_assets.py` | One-time script: copy assets, modify URDF joint_2 |
| `generated_datasets/LeverDoor_direct_pull/` etc. (8 dirs) | New asset categories with `joint_2=[0,0]` |
| `generated_datasets/STRUCTURE.md` | Documentation of dataset structure |
| `3D-Diffusion-Policy/.../gen_demonstration_unidoormanip.py` | Expert trajectory: `_TASK_BASE_TASK`, `_DIRECT_TASKS`, stage-2 skip |
| `3D-Diffusion-Policy/.../unidoormanip_wrapper.py` | `TASK_ASSET_DEFAULTS` mapping |
| `3D-Diffusion-Policy/scripts/gen_demonstration_unidoormanip.sh` | Shell: `TASK_DIR_MAP`, valid task names comment |
| `3D-Diffusion-Policy/.../config/task/unidoormanip_*_direct_*_mobile.yaml` (8 files) | Hydra training/eval configs |

---

### Task 1: Create the Asset Generation Script

**Files:**
- Create: `/tmp/generate_direct_assets.py`

- [ ] **Step 1: Write the generation script**

```python
#!/usr/bin/env python3
"""One-time script: generate 'direct' asset categories by copying ccw source
categories and setting joint_2 handle limits to [0, 0] in each mobility.urdf."""
import os
import sys
import shutil
import json
import xml.etree.ElementTree as ET

SOURCE_ROOT = "/Titan/code/robohike_ws/src/UniDoorManip/generated_datasets"

MAPPING = {
    "LeverDoor_direct_pull":  "LeverDoor_ccw_pull",
    "LeverDoor_direct_push":  "LeverDoor_ccw_push",
    "RoundDoor_direct_pull":  "RoundDoor_ccw_pull",
    "RoundDoor_direct_push":  "RoundDoor_ccw_push",
    "Cabinet_direct_pull":    "Cabinet_ccw_pull",
    "Fridge_direct_pull":     "Fridge_ccw_pull",
    "Window_direct_pull":     "Window_ccw_pull",
    "Safe_direct_pull":       "Safe_ccw_pull",
}

def modify_urdf_handle_limits(urdf_path: str):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    found = False
    for joint in root.iter("joint"):
        if joint.get("name") == "joint_2":
            if joint.get("type") != "revolute":
                raise RuntimeError(
                    f"{urdf_path}: joint_2 type is '{joint.get('type')}', expected 'revolute'"
                )
            limit = joint.find("limit")
            if limit is None:
                raise RuntimeError(f"{urdf_path}: joint_2 has no <limit> element")
            limit.set("lower", "0")
            limit.set("upper", "0")
            found = True
            break
    if not found:
        raise RuntimeError(f"{urdf_path}: joint_2 not found")
    tree.write(urdf_path, xml_declaration=False, encoding="utf-8")

def main():
    for dst_name, src_name in MAPPING.items():
        src_dir = os.path.join(SOURCE_ROOT, src_name)
        dst_dir = os.path.join(SOURCE_ROOT, dst_name)
        if not os.path.isdir(src_dir):
            print(f"SKIP: source {src_dir} does not exist")
            continue
        if os.path.exists(dst_dir):
            print(f"SKIP: target {dst_dir} already exists")
            continue

        print(f"Generating {dst_name} from {src_name} ...")

        os.makedirs(dst_dir, exist_ok=True)

        split_src = os.path.join(src_dir, "train_test_split.json")
        split_dst = os.path.join(dst_dir, "train_test_split.json")
        shutil.copy2(split_src, split_dst)
        print(f"  copied train_test_split.json")

        asset_dirs = sorted(
            d for d in os.listdir(src_dir)
            if os.path.isdir(os.path.join(src_dir, d))
            and os.path.isfile(os.path.join(src_dir, d, "mobility.urdf"))
        )
        for asset_id in asset_dirs:
            src_asset = os.path.join(src_dir, asset_id)
            dst_asset = os.path.join(dst_dir, asset_id)
            shutil.copytree(src_asset, dst_asset)
            urdf_path = os.path.join(dst_asset, "mobility.urdf")
            modify_urdf_handle_limits(urdf_path)
            print(f"  {asset_id}: joint_2 -> [0, 0]")

        print(f"  Done: {len(asset_dirs)} assets\n")

    print("All categories generated.")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script**

Run: `python /tmp/generate_direct_assets.py`
Expected: Output showing each category being generated with per-asset progress lines.

- [ ] **Step 3: Verify script output — count assets per category match source**

Run:
```bash
for pair in "LeverDoor_direct_pull LeverDoor_ccw_pull" "LeverDoor_direct_push LeverDoor_ccw_push" "RoundDoor_direct_pull RoundDoor_ccw_pull" "RoundDoor_direct_push RoundDoor_ccw_push" "Cabinet_direct_pull Cabinet_ccw_pull" "Fridge_direct_pull Fridge_ccw_pull" "Window_direct_pull Window_ccw_pull" "Safe_direct_pull Safe_ccw_pull"; do
    dst=$(echo $pair | awk '{print $1}')
    src=$(echo $pair | awk '{print $2}')
    cnt_dst=$(ls -d /Titan/code/robohike_ws/src/UniDoorManip/generated_datasets/$dst/*/mobility.urdf 2>/dev/null | wc -l)
    cnt_src=$(ls -d /Titan/code/robohike_ws/src/UniDoorManip/generated_datasets/$src/*/mobility.urdf 2>/dev/null | wc -l)
    status="OK"
    [ "$cnt_dst" != "$cnt_src" ] && status="MISMATCH"
    echo "$dst: $cnt_dst assets (source: $cnt_src) $status"
done
```
Expected: All 8 pairs show OK with matching asset counts.

- [ ] **Step 4: Commit**

```bash
cd /Titan/code/robohike_ws/src/UniDoorManip
git add generated_datasets/LeverDoor_direct_pull/ generated_datasets/LeverDoor_direct_push/ generated_datasets/RoundDoor_direct_pull/ generated_datasets/RoundDoor_direct_push/ generated_datasets/Cabinet_direct_pull/ generated_datasets/Fridge_direct_pull/ generated_datasets/Window_direct_pull/ generated_datasets/Safe_direct_pull/
git commit -m "feat: add direct door asset categories (handle fixed at zero angle)"
```

---

### Task 2: Verify Asset Integrity

**Files:**
- Verify: All 8 `generated_datasets/{Category}/` directories

- [ ] **Step 1: Verify URDF joint_2 limits are [0, 0] in every asset**

```bash
python3 -c "
import os, xml.etree.ElementTree as ET
root = '/Titan/code/robohike_ws/src/UniDoorManip/generated_datasets'
categories = ['LeverDoor_direct_pull','LeverDoor_direct_push','RoundDoor_direct_pull','RoundDoor_direct_push','Cabinet_direct_pull','Fridge_direct_pull','Window_direct_pull','Safe_direct_pull']
errors = []
for cat in categories:
    cat_dir = os.path.join(root, cat)
    for asset_id in sorted(os.listdir(cat_dir)):
        urdf = os.path.join(cat_dir, asset_id, 'mobility.urdf')
        if not os.path.isfile(urdf):
            continue
        tree = ET.parse(urdf)
        for joint in tree.getroot().iter('joint'):
            if joint.get('name') == 'joint_2':
                limit = joint.find('limit')
                lo, up = limit.get('lower'), limit.get('upper')
                if lo != '0' or up != '0':
                    errors.append(f'{cat}/{asset_id}: joint_2 limits=[{lo},{up}] expected [0,0]')
                break
if errors:
    print('ERRORS:')
    for e in errors: print(f'  {e}')
    exit(1)
else:
    print(f'All OK: joint_2=[0,0] verified across all assets')
"
```
Expected: `All OK: joint_2=[0,0] verified across all assets`

- [ ] **Step 2: Verify train_test_split.json exists in each category**

Run:
```bash
for cat in LeverDoor_direct_pull LeverDoor_direct_push RoundDoor_direct_pull RoundDoor_direct_push Cabinet_direct_pull Fridge_direct_pull Window_direct_pull Safe_direct_pull; do
    f="/Titan/code/robohike_ws/src/UniDoorManip/generated_datasets/$cat/train_test_split.json"
    if [ -f "$f" ]; then
        echo "$cat: OK ($(python3 -c "import json; d=json.load(open('$f')); print(f\"train={len(d['train'])}, test={len(d['test'])}\")"))"
    else
        echo "$cat: MISSING"
    fi
done
```
Expected: All 8 categories show OK with train/test counts.

- [ ] **Step 3: Verify key files exist in each asset subdirectory**

Run:
```bash
for cat in LeverDoor_direct_pull LeverDoor_direct_push RoundDoor_direct_pull RoundDoor_direct_push Cabinet_direct_pull Fridge_direct_pull Window_direct_pull Safe_direct_pull; do
    dir="/Titan/code/robohike_ws/src/UniDoorManip/generated_datasets/$cat"
    first=$(ls -d "$dir"/*/ 2>/dev/null | head -1)
    missing=""
    for f in mobility.urdf bounding_box.json handle_bounding.json; do
        [ -f "$first/$f" ] || missing="$missing $f"
    done
    [ -d "$first/texture_dae" ] || missing="$missing texture_dae/"
    if [ -z "$missing" ]; then
        echo "$cat: OK"
    else
        echo "$cat: MISSING$missing"
    fi
done
```
Expected: All 8 categories show OK.

- [ ] **Step 4: Commit (if any fixes needed)**

### Task 3: Update STRUCTURE.md

**Files:**
- Modify: `/Titan/code/robohike_ws/src/UniDoorManip/generated_datasets/STRUCTURE.md`

- [ ] **Step 1: Add direct categories to the tree diagram (after line 44)**

Insert after line 44 (`├── LeverDoor_cw_push/`):
```
│
├── LeverDoor_direct_pull/                 # Lever already unlocked + pull (20 assets)
├── LeverDoor_direct_push/                 # Lever already unlocked + push (20 assets)
├── RoundDoor_direct_pull/                 # Round knob already unlocked + pull (18 assets)
├── RoundDoor_direct_push/                 # Round knob already unlocked + push (18 assets)
```

- [ ] **Step 2: Fix pre-existing outdated category names (lines 46, 58, 66, 78, 86)**

Replace the old single-word names with the actual directory names:

Line 46: `├── Cabinet/` → `├── Cabinet_ccw_pull/` and add `├── Cabinet_cw_pull/`
Line 58: `├── Fridge/` → `├── Fridge_ccw_pull/` and add `├── Fridge_cw_pull/`
Line 66: `├── Safe/` → `├── Safe_ccw_pull/` and add `├── Safe_cw_pull/`
Line 78: `├── Car/` → keep as is (only one variant)
Line 86: `├── Window/` → `├── Window_ccw_pull/` and add `├── Window_cw_pull/`

Also add direct variants after each group:
```
├── Cabinet_direct_pull/                   # Cabinet already unlocked + pull (9 assets)
├── Fridge_direct_pull/                    # Fridge already unlocked + pull (7 assets)
├── Safe_direct_pull/                      # Safe already unlocked + pull (13 assets)
├── Window_direct_pull/                    # Window already unlocked + pull (24 assets)
```

- [ ] **Step 3: Update Quick Statistics (line 274)**

Change:
```
Total Assets:     140
Total Categories: 7
```
To:
```
Total Assets:     280
Total Categories: 15
```

- [ ] **Step 4: Update "By Category" code block (line 285)**

Replace the `categories` list with:
```python
categories = [
    "RoundDoor_ccw_pull", "RoundDoor_ccw_push", "RoundDoor_cw_pull", "RoundDoor_cw_push",
    "RoundDoor_direct_pull", "RoundDoor_direct_push",
    "LeverDoor_ccw_pull", "LeverDoor_ccw_push", "LeverDoor_cw_pull", "LeverDoor_cw_push",
    "LeverDoor_direct_pull", "LeverDoor_direct_push",
    "Cabinet_ccw_pull", "Cabinet_cw_pull", "Cabinet_direct_pull",
    "Fridge_ccw_pull", "Fridge_cw_pull", "Fridge_direct_pull",
    "Safe_ccw_pull", "Safe_cw_pull", "Safe_direct_pull",
    "Car_pull",
    "Window_ccw_pull", "Window_cw_pull", "Window_direct_pull",
]
```

- [ ] **Step 5: Commit**

```bash
cd /Titan/code/robohike_ws/src/UniDoorManip
git add generated_datasets/STRUCTURE.md
git commit -m "docs: add direct door categories to STRUCTURE.md, fix outdated names"
```

---

### Task 4: Update DP3 Expert Trajectory Generator

**Files:**
- Modify: `/Titan/code/robohike_ws/src/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/gen_demonstration_unidoormanip.py`

- [ ] **Step 1: Add `_DIRECT_TASKS` and expand `_TASK_BASE_TASK`**

At line ~93, modify `_TASK_BASE_TASK` to add direct entries:

```python
    _TASK_BASE_TASK = {
        'leverdoor_ccw_pull': 'leverdoor',
        'leverdoor_ccw_push': 'leverdoor',
        'leverdoor_cw_pull':  'leverdoor',
        'leverdoor_cw_push':  'leverdoor',
        'leverdoor_direct_pull':  'leverdoor',
        'leverdoor_direct_push':  'leverdoor',
        'rounddoor_ccw_pull': 'rounddoor',
        'rounddoor_ccw_push': 'rounddoor',
        'rounddoor_cw_pull':  'rounddoor',
        'rounddoor_cw_push':  'rounddoor',
        'rounddoor_direct_pull':  'rounddoor',
        'rounddoor_direct_push':  'rounddoor',
        'cabinet_ccw_pull':   'cabinet',
        'cabinet_cw_pull':    'cabinet',
        'cabinet_direct_pull':    'cabinet',
        'fridge_ccw_pull':    'fridge',
        'fridge_cw_pull':     'fridge',
        'fridge_direct_pull':     'fridge',
        'window_ccw_pull':    'window',
        'window_cw_pull':     'window',
        'window_direct_pull':     'window',
        'car_pull':           'car',
        'safe_ccw_pull':      'safe',
        'safe_cw_pull':       'safe',
        'safe_direct_pull':       'safe',
    }
```

Immediately after `_TASK_BASE_TASK`, add the new class attribute:

```python
    _DIRECT_TASKS = {
        'leverdoor_direct_pull', 'leverdoor_direct_push',
        'rounddoor_direct_pull', 'rounddoor_direct_push',
        'cabinet_direct_pull', 'fridge_direct_pull',
        'window_direct_pull', 'safe_direct_pull',
    }
```

- [ ] **Step 2: Add stage-2 skip logic for direct tasks**

At line ~406 (right after `open_dir = quat_axis(handle_q, axis=2)`), insert:

```python
            # Direct tasks: handle is already unlocked, skip handle rotation entirely
            if self.task_name in self._DIRECT_TASKS:
                isaac.open_door_flag[mask2] = True
```

- [ ] **Step 3: Update docstring (line 7-9)**

Change:
```
    Stage 3: Open (180 steps) - Rotate handle + pull object
```
Add to supported categories list (line 9):
```
leverdoor_direct_pull, leverdoor_direct_push, rounddoor_direct_pull, rounddoor_direct_push, cabinet_direct_pull, fridge_direct_pull, window_direct_pull, safe_direct_pull
```

- [ ] **Step 4: Update `run_collection` docstring (line 866)**

Add the 8 new task names to the `task_name` description.

- [ ] **Step 5: Verify no syntax errors**

Run: `python -c "import ast; ast.parse(open('/Titan/code/robohike_ws/src/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/gen_demonstration_unidoormanip.py').read()); print('Syntax OK')"`
Expected: `Syntax OK`

- [ ] **Step 6: Commit**

```bash
cd /Titan/code/robohike_ws/src/3D-Diffusion-Policy
git add 3D-Diffusion-Policy/diffusion_policy_3d/gen_demonstration_unidoormanip.py
git commit -m "feat: add direct door task support to expert trajectory generator"
```

---

### Task 5: Update DP3 Environment Wrapper

**Files:**
- Modify: `/Titan/code/robohike_ws/src/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/env/unidoormanip/unidoormanip_wrapper.py`

- [ ] **Step 1: Add 8 new entries to `TASK_ASSET_DEFAULTS` (after line 83)**

Insert after the `'window_cw_pull'` entry:

```python
    'leverdoor_direct_pull':  ('LeverDoor_direct_pull',  'opensource_leverdoor',  '99650069960003'),
    'leverdoor_direct_push':  ('LeverDoor_direct_push',  'opensource_leverdoor',  '99650069960003'),
    'rounddoor_direct_pull':  ('RoundDoor_direct_pull',  'opensource_rounddoor',  '99650069962004'),
    'rounddoor_direct_push':  ('RoundDoor_direct_push',  'opensource_rounddoor',  '99650069962004'),
    'cabinet_direct_pull':    ('Cabinet_direct_pull',    'opensource_cabinet',    '99613029962004'),
    'fridge_direct_pull':     ('Fridge_direct_pull',     'opensource_fridge',     '99614019960005'),
    'window_direct_pull':     ('Window_direct_pull',     'opensource_window',     '99690049969510'),
    'safe_direct_pull':       ('Safe_direct_pull',       'opensource_safe',       '99611129961204'),
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('/Titan/code/robohike_ws/src/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/env/unidoormanip/unidoormanip_wrapper.py').read()); print('Syntax OK')"`
Expected: `Syntax OK`

- [ ] **Step 3: Commit**

```bash
cd /Titan/code/robohike_ws/src/3D-Diffusion-Policy
git add 3D-Diffusion-Policy/diffusion_policy_3d/env/unidoormanip/unidoormanip_wrapper.py
git commit -m "feat: add direct door task defaults to environment wrapper"
```

---

### Task 6: Update DP3 Shell Script

**Files:**
- Modify: `/Titan/code/robohike_ws/src/3D-Diffusion-Policy/scripts/gen_demonstration_unidoormanip.sh`

- [ ] **Step 1: Update "Valid task names" comment (line 19-24)**

Replace the comment block with:
```bash
# Valid task names (must match generated_datasets/ folder names):
#   leverdoor_ccw_pull  leverdoor_ccw_push  leverdoor_cw_pull  leverdoor_cw_push
#   leverdoor_direct_pull  leverdoor_direct_push
#   rounddoor_ccw_pull  rounddoor_ccw_push  rounddoor_cw_pull  rounddoor_cw_push
#   rounddoor_direct_pull  rounddoor_direct_push
#   cabinet_ccw_pull  cabinet_cw_pull  cabinet_direct_pull
#   car_pull
#   fridge_ccw_pull  fridge_cw_pull  fridge_direct_pull
#   safe_ccw_pull  safe_cw_pull  safe_direct_pull
#   window_ccw_pull  window_cw_pull  window_direct_pull
```

- [ ] **Step 2: Add 8 new entries to `TASK_DIR_MAP` (after line 96)**

Insert after `[window_cw_pull]="Window_cw_pull"`:

```bash
    [leverdoor_direct_pull]="LeverDoor_direct_pull"
    [leverdoor_direct_push]="LeverDoor_direct_push"
    [rounddoor_direct_pull]="RoundDoor_direct_pull"
    [rounddoor_direct_push]="RoundDoor_direct_push"
    [cabinet_direct_pull]="Cabinet_direct_pull"
    [fridge_direct_pull]="Fridge_direct_pull"
    [window_direct_pull]="Window_direct_pull"
    [safe_direct_pull]="Safe_direct_pull"
```

- [ ] **Step 3: Verify shell syntax**

Run: `bash -n /Titan/code/robohike_ws/src/3D-Diffusion-Policy/scripts/gen_demonstration_unidoormanip.sh`
Expected: No output (script passes syntax check)

- [ ] **Step 4: Commit**

```bash
cd /Titan/code/robohike_ws/src/3D-Diffusion-Policy
git add scripts/gen_demonstration_unidoormanip.sh
git commit -m "feat: add direct door tasks to collection shell script"
```

---

### Task 7: Create YAML Task Configs

**Files:**
- Create: 8 YAML files under `/Titan/code/robohike_ws/src/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/config/task/`

- [ ] **Step 1: Create `unidoormanip_leverdoor_direct_pull_mobile.yaml`**

Copy the source config as a template, noting that leverdoor has 20 train assets:
```bash
cd /Titan/code/robohike_ws/src/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/config/task
cp unidoormanip_leverdoor_ccw_pull_mobile.yaml unidoormanip_leverdoor_direct_pull_mobile.yaml
```
Then modify the file: replace `leverdoor_ccw_pull` with `leverdoor_direct_pull` everywhere in the file. The `asset_ids` list and `aug_*` parameters stay identical to the source.

- [ ] **Step 2: Create `unidoormanip_leverdoor_direct_push_mobile.yaml`**

```bash
cd /Titan/code/robohike_ws/src/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/config/task
cp unidoormanip_leverdoor_ccw_push_mobile.yaml unidoormanip_leverdoor_direct_push_mobile.yaml
```
Replace all `leverdoor_ccw_push` with `leverdoor_direct_push`.

- [ ] **Step 3: Create `unidoormanip_rounddoor_direct_pull_mobile.yaml`**

```bash
cd /Titan/code/robohike_ws/src/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/config/task
cp unidoormanip_rounddoor_ccw_pull_mobile.yaml unidoormanip_rounddoor_direct_pull_mobile.yaml
```
Replace all `rounddoor_ccw_pull` with `rounddoor_direct_pull`. (18 train assets, zarr name uses `_18_`)

- [ ] **Step 4: Create `unidoormanip_rounddoor_direct_push_mobile.yaml`**

```bash
cd /Titan/code/robohike_ws/src/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/config/task
cp unidoormanip_rounddoor_ccw_push_mobile.yaml unidoormanip_rounddoor_direct_push_mobile.yaml
```
Replace all `rounddoor_ccw_push` with `rounddoor_direct_push`.

- [ ] **Step 5: Create `unidoormanip_cabinet_direct_pull_mobile.yaml`**

```bash
cd /Titan/code/robohike_ws/src/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/config/task
cp unidoormanip_cabinet_ccw_pull_mobile.yaml unidoormanip_cabinet_direct_pull_mobile.yaml
```
Replace all `cabinet_ccw_pull` with `cabinet_direct_pull`. (9 train assets, zarr uses `_9_`)

- [ ] **Step 6: Create `unidoormanip_fridge_direct_pull_mobile.yaml`**

```bash
cd /Titan/code/robohike_ws/src/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/config/task
cp unidoormanip_fridge_ccw_pull_mobile.yaml unidoormanip_fridge_direct_pull_mobile.yaml
```
Replace all `fridge_ccw_pull` with `fridge_direct_pull`. (7 train assets, zarr uses `_7_`)

- [ ] **Step 7: Create `unidoormanip_window_direct_pull_mobile.yaml`**

```bash
cd /Titan/code/robohike_ws/src/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/config/task
cp unidoormanip_window_ccw_pull_mobile.yaml unidoormanip_window_direct_pull_mobile.yaml
```
Replace all `window_ccw_pull` with `window_direct_pull`. (24 train assets, zarr uses `_24_`)

- [ ] **Step 8: Create `unidoormanip_safe_direct_pull_mobile.yaml`**

```bash
cd /Titan/code/robohike_ws/src/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/config/task
cp unidoormanip_safe_ccw_pull_mobile.yaml unidoormanip_safe_direct_pull_mobile.yaml
```
Replace all `safe_ccw_pull` with `safe_direct_pull`. (13 train assets, zarr uses `_13_`)

- [ ] **Step 9: Verify all 8 new configs**

Run:
```bash
for f in leverdoor_direct_pull leverdoor_direct_push rounddoor_direct_pull rounddoor_direct_push cabinet_direct_pull fridge_direct_pull window_direct_pull safe_direct_pull; do
    path="/Titan/code/robohike_ws/src/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/config/task/unidoormanip_${f}_mobile.yaml"
    task=$(grep 'task_name:' "$path" | head -1 | awk '{print $2}')
    zarr=$(grep 'zarr_path:' "$path" | awk '{print $2}')
    echo "$f: task=$task zarr=$zarr"
done
```
Expected: All 8 show consistent `task=` and `zarr=` values matching their filename patterns. No entry should reference `ccw_`, `cw_`, or the original source category names.

- [ ] **Step 10: Commit**

```bash
cd /Titan/code/robohike_ws/src/3D-Diffusion-Policy
git add 3D-Diffusion-Policy/diffusion_policy_3d/config/task/unidoormanip_leverdoor_direct_pull_mobile.yaml \
    3D-Diffusion-Policy/diffusion_policy_3d/config/task/unidoormanip_leverdoor_direct_push_mobile.yaml \
    3D-Diffusion-Policy/diffusion_policy_3d/config/task/unidoormanip_rounddoor_direct_pull_mobile.yaml \
    3D-Diffusion-Policy/diffusion_policy_3d/config/task/unidoormanip_rounddoor_direct_push_mobile.yaml \
    3D-Diffusion-Policy/diffusion_policy_3d/config/task/unidoormanip_cabinet_direct_pull_mobile.yaml \
    3D-Diffusion-Policy/diffusion_policy_3d/config/task/unidoormanip_fridge_direct_pull_mobile.yaml \
    3D-Diffusion-Policy/diffusion_policy_3d/config/task/unidoormanip_window_direct_pull_mobile.yaml \
    3D-Diffusion-Policy/diffusion_policy_3d/config/task/unidoormanip_safe_direct_pull_mobile.yaml
git commit -m "feat: add YAML task configs for 8 direct door categories"
```

---

### Task 8: Integration Test — Expert Collection Smoke Test

**Files:**
- Test: LeverDoor_direct_pull asset with DP3 expert collection

- [ ] **Step 1: Run quick smoke test for leverdoor_direct_pull**

Run:
```bash
cd /Titan/code/robohike_ws/src/3D-Diffusion-Policy/3D-Diffusion-Policy
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia:/usr/local/cuda/lib64
export MUJOCO_GL=egl
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
python -m diffusion_policy_3d.gen_demonstration_unidoormanip \
    --task_name leverdoor_direct_pull \
    --asset_ids "99650069960003" \
    --episodes_per_asset 5 \
    --num_envs 5 \
    --max_attempts 30 \
    --headless \
    --save_video
```
**Prerequisites:** Isaac Gym installed, GPU available, Python environment with DP3 installed.

Expected output includes:
- `Task: leverdoor_direct_pull`
- Episodes collected (target: 5 per asset)
- Door opens without the handle-rotation phase visible in trajectory
- A zarr file created under `data/` with `unidoormanip_leverdoor_direct_pull` in the filename

- [ ] **Step 2: Verify handle DOF is zero in collected data**

Run:
```bash
python3 -c "
import zarr, numpy as np, glob, os
# Find the most recent direct test zarr
root = '/Titan/code/robohike_ws/src/3D-Diffusion-Policy/data'
matches = sorted(glob.glob(os.path.join(root, '*leverdoor_direct*quick*')))
if not matches:
    matches = sorted(glob.glob(os.path.join(root, '*leverdoor_direct*expert*.zarr')), key=os.path.getmtime, reverse=True)
if not matches:
    print('No zarr found for leverdoor_direct')
    exit(1)
z = zarr.open(matches[-1], mode='r')
handle_kin = z['data/joint_kinematics_handle'][:]
# For direct tasks, handle theta_min == theta_max == 0 (both should be 0 or near 0)
theta_min = handle_kin[:, 1]
theta_max = handle_kin[:, 2]
print(f'zarr: {matches[-1]}')
print(f'handle theta_min = {np.unique(theta_min)}')
print(f'handle theta_max = {np.unique(theta_max)}')
assert np.allclose(theta_min, 0, atol=1e-6), f'Expected theta_min=0, got {theta_min}'
assert np.allclose(theta_max, 0, atol=1e-6), f'Expected theta_max=0, got {theta_max}'
print('PASS: handle kinematics confirm [0,0] limits')
" 2>&1
```
Expected: `PASS: handle kinematics confirm [0,0] limits`

- [ ] **Step 3: Regression test — verify original task still works**

Run:
```bash
cd /Titan/code/robohike_ws/src/3D-Diffusion-Policy/3D-Diffusion-Policy
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia:/usr/local/cuda/lib64
export MUJOCO_GL=egl
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
python -m diffusion_policy_3d.gen_demonstration_unidoormanip \
    --task_name leverdoor_ccw_pull \
    --asset_ids "99650069960003" \
    --episodes_per_asset 5 \
    --num_envs 5 \
    --max_attempts 30 \
    --headless \
    --save_video
```
Expected: The original leverdoor_ccw_pull still collects successfully. The video should show handle rotation before door pull (verifying handle-rotation phase is still present).

- [ ] **Step 4: Commit (if any fixes needed)**

---

### Task 9: Self-Review Checklist

Run through this checklist before declaring completion:

```bash
echo "=== Asset checks ==="
echo "Direct categories exist:"
ls -d /Titan/code/robohike_ws/src/UniDoorManip/generated_datasets/*direct* 2>/dev/null | wc -l
echo "(expected: 8)"

echo ""
echo "joint_2 limits all [0,0]:"
python3 -c "
import os, xml.etree.ElementTree as ET
root = '/Titan/code/robohike_ws/src/UniDoorManip/generated_datasets'
cats = [d for d in os.listdir(root) if 'direct' in d.lower()]
bad = 0
for cat in cats:
    for aid in os.listdir(os.path.join(root, cat)):
        urdf = os.path.join(root, cat, aid, 'mobility.urdf')
        if not os.path.isfile(urdf): continue
        tree = ET.parse(urdf)
        for j in tree.getroot().iter('joint'):
            if j.get('name')=='joint_2':
                lo=j.find('limit').get('lower'); up=j.find('limit').get('upper')
                if lo != '0' or up != '0':
                    print(f'BAD: {cat}/{aid} joint_2=[{lo},{up}]'); bad+=1
print(f'Bad URDFs: {bad} (expected: 0)')
"

echo ""
echo "=== DP3 code checks ==="
echo "TASK_BASE_TASK direct entries:"
grep -c 'direct' /Titan/code/robohike_ws/src/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/gen_demonstration_unidoormanip.py | head -1
echo "(expected: >0)"

echo "TASK_ASSET_DEFAULTS direct entries:"
grep -c 'direct' /Titan/code/robohike_ws/src/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/env/unidoormanip/unidoormanip_wrapper.py | head -1
echo "(expected: >0)"

echo "TASK_DIR_MAP direct entries:"
grep -c 'direct' /Titan/code/robohike_ws/src/3D-Diffusion-Policy/scripts/gen_demonstration_unidoormanip.sh | head -1
echo "(expected: >0)"

echo "YAML configs:"
ls /Titan/code/robohike_ws/src/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/config/task/*direct* 2>/dev/null | wc -l
echo "(expected: 8)"
```
