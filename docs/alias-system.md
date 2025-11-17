# Branch Alias System

## Problem

PID branches have different names in data vs MC files:
- **Data files**: Use `MC15TuneV1` PID branches (e.g., `h1_MC15TuneV1_ProbNNk`)
- **MC files**: Use `MC12TuneV2/V3/V4` PID branches (e.g., `h1_MC12TuneV4_ProbNNk`)

This means our analysis code would need different branch names depending on whether you're analyzing data or MC.

## Solution

The **branch alias system** provides:
1. **Common names** for use in our analysis code
2. **Automatic resolution** to file-specific names when loading
3. **Automatic normalization** back to common names after loading

## How It Works

### 1. Define Aliases in `branches_config.toml`

```toml
[aliases.pid]
# Common name = {data = "data_branch", mc = "mc_branch"}
h1_ProbNNk = {data = "h1_MC15TuneV1_ProbNNk", mc = "h1_MC12TuneV4_ProbNNk"}
h2_ProbNNk = {data = "h2_MC15TuneV1_ProbNNk", mc = "h2_MC12TuneV4_ProbNNk"}
```

### 2. Use Common Names in Config

In `branches_config.toml`, use common names:
```toml
[branches.pid]
h1 = ["h1_ProbNNk", "h1_ProbNNpi", "h1_ProbNNp"]
h2 = ["h2_ProbNNk", "h2_ProbNNpi", "h2_ProbNNp"]
```

In `selection.toml`, use common names:
```toml
[[hadronic_kaons.cuts]]
name = "h1_kaon_id"
branch = "h1_ProbNNk"  # Common name!
operator = ">"
value = 0.4
```

### 3. Loaders Handle Everything Automatically

When we load data:
```python
from data_loader import DataLoader

loader = DataLoader(data_dir)
data = loader.load_data(
    years=['16'],
    polarities=['MD'],
    track_types=['LL'],
    channel_name='B2L0barPKpKm',
    preset='standard',
    use_aliases=True  # Default
)

# Events have common names: h1_ProbNNk, h2_ProbNNk, etc.
events = data['16_MD_LL']
print(events.fields)  # ['h1_ProbNNk', 'h2_ProbNNk', ...]
```

What happens behind the scenes:
1. Request includes `h1_ProbNNk` (common name)
2. Loader resolves to `h1_MC15TuneV1_ProbNNk` (data-specific)
3. Loads from file using data-specific name
4. Renames back to `h1_ProbNNk` (common name)

When we load MC:
```python
from mc_loader import MCLoader

loader = MCLoader(mc_dir)
mc_data = loader.load_reconstructed(
    sample_name='Jpsi',
    years=['18'],
    polarities=['MD'],
    track_types=['LL'],
    channel_name='B2L0barPKpKm',
    preset='standard',
    use_aliases=True  # Default
)

# Events have common names: h1_ProbNNk, h2_ProbNNk, etc.
events = mc_data['18_MD_LL']
print(events.fields)  # ['h1_ProbNNk', 'h2_ProbNNk', ...]
```

What happens behind the scenes:
1. Request includes `h1_ProbNNk` (common name)
2. Loader resolves to `h1_MC12TuneV4_ProbNNk` (MC-specific)
3. Loads from file using MC-specific name
4. Renames back to `h1_ProbNNk` (common name)

## Result

**Our analysis code is identical for data and MC!**

```python
# This works for BOTH data and MC:
def apply_pid_cut(events):
    return events[events.h1_ProbNNk > 0.4]

# selection.toml works for BOTH data and MC:
[[hadronic_kaons.cuts]]
branch = "h1_ProbNNk"  # Common name works everywhere!
```

## Aliased Branches

Currently aliased PID branches:

| Common Name | Data Branch | MC Branch |
|-------------|-------------|-----------|
| `h1_ProbNNk` | `h1_MC15TuneV1_ProbNNk` | `h1_MC12TuneV4_ProbNNk` |
| `h1_ProbNNp` | `h1_MC15TuneV1_ProbNNp` | `h1_MC12TuneV4_ProbNNp` |
| `h1_ProbNNpi` | `h1_MC15TuneV1_ProbNNpi` | `h1_MC12TuneV4_ProbNNpi` |
| `h1_ProbNNmu` | `h1_MC15TuneV1_ProbNNmu` | `h1_MC12TuneV4_ProbNNmu` |
| `h2_ProbNNk` | `h2_MC15TuneV1_ProbNNk` | `h2_MC12TuneV4_ProbNNk` |
| `h2_ProbNNp` | `h2_MC15TuneV1_ProbNNp` | `h2_MC12TuneV4_ProbNNp` |
| `h2_ProbNNpi` | `h2_MC15TuneV1_ProbNNpi` | `h2_MC12TuneV4_ProbNNpi` |
| `h2_ProbNNmu` | `h2_MC15TuneV1_ProbNNmu` | `h2_MC12TuneV4_ProbNNmu` |
| `p_ProbNNp` | `p_MC15TuneV1_ProbNNp` | `p_MC12TuneV4_ProbNNp` |
| `p_ProbNNpi` | `p_MC15TuneV1_ProbNNpi` | `p_MC12TuneV4_ProbNNpi` |
| `Lp_ProbNNp` | `Lp_MC15TuneV1_ProbNNp` | `Lp_MC12TuneV4_ProbNNp` |
| `Lp_ProbNNpi` | `Lp_MC15TuneV1_ProbNNpi` | `Lp_MC12TuneV4_ProbNNpi` |
| `Lpi_ProbNNp` | `Lpi_MC15TuneV1_ProbNNp` | `Lpi_MC12TuneV4_ProbNNp` |
| `Lpi_ProbNNpi` | `Lpi_MC15TuneV1_ProbNNpi` | `Lpi_MC12TuneV4_ProbNNpi` |

**ID branches** (like `h1_ID`, `h2_ID`) are **NOT aliased** because they have the same name in both data and MC.

## Adding New Aliases

To add more aliases, edit `branches_config.toml`:

```toml
[aliases.pid]
# Add new alias
new_common_name = {data = "data_specific_name", mc = "mc_specific_name"}
```

## Testing

Verify the alias system works:
```bash
cd /data/home/melashri/analyses/bu2lambdapKK/analysis
python tests/test_aliases.py
```

## Technical Details

### BranchConfig Methods

```python
# Resolve common names to file-specific names
data_branches = config.resolve_aliases(['h1_ProbNNk'], is_mc=False)
# Returns: ['h1_MC15TuneV1_ProbNNk']

mc_branches = config.resolve_aliases(['h1_ProbNNk'], is_mc=True)
# Returns: ['h1_MC12TuneV4_ProbNNk']

# Normalize file-specific names back to common names
rename_map = config.normalize_branches(['h1_MC15TuneV1_ProbNNk'], is_mc=False)
# Returns: {'h1_MC15TuneV1_ProbNNk': 'h1_ProbNNk'}
```

### Disabling Aliases

If we need to disable alias resolution:
```python
data = loader.load_data(..., use_aliases=False)
# Will load exact branch names from config without resolution
```
