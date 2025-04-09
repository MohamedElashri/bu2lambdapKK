import uproot
from pathlib import Path
from tqdm import tqdm
import logging
import traceback

# ========== CONFIG ==========
input_base = Path("/share/lazy/Mohamed/bu2kskpik/RD/reduced")
output_base = Path("/share/lazy/Mohamed/bu2kskpik/RD/compressed")
compression = uproot.ZSTD(15)  # we can switch to ZLIB(9), LZMA(9), etc.
log_path = Path("compress_root.log")
# ============================

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)

# Find all ROOT files
root_files = sorted(list(input_base.rglob("*.root")))
logging.info(f"Discovered {len(root_files)} ROOT files to process.")

# Progress bar over files
for root_path in tqdm(root_files, desc="Compressing ROOT files", unit="file"):
    rel_path = root_path.relative_to(input_base)
    output_path = output_base / rel_path

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"Processing {root_path} → {output_path}")

        # Open and list trees
        f_in = uproot.open(root_path)
        tree_names = [k.split(";")[0] for k in f_in.keys()]

        with uproot.recreate(output_path, compression=compression) as f_out:
            for name in tree_names:
                try:
                    arrays = f_in[name].arrays(library="np")
                    f_out[name] = arrays
                    logging.info(f"  ✅ Wrote tree: {name} ({len(arrays)} branches)")
                except Exception as e_tree:
                    logging.warning(f"  ⚠️ Skipped tree '{name}' in '{root_path.name}': {e_tree}")
                    logging.debug(traceback.format_exc())

    except Exception as e_file:
        logging.error(f"❌ Failed to process file: {root_path}: {e_file}")
        logging.debug(traceback.format_exc())
