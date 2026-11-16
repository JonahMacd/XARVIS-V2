# legacy_xarvis: the hackathon-era code

This folder holds XARVIS as it was laid out during the NASA Space Apps Challenge 2026, kept for reference.
**Nothing in the rest of this repository uses it.** The current code is in `website/` and `xarvis_pipeline/` at the
repository root.

The original repository, with its full history, is:

**https://github.com/nfortier7972/Xarvis.git**

## What is here

| Path | What it was |
|---|---|
| `Jonah_Code_File/` | The analysis pipeline as on the original `main` branch. Now `xarvis_pipeline/` |
| `Test_Robert_Code_Folder/` | The website as on the original `main` branch, before the pipeline was connected. Now `website/` |
| `Test_Jonah_code.py` | Two-line launcher for the desktop pipeline. Now `python -m xarvis_pipeline` |
| `config.json`, `requirements.txt` | The settings and dependency list that went with the code above |
| `test_config.json`, `test_config_from_delivery.json` | An early settings file that no code reads (two versions) |
| `results/433_Eros.ecsv` | One saved pipeline run, committed during the hackathon |
| `XARVIS_integration.patch` | The diff that connected the pipeline to the website |
| `XARVIS_delivery_downloaded.txt` | A file listing left over from that merge |
| `ORIGINAL_README.md` | The original repository's README |

## If you need the old layout running

Use the original repository rather than this folder:

```
git clone https://github.com/nfortier7972/Xarvis.git
```
