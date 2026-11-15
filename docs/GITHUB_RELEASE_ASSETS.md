# GitHub Release Assets

Source, reports, manifests, scripts, normal-sized evidence, and reconstruction metadata are published in the Git repository.

Large research files are attached to the GitHub Release assets for tag `r6a-post-v11-2-publication`.

For split files, download every `.part001`, `.part002`, and so on into the same folder. Do not rename parts. Reconstruct the original before opening it, then verify SHA-256 against `docs/github_release_assets_manifest.json`.

Use:

```bat
python tools\reconstruct_release_asset.py --manifest <parts_manifest.json> --output <output_file>
```

This preserves a post-v11.2 snapshot and does not assert a new scientific verdict.
