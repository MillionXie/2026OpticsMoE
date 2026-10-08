# Third-party assets and parameters

OpenMoji 17.0.0 official 72px color icons: https://github.com/hfg-gmuend/openmoji/releases/tag/17.0.0 . Artwork by OpenMoji contributors, licensed under CC BY-SA 4.0: https://creativecommons.org/licenses/by-sa/4.0/ . The included scenes arrange 16 of these icons on a grid. Keep attribution and applicable share-alike terms when redistributing derived imagery. See assets/asset_manifest.json for exact codepoints.

Qwen3-VL-2B-Instruct: https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct . The upstream model card identifies Apache-2.0; the standard license text is included as APACHE-2.0.txt. The package contains derived frozen patch/position parameters and raw token embedding outputs for the provided instruction corpus, not the complete Qwen model. Preserve attribution to the Qwen team; do not misrepresent the parameters as independently trained from scratch. Modification here: extracting the frozen input layers and caching selected word-embedding outputs, then training our optical task network.

This is a research collaboration artifact, not a blanket relicensing of the repository or third-party materials. Python packages retain their own licenses. Consult upstream terms before public or commercial distribution.
