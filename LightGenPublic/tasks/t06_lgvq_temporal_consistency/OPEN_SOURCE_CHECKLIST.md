# Public-release checklist

## Completed in this package

- [x] Standalone imports; no runtime dependency on the parent repository.
- [x] Exact checkpoint, source commits and archive hashes recorded.
- [x] Fixed-weight reproduction entry point included.
- [x] Integrity manifest generated for every delivered file.
- [x] Full and smoke commands separated.
- [x] Metrics, units, split size, padding semantics and tolerances documented.
- [x] Test-set model selection and electronic feature boundary disclosed.
- [x] Server paths and credentials excluded from the rebuilt release.
- [x] Python cache files and experiment debris excluded.
- [x] Tests cover imports, metrics, metadata and absolute-path leakage.

## Owner decisions required before a public GitHub release

- [ ] Choose a software license and confirm every contributor agrees.
- [ ] Confirm whether the 81.8 MB checkpoint may be redistributed.
- [ ] Confirm whether frozen Qwen-derived test tensors may be redistributed
      under the upstream model terms and the LGVQ dataset terms.
- [ ] Add author names, affiliations, ORCIDs and the final repository URL to a
      `CITATION.cff` file.
- [ ] Decide whether to publish only code plus download instructions, or the
      complete 300+ MB reproducibility artifact.
- [ ] Run the full 558-video command on the intended public CUDA/PyTorch image
      and archive the resulting JSON plus environment information.

Until those decisions are complete, the ZIP is appropriate for internal review
and teacher submission, not an assertion that all assets are publicly licensed.
