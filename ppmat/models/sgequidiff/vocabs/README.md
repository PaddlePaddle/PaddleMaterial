# SGEquiDiff Model Static Resources (vocabs)

This directory holds the static resource files required by the SGEquiDiff model.
All files are **reused verbatim from the upstream SGEquiDiff repository**
(copied as-is, unmodified and unformatted, byte-identical to the original).

## File Inventory

| File | Purpose | Upstream location |
|------|---------|-------------------|
| `cgcnn_atom_init.json` | Element embeddings: atomic numbers 0-100 mapped to 92-dim sparse vectors (one-hot-style element fingerprints, each element activates ~9 bits); used as the input representation of the `element_emb` / `ele_emb` linear layers | `data/init_tokens/cgcnn_atom_init.json` |
| `init_tokens/space_group_features/space_group_embeddings_62dim.json` | Space-group embeddings: space group numbers 1-230 mapped to 62-dim vectors; consumed by `SpaceGroupEncoder` to encode space-group indices | `data/init_tokens/space_group_features/space_group_embeddings_62dim.json` |
| `init_tokens/wyckoff_features/wyckoff_embeddings_231dim.json` | Wyckoff embeddings: each Wyckoff letter of every space group mapped to a 231-dim vector; used by the Wyckoff sampler | `data/init_tokens/wyckoff_features/wyckoff_embeddings_231dim.json` |
| `wyckoff_positions/clean_wyckoffs_in_asu_v6.json` | Wyckoff-site geometry inside the asymmetric unit: vertices / dimensionality / hall_number for all 230 space groups; drives the ASU constraints and Wyckoff-shape sampling of the diffusion model | `data/wyckoff_positions/clean_wyckoffs_in_asu_v6.json` |
