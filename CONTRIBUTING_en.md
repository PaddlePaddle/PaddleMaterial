# Contributing to PaddleMaterials

Thank you for your interest in contributing to PaddleMaterials!

This guide applies to the online repository [PaddlePaddle/PaddleMaterials](https://github.com/PaddlePaddle/PaddleMaterials). PaddleMaterials is an AI4Materials toolkit built on PaddlePaddle. Its core Python package is `ppmat`, and pretrained model weights commonly use the `.pdparams` suffix.

## 1. Development Principles

Code submitted to PaddleMaterials should do more than run successfully. It should also meet the following project expectations:

- Follow the existing task split and repository structure.
- Match the existing code logic, configuration style, and invocation patterns.
- Allow the main training, evaluation, inference, or sampling workflow to run by following the README.
- Keep documentation, configuration files, implementation, and reproduced results consistent.
- Prioritize maintainability, reproducibility, and extensibility.
- Avoid breaking existing models, datasets, and task entrypoints.

For model reproduction PRs, maintainers will pay particular attention to:

- Whether the contribution is placed under the correct task directory.
- Whether it reuses the existing PaddleMaterials data flow, configuration flow, and entry scripts.
- Whether it reuses existing `ppmat` abstractions such as models, datasets, trainers, metrics, schedulers, predictors, and samplers.
- Whether the README, configuration files, tests, or minimum verification instructions are complete.
- Whether the implementation follows the style of existing modules instead of directly copying code from an external repository.

## 2. Repository Structure

Please place code according to the existing repository structure. Do not introduce a separate parallel system unless maintainers have agreed to it.

Current task directories include:

- `property_prediction/`: property prediction tasks (PP).
- `structure_generation/`: structure generation tasks (SG).
- `interatomic_potentials/`: machine learning interatomic potential tasks (MLIP).
- `electronic_structure/`: electronic structure tasks (ES).
- `spectrum_elucidation/`: spectrum elucidation tasks (SE).
- `spectrum_enhancement/`: spectrum or microscopy image enhancement tasks (SPEN).
- `ppmatSim/`: material simulation workflows based on trained models or interatomic potentials.
- `research/`: research-oriented or experimental workflows.
- `jointContribution/`: joint contribution or external collaboration code. This directory is currently excluded from pre-commit checks. New mainline functionality should not be placed here by default unless maintainers explicitly request it.

Shared implementation should generally live in:

- `ppmat/models/`: model definitions, graph converters, and model builders.
- `ppmat/datasets/`: datasets, data builders, collate functions, transforms, and dataloader builders.
- `ppmat/trainer/`: training logic and trainer state.
- `ppmat/losses/`, `ppmat/metrics/`, `ppmat/optimizer/`, `ppmat/schedulers/`: losses, metrics, optimizers, and schedulers.
- `ppmat/predictor/`, `ppmat/sampler/`: shared inference and sampling wrappers.
- `ppmat/calculator/`: external interfaces such as ASE integration.
- `ppmat/utils/`: common utilities.
- `test/`: project tests or validation scripts.
- `docs/`: project documentation.

## 3. Adding a New Model

Adding a new model usually involves some of the following locations:

- `ppmat/models/<model_name>/`: the model implementation, graph converter, or model-specific utilities.
- `ppmat/models/__init__.py`: import the model class, update `__all__`, and add `MODEL_REGISTRY` entries if pretrained weights are provided.
- `configs/<model_name>/` under the corresponding task directory: model configurations and README.
- `train.py`, `predict.py`, `sample.py`, or related entrypoints under the corresponding task directory when adaptation is necessary.
- `test/`: tests or minimum verification scripts when needed.

Do not put the full model implementation directly into task scripts. Do not bypass existing `ppmat` abstractions by creating a separate training framework.

When adding a model:

- Use PaddlePaddle APIs. Model classes usually inherit from `paddle.nn.Layer`.
- Continue using the `__class_name__` and `__init_params__` convention in configuration files.
- Reuse existing `Trainer`, `Dataset`, `Optimizer`, `Metric`, `Predict`, and `Sample` configuration structures whenever possible.
- Follow existing pretrained weight naming conventions. Common names include `best.pdparams` and `latest.pdparams`.
- The repository already contains shared `ppmat/predictor/` and `ppmat/sampler/` wrappers. Prefer reusing or extending these shared implementations. If task-specific inference or sampling code is necessary, keep the task entrypoint lightweight and consistent with nearby examples.
- If you modify shared predictor or sampler logic, verify that related existing models still work.

Code from other frameworks or repositories must not be copied into PaddleMaterials without adaptation. It must be converted to the PaddleMaterials style, including:

- PaddlePaddle API usage.
- Consistent parameter organization.
- Consistent configuration naming.
- Consistent training, inference, and sampling interfaces.
- Consistent documentation style.
- Clear dependency and license information.

## 4. Adding a New Dataset

New datasets should generally be added under `ppmat/datasets/`. If the processing logic is complex, create a subdirectory under `ppmat/datasets/` to organize it.

Dataset implementations should follow the style of existing datasets:

- Keep initialization parameters consistent with existing datasets.
- Make train, validation, and test split logic clear.
- Use clear field names.
- Return a stable sample structure.
- Stay compatible with existing dataloaders, collate functions, and transforms.
- Match the fields expected by downstream models.
- Avoid excessive task-specific branching inside the dataset class.
- Avoid tightly coupling preprocessing, training logic, and dataset logic.

Dataset integration usually requires:

- Adding the dataset class under `ppmat/datasets/`.
- Importing the dataset class in `ppmat/datasets/__init__.py` so that `build_dataloader` can find it through `__class_name__` in the config.
- Adding dataset info, transforms, collate logic, or graph building logic when necessary.
- Reusing existing factory functions, utility classes, and data processing logic whenever possible.

Dataset PRs should use existing PaddleMaterials datasets as references instead of directly copying the style of an external repository.

The dataset README should describe at least:

- Dataset name.
- Source link and license restrictions.
- Original format.
- Download method.
- Preprocessing steps.
- Split strategy.
- Label meanings, units, and field names.
- How paths and cache paths in the configuration correspond to the prepared files.

At minimum, a dataset PR should show that:

- The dataset class can be instantiated.
- Data can be read successfully.
- Sample fields match downstream model requirements.
- The basic workflow runs by following the README.
- Documentation and code are consistent.

## 5. Utility and Conversion Scripts

Scripts under `test/`, `tools/`, or task directories should be limited to:

- Official tools that users are expected to invoke.
- Long-term-maintained scripts for weight conversion, data conversion, evaluation, or result alignment.
- Migration tools with README instructions or command examples.

Do not submit:

- Temporary debugging scripts.
- Scripts that only work in the author's local environment.
- Experimental scripts with hard-coded personal paths, private machine names, or credentials.
- One-off alignment scripts without documentation.
- Large logs, caches, training outputs, or unexplained intermediate files.

## 6. PaddleMaterials-Specific Requirements

### 6.1 Reuse Existing Entrypoints and Organization

New functionality should be integrated into existing workflows instead of introducing a new standalone system. This includes:

- Training entrypoints.
- Inference entrypoints.
- Sampling entrypoints.
- Configuration organization.
- Dataset registration and invocation.
- Model invocation interfaces.
- Weight download and loading logic.

If a similar model or task already exists, align with the implementation in the corresponding task directory.

For example:

- For crystal or molecular property prediction, refer to model and config organization under `property_prediction/`.
- For machine learning interatomic potentials, refer to model and README organization under `interatomic_potentials/`.
- For structure generation, refer to MatterGen and DiffCSP configurations and sampling entrypoints under `structure_generation/`.
- For spectrum elucidation, refer to DiffNMR training and sampling workflows under `spectrum_elucidation/`.
- For spectrum or microscopy image enhancement, refer to SFIN configurations and prediction entrypoints under `spectrum_enhancement/`.
- For dataset adaptation, refer to existing implementations under `ppmat/datasets/`.

### 6.2 Do Not Force External Repository Style Into PaddleMaterials

Code from other frameworks or repositories cannot be submitted as a direct copy.

It must be adapted to the PaddleMaterials style, including:

- PaddlePaddle API adaptation. Mainline functionality should not depend on another deep learning framework.
- Consistent parameter organization.
- Consistent configuration naming.
- Consistent training, inference, and sampling interfaces.
- Consistent documentation style.
- Clear license, citation, and third-party code boundary information.

### 6.3 Use Official Paddle Releases

- Use officially released PaddlePaddle versions.
- Do not make the Paddle develop branch the only supported runtime.
- If a minimum PaddlePaddle version is required, state it clearly in the README.
- If specific CUDA, Python, `paddle_scatter`, `pgl`, `pymatgen`, `ase`, or other dependency versions are required, document them as well.

For PR acceptance, functionality that only runs on the Paddle develop branch or a private environment is generally not considered ready to merge.

## 7. README and Documentation Requirements

Contributions to PaddleMaterials, especially model reproduction and dataset adaptation PRs, must include usable documentation.

The README should include at least:

- Task introduction.
- Model or dataset introduction.
- Data preparation instructions.
- Environment dependencies and verified versions.
- Training command.
- Evaluation command.
- Inference or sampling command, if applicable.
- Key configuration explanation.
- Reference results.
- Pretrained weight or dataset download links, if applicable.
- Reference paper, official implementation, or data source links.

The README must satisfy the following:

- Following the README should allow users to complete the main workflow.
- Documentation and code must be consistent.
- Commands should be copyable from the repository root, with no hidden prerequisites.
- Key steps must not be omitted.
- The README must not describe one workflow while the code implements another.

For model reproduction PRs, we recommend documenting:

- Reproduction target.
- Dataset version.
- Key hyperparameters.
- Number of epochs, batch size, and learning rate.
- Hardware environment.
- Evaluation metrics.
- Comparison with the original paper or reference implementation.
- Deviation range and possible reasons, if any.

If the reproduced result does not fully match the original paper yet, clearly state the current status instead of vaguely saying that the model is supported.

## 8. Configuration File Requirements

Configuration files should follow the existing PaddleMaterials YAML style.

Common top-level fields include:

- `Global`
- `Trainer`
- `Model`
- `Optimizer`
- `Metric`
- `Dataset`
- `Predict`
- `Sample` or `Sampler`
- `Loss`, when required by the task

Configurations should be clear and complete. They should generally include:

- Model construction parameters.
- Dataset paths, fields, transforms, graph converters, or preprocessing parameters.
- Dataloader, sampler, batch size, and number of workers.
- Optimizer and learning rate scheduler.
- Loss and metrics.
- Trainer, predictor, or sampler settings.
- Save, log, and evaluation settings.
- Pretrained model or checkpoint settings, if applicable.

Default configurations should be runnable as much as possible. Submitted configurations do not all need to be large-scale training configs, but they should at least:

- Parse successfully.
- Build the model.
- Build the dataset, or clearly describe how to download and prepare the data.
- Start a minimal training, evaluation, inference, or sampling workflow.

Configuration names should reflect the task, model name, dataset name, and training target. For example:

- `megnet_mp2018_train_60k_e_form.yaml`
- `chgnet_oc20_s2ef_energy.yaml`
- `dimenet++_mp2018_train_60k_band_gap.yaml`
- `diffcsp_mp20.yaml`
- `mattergen_alex_mp20_dft_band_gap.yaml`
- `infgcn_qm9.yaml`
- `sfin_haadf_enhance.yaml`

Avoid unclear or temporary names such as:

- `final_new.yaml`
- `test2.yaml`
- `run_ok.yaml`
- `debug.yaml`

## 9. Dataset Files and Pretrained Weights

Large datasets, pretrained weights, cache files, and training outputs should not be committed directly to git.

Dataset or model adaptation PRs should provide the files required for review, such as dataset files or pretrained model weights. You may upload them to cloud storage first and provide the link in the PR, or follow the project convention and ask a reviewer to help upload them to BCE.

After maintainers or reviewers upload dataset files or pretrained weights to BCE, contributors should update the official links in the corresponding locations, such as:

- Download instructions in the README.
- Default paths or notes in configuration files.
- `MODEL_REGISTRY` in `ppmat/models/__init__.py`, if the file is a pretrained model.

Please also provide:

- File name and size.
- Directory structure after extraction.
- Checksum, recommended when available.
- Data or weight source.
- License restrictions.

## 10. Testing and Acceptance Requirements

Before a PR is merged, the core functionality should satisfy the minimum verification requirements:

- Code imports successfully.
- Key configurations parse successfully.
- The model can complete a minimal forward pass.
- The dataset can be loaded.
- The main training, evaluation, inference, or sampling workflow can run.
- Existing functionality is not obviously broken.

We recommend adding one or more of the following tests for new functionality:

- Dataset smoke test.
- Model forward test.
- Config load test.
- Train, evaluation, prediction, or sampling smoke test.
- Unit tests for key modules.
- Regression tests for shared builders, transforms, collate functions, metrics, or schedulers.

Model reproduction PRs do not need to be fully productionized in one step, but they should at least:

- Have a clear reproduction target.
- Provide basic result evidence.
- Include a runnable README.
- Keep configurations and code aligned.
- Make the main metrics reproducible or explainable.

Code migration alone, without a basic runnable workflow and explanation, is not recommended for merge.

## 11. Code Style and Formatting

This repository uses pre-commit for basic formatting checks. Current checks include isort, black, ruff, Markdown CRLF/tab checks, YAML/JSON checks, and clang-format for C/CUDA files.

Before submitting, we recommend running:

```bash
pre-commit run --all-files
```

If you add or modify tests, also run the corresponding test command, for example:

```bash
python -m pytest test
```

If you cannot run the full test suite, explain in the PR description what was not run, why it was not run, and what alternative verification was performed.

Basic formatting requirements:

- Python code should follow the existing repository style.
- YAML and JSON files must be valid.
- Markdown files should not contain CRLF line endings or tab characters.
- C/CUDA/C++/proto files should pass clang-format.
- Do not keep unrelated debug output.
- Do not commit conflict markers, private keys, tokens, or unrelated temporary files.
- Do not commit `__pycache__`, `.pytest_cache`, training outputs, large logs, or local caches.

The following content is generally not recommended for direct merge into the PaddleMaterials main repository:

- Code unrelated to the main task directions of the repository.
- Model migrations without README, explanation, or minimum verification.
- Code containing personal absolute paths.
- Scripts that strongly depend on private environments.
- Temporary experiment files.
- Large weights or datasets without source explanation.
- Implementations where documentation and code are clearly inconsistent.
- Features that only run on the Paddle develop branch.
- Implementations that bypass the existing project structure and introduce a separate training framework.

## 12. Pre-Submission Checklist

Before submitting a PR, please confirm that:

- The change belongs to an existing PaddleMaterials task direction, or has been discussed with maintainers.
- Files are placed in the correct directories.
- Existing code organization, configuration style, and invocation entrypoints are reused.
- An officially released PaddlePaddle version is used.
- New dependencies are reasonable, and version or environment requirements are documented.
- The README has been updated and is usable.
- The main training, evaluation, inference, or sampling workflow is clearly described.
- Configuration files can be parsed successfully.
- The minimum dataset or model workflow has been verified.
- Formatting checks have been run, or the reason for not running them is explained.
- No unrelated temporary files, large logs, or cache files are committed.
- No large weight or data files are committed without explanation.
- The PR description is complete, including motivation, changes, verification, and known limitations.
- If the contribution belongs to a specific campaign or activity, update the corresponding issue or activity page as required, for example `#194`.

## 13. Review Focus

During review, maintainers usually focus on:

- Whether the contribution fits the current PaddleMaterials task directions.
- Whether it follows the existing repository structure and logic.
- Whether it reuses shared abstractions instead of bypassing the main workflow.
- Whether the README supports complete usage.
- Whether documentation, configuration, and code are consistent.
- Whether minimum reproducibility is provided.
- Whether the implementation is maintainable long term.
- Whether existing functionality is affected.
- Whether sources of data, weights, and external code are clear and compliant.

Review is not meant to raise unnecessary barriers. Its purpose is to preserve the quality of PaddleMaterials as a unified AI4Materials toolkit.

## 14. Acknowledgements

Thank you to every contributor who supports PaddleMaterials. We look forward to improving PaddleMaterials together across models, datasets, training and evaluation, inference and sampling, and material simulation workflows.
