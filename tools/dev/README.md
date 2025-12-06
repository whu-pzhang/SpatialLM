# Development Tools

This directory contains tools for developing and modifying the SpatialLM model and its components.

## create_grid_model.py

This script modifies a tokenizer (and optionally the model) to support a special "Grid Token" representation for coordinates. It replaces the last $N$ tokens of the vocabulary with special tokens in the format `<integer>`, allowing the model to predict discrete coordinates directly.

### Usage

```bash
python tools/dev/create_grid_model.py \
    --model_path manycore-research/SpatialLM1.1-Qwen-0.5B \
    --output_path SpatialLM1.2-Qwen-0.5B-GridSpecialTokens \
    --grid_size 1600 \
    --copy_weights
```

### Arguments

- `--model_path`: Path or name of the original model/tokenizer. Default: `manycore-research/SpatialLM1.1-Qwen-0.5B`.
- `--output_path`: Directory to save the new tokenizer (and model if weights are copied). Default: `SpatialLM1.2-Qwen-0.5B-GridSpecialTokens`.
- `--grid_size`: Number of special grid tokens to add. These will reuse the IDs of the last `grid_size` tokens in the original vocabulary. Default: `1600`.
- `--copy_weights`: If set, copies the original model weights and configuration to the output directory.

### How it works

1.  **Calculates Start ID**: Determines where the special tokens should start based on `vocab_size - grid_size`.
2.  **Replaces Tokens**: Removes the last `grid_size` tokens from the base vocabulary and replaces them with special tokens `<0>`, `<1>`, ..., `<N-1>`.
3.  **Cleans Merges**: Removes any BPE merge rules that referenced the deleted tokens.
4.  **Updates Config**: Updates `tokenizer.json` and `tokenizer_config.json` to register these new tokens as special tokens.
5.  **Verification**: Runs a self-test to ensure tokens are correctly encoded and decoded.

## Entity Tokenization Support

The `SpatialLM/spatiallm/layout/entity.py` file has been updated to support this new tokenization scheme.

### `to_token_string()`

A new method `to_token_string()` has been added to `Wall`, `Door`, `Window`, and `Bbox` classes. This method converts the entity's attributes (coordinates, dimensions) into the special grid token format.

**Example Comparison:**

*   **Standard String (`to_language_string`)**:
    ```
    wall_0=Wall(10,20,0,100,20,0,50,5)
    ```
*   **Token String (`to_token_string`)**:
    ```
    wall_0=Wall(<10>,<20>,<0>,<100>,<20>,<0>,<50>,<5>)
    ```

This ensures that numerical values are tokenized as single special tokens rather than being split into multiple digit tokens.

## Training Parsing Flow

The training data processing pipeline handles the synchronization between point cloud inputs and layout targets. This logic is implemented in `spatiallm/tuner/data/mm_plugin.py`, specifically in the `process_messages` method.

### `process_messages` Workflow

The `process_messages` function ensures that the layout labels are consistent with the augmented point cloud data. The flow is as follows:

1.  **Input Extraction**: The function parses the input messages to find layout content (`<|layout_s|>...<|layout_e|>`) and point cloud placeholders.
2.  **Transformation Synchronization**: It retrieves the data augmentation parameters applied to the point cloud (random rotation, scaling, translation).
3.  **Layout Transformation**: The original layout is transformed to match the point cloud:
    *   **Centering**: Translated by `-center_pt`.
    *   **Scaling**: Scaled by the augmentation factor.
    *   **Rotation**: Rotated by `angle_z`.
    *   **Restoration**: Translated back by `center_pt`.
    *   **Normalization**: Translated by `-min_bound` to align with the point cloud's local coordinate system.
4.  **Discretization**: The continuous coordinates are discretized into grid bins (default 1280 bins) using `normalize_and_discretize`.
5.  **Tokenization**: The transformed and discretized layout is converted to the grid token format using `layout.to_token_string()`.
6.  **Content Replacement**: The original human-readable layout string in the message is replaced with the new token-based string.

This process ensures that the model learns to predict layouts that are perfectly aligned with the visual point cloud input, using the efficient grid token vocabulary.
