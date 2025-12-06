import argparse
import json
import shutil
from pathlib import Path

from transformers import AutoTokenizer

from spatiallm import Layout  # noqa


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Modify tokenizer to replace the last N tokens with special grid tokens."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="manycore-research/SpatialLM1.1-Qwen-0.5B",
        help="Path or name of the original model/tokenizer",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="SpatialLM1.2-Qwen-0.5B-GridSpecialTokens",
        help="Directory to save the new tokenizer",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=1600,
        help="Number of special grid tokens to add (reusing last N IDs)",
    )
    parser.add_argument(
        "--copy_weights",
        action="store_true",
        help=
        "Copy original model weights and configuration to the output directory",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    GRID_SIZE = args.grid_size
    ori_model_name_or_path = args.model_path
    new_model_path = Path(args.output_path)

    print(f"Original Model: {ori_model_name_or_path}")
    print(f"Output Path: {new_model_path}")
    print(f"Grid Size: {GRID_SIZE}")

    new_model_path.mkdir(exist_ok=True, parents=True)

    if args.copy_weights:
        print(
            f"Loading and saving model weights from {ori_model_name_or_path}..."
        )
        try:
            from transformers import AutoModelForCausalLM
            # Use AutoModelForCausalLM to download/load and save weights
            model = AutoModelForCausalLM.from_pretrained(
                ori_model_name_or_path,
                trust_remote_code=True,
                device_map=
                "cpu"  # Load to CPU to save memory if GPU is not needed
            )
            model.save_pretrained(new_model_path)
            print("✅ Model weights saved successfully.")
        except Exception as e:
            print(f"❌ Failed to copy model weights: {e}")
            print("Continuing with tokenizer modification only...")

    tokenizer = AutoTokenizer.from_pretrained(ori_model_name_or_path)
    tokenizer.save_pretrained(new_model_path)

    # 1. Calculate start_id based on vocab_size to replace the last 1600 normal tokens
    # We want to replace the last GRID_SIZE tokens of the base vocabulary.
    # Base vocab indices are [0, tokenizer.vocab_size - 1].
    # So we start at tokenizer.vocab_size - GRID_SIZE.
    start_id = tokenizer.vocab_size - GRID_SIZE
    print(f"Base vocab size: {tokenizer.vocab_size}")
    print(
        f"Replacing tokens in range: [{start_id}, {tokenizer.vocab_size - 1}]")

    special_tokens = [f"<{i}>" for i in range(0, GRID_SIZE)]
    token_ids = list(range(start_id, start_id + len(special_tokens)))

    added_tokens = []
    for tok, idx in zip(special_tokens, token_ids):
        added_tokens.append(
            {
                "id": idx,
                "content": tok,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }, )

    with open(new_model_path.joinpath("tokenizer.json"), "r") as f:
        tokenizer = json.load(f)

    # Remove conflicting IDs from base vocab to ensure added_tokens take precedence
    vocab = tokenizer["model"]["vocab"]
    keys_to_remove = [
        k for k, v in vocab.items() if start_id <= v < start_id + GRID_SIZE
    ]
    print(
        f"Removing {len(keys_to_remove)} tokens from base vocab to make space for special tokens."
    )
    for k in keys_to_remove:
        del vocab[k]
    tokenizer["model"]["vocab"] = vocab

    # Also filter merges to remove rules referencing removed tokens
    if "merges" in tokenizer["model"]:
        print("Filtering merges to remove rules using deleted tokens...")
        merges = tokenizer["model"]["merges"]
        removed_set = set(keys_to_remove)
        new_merges = []
        skipped_count = 0

        # Check merge format
        is_list_format = False
        if merges and isinstance(merges[0], list):
            is_list_format = True

        print(f"Is list format: {is_list_format}")

        for merge in merges:
            # merge is either "a b" or ["a", "b"]
            parts = merge
            if not is_list_format:
                parts = merge.split(" ")

            if len(parts) != 2:
                # Should be 2 parts usually
                pass

            p1, p2 = parts[0], parts[1]
            # Check if components are removed
            if p1 in removed_set or p2 in removed_set:
                skipped_count += 1
                continue

            # Check if result is removed (merge result must be in vocab)
            # BPE merge A B -> AB. AB must be in vocab.
            merged_token = p1 + p2
            if merged_token in removed_set:
                skipped_count += 1
                continue

            new_merges.append(merge)

        print(f"Removed {skipped_count} merge rules.")
        tokenizer["model"]["merges"] = new_merges

    ori_added_tokens = tokenizer["added_tokens"]
    tokenizer["added_tokens"] = added_tokens + ori_added_tokens

    with open(new_model_path.joinpath("tokenizer.json"), "w") as f:
        json.dump(tokenizer, f, indent=4, ensure_ascii=False)

    # 2.modify tokenizer_config.json
    with open(new_model_path.joinpath("tokenizer_config.json"), "r") as f:
        tokenizer_config = json.load(f)

    ori_added_tokens_decoder = tokenizer_config["added_tokens_decoder"]
    for idx, tok in zip(token_ids, special_tokens):
        ori_added_tokens_decoder[str(idx)] = {  # Ensure key is string
            "content": tok,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True,
        }

    ori_additional_special_tokens = tokenizer_config[
        "additional_special_tokens"]
    tokenizer_config["additional_special_tokens"].extend(special_tokens)
    tokenizer_config["added_tokens_decoder"] = ori_added_tokens_decoder

    with open(new_model_path.joinpath("tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 20 + " Verification " + "=" * 20)
    try:
        print(f"Loading new tokenizer from {new_model_path}...")
        new_tokenizer = AutoTokenizer.from_pretrained(new_model_path,
                                                      trust_remote_code=True)

        print(f"New tokenizer vocab size: {new_tokenizer.vocab_size}")
        print(f"Len of new tokenizer: {len(new_tokenizer)}")

        # Test specific special tokens
        test_indices = [0, GRID_SIZE // 2, GRID_SIZE - 1]
        test_tokens = [f"<{i}>" for i in test_indices]
        test_text = f"Start {' '.join(test_tokens)} End"
        print(f"\nTest input text: '{test_text}'")

        encoded = new_tokenizer(test_text)
        input_ids = encoded["input_ids"]
        print(f"Encoded IDs: {input_ids}")

        decoded = new_tokenizer.decode(input_ids, skip_special_tokens=False)
        print(f"Decoded text: '{decoded}'")

        # Verify specific tokens
        all_passed = True
        for tok in test_tokens:
            if tok not in new_tokenizer.get_vocab():
                # It might be in added_tokens but not vocab, so check via convert_tokens_to_ids
                tid = new_tokenizer.convert_tokens_to_ids(tok)
                if tid == new_tokenizer.unk_token_id and tok != new_tokenizer.unk_token:
                    print(f"❌ Token {tok} maps to UNK!")
                    all_passed = False
                else:
                    print(f"✅ Token {tok} ID: {tid}")
            else:
                print(
                    f"✅ Token {tok} ID: {new_tokenizer.convert_tokens_to_ids(tok)}"
                )

        # Check if the tokens are split or kept as one
        tokens = new_tokenizer.tokenize(test_text)
        print(f"Tokens: {tokens}")
        for tok in test_tokens:
            if tok not in tokens:
                print(
                    f"❌ Token {tok} was not preserved in tokenization! (Got split?)"
                )
                all_passed = False

        # Verify original special tokens are preserved
        print("\nVerifying original special tokens...")
        original_special = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]
        for ost in original_special:
            if ost in new_tokenizer.get_vocab(
            ) or new_tokenizer.convert_tokens_to_ids(
                    ost) != new_tokenizer.unk_token_id:
                oid = new_tokenizer.convert_tokens_to_ids(ost)
                print(f"✅ Original special token {ost} preserved at ID {oid}")
            else:
                print(f"❌ Original special token {ost} MISSING or UNK!")
                all_passed = False

        if all_passed:
            print(
                "\n✅ Verification PASSED: Special tokens are correctly added and tokenized."
            )
        else:
            print("\n❌ Verification FAILED: Some checks did not pass.")

    except Exception as e:
        print(f"\n❌ Verification failed with exception: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
