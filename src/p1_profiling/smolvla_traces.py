import torch
from torch.profiler import profile, record_function, ProfilerActivity
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def generate_dummy_observations(model, device):
    batch_size = 1

    # Match the checkpoint configuration.
    state_dim = model.config.input_features["observation.state"].shape[0]
    tokenizer_max_length = model.config.tokenizer_max_length

    # Three RGB cameras: (B, C, H, W).
    obs = {
        "observation.images.camera1": torch.rand(
            batch_size, 3, 256, 256, device=device
        ),
        "observation.images.camera2": torch.rand(
            batch_size, 3, 256, 256, device=device
        ),
        "observation.images.camera3": torch.rand(
            batch_size, 3, 256, 256, device=device
        ),
        "observation.state": torch.zeros(batch_size, state_dim, device=device),
    }

    # Tokenize an actual text prompt using SmolVLA's tokenizer.
    tokenizer = model.model.vlm_with_expert.processor.tokenizer

    encoded = tokenizer(
        ["pick up the object"],
        padding="max_length",
        truncation=True,
        max_length=tokenizer_max_length,
        return_tensors="pt",
    )

    obs["observation.language.tokens"] = encoded["input_ids"].to(device)
    obs["observation.language.attention_mask"] = (
        encoded["attention_mask"].to(device).bool()
    )

    return obs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Targeting device: {device}")

    local_model_path = "/workspace/3-MDP/smolvla_base"
    print(f"Loading SmolVLA from: {local_model_path}")

    model = SmolVLAPolicy.from_pretrained(local_model_path).to(device)
    model.eval()

    obs = generate_dummy_observations(model, device)

    # Validate input keys before inference.
    print("\nInput tensor shapes:")
    for key, value in obs.items():
        print(f"  {key}: {tuple(value.shape)}")

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    print("\nRunning warmup iterations...")

    # Use predict_action_chunk to force a full inference pass each time.
    with torch.inference_mode():
        for _ in range(5):
            _ = model.predict_action_chunk(obs)

    if device.type == "cuda":
        torch.cuda.synchronize()

    print("Starting hardware profiling...")

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    ) as prof:
        with record_function("smolvla_full_action_chunk_inference"):
            with torch.inference_mode():
                actions = model.predict_action_chunk(obs)

        if device.type == "cuda":
            torch.cuda.synchronize()

    trace_filename = "smolvla_trace.json"
    prof.export_chrome_trace(trace_filename)

    print(f"\nTrace exported to {trace_filename}")
    print(f"Output action chunk shape: {tuple(actions.shape)}")

    sort_key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"

    print("\n--- Top 20 Operations ---")
    print(
        prof.key_averages().table(
            sort_by=sort_key,
            row_limit=20,
        )
    )


if __name__ == "__main__":
    main()
