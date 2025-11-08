import tensorrt as trt
import argparse


def build(onnx_path, engine_path, fp16=False, verbose=False):
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    print(f"🔧 Parsing ONNX: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("❌ Failed to parse ONNX file:")
            for i in range(parser.num_errors):
                err = parser.get_error(i)
                print(err)
            return

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
    config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)  # Sparsity destekliyse %10–60 hızlanma
    config.builder_optimization_level = 5             # Max optimization (TRT10+)

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("⚡ FP16 mode enabled")

    # ----------------------------
    # Detect dynamic inputs
    # ----------------------------
    num_inputs = network.num_inputs
    print(f"📥 Found {num_inputs} input tensors:")
    for i in range(num_inputs):
        inp = network.get_input(i)
        print(f"  - {inp.name}: shape={inp.shape}, dtype={inp.dtype}")

    # Check if any input is dynamic
    is_dynamic = any(-1 in network.get_input(i).shape for i in range(num_inputs))

    if is_dynamic:
        print("🌀 Dynamic input detected → adding optimization profile.")
        profile = builder.create_optimization_profile()

        for i in range(num_inputs):
            inp = network.get_input(i)
            name = inp.name
            shape = inp.shape

            # varsayılan min/opt/max
            # Eğer H,W belli değilse -> (1, C, 320, 320)
            # Eğer 2D feature ise -> (1, D)
            if len(shape) == 4:  # e.g. NCHW
                c = shape[1] if shape[1] != -1 else 3
                min_shape = (1, c, 360, 640)
                opt_shape = (1, c, 360, 640)
                max_shape = (1, c, 360, 640)
            elif len(shape) == 3:  # (N, seq, feat)
                f = shape[-1] if shape[-1] != -1 else 256
                min_shape = (1, 4, f)
                opt_shape = (1, 16, f)
                max_shape = (4, 64, f)
            elif len(shape) == 2:  # (N, D)
                d = shape[1] if shape[1] != -1 else 256
                min_shape = (1, d)
                opt_shape = (1, d)
                max_shape = (4, d)
            else:
                print(f"⚠️ Skipping unusual input shape for {name}: {shape}")
                continue

            profile.set_shape(name, min_shape, opt_shape, max_shape)
            print(f"  ↳ Profile set for {name}:")
            print(f"    min={min_shape}, opt={opt_shape}, max={max_shape}")

        config.add_optimization_profile(profile)
    else:
        print("📏 All input shapes are static → no profile added.")

    # ----------------------------
    # Build serialized engine
    # ----------------------------
    print("🚀 Building TensorRT engine...")
    serialized_engine = builder.build_serialized_network(network, config)
    if not serialized_engine:
        raise RuntimeError("❌ Engine build failed.")

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print(f"✅ Engine saved to {engine_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input ONNX path")
    parser.add_argument("--output", required=True, help="Output engine path")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 precision")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose TensorRT logs")
    args = parser.parse_args()

    print(f"TensorRT version: {trt.__version__}")
    build(args.input, args.output, fp16=args.fp16, verbose=args.verbose)
