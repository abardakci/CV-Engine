import tensorrt as trt
import argparse

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def inspect_plan(plan_path):
    print(f"🔍 Inspecting TensorRT Engine: {plan_path}")
    with open(plan_path, "rb") as f:
        engine_data = f.read()

    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_data)

    print(f"\nTensorRT version: {trt.__version__}")
    print(f"Engine has {engine.num_io_tensors} IO tensors")
    print(f"Optimization profiles: {engine.num_optimization_profiles}")

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        dtype = engine.get_tensor_dtype(name)
        mode = engine.get_tensor_mode(name)
        print(f"\n[{i}] Tensor: {name}")
        print(f"    Shape: {shape}")
        print(f"    Dtype: {dtype}")
        print(f"    Mode : {'INPUT' if mode == trt.TensorIOMode.INPUT else 'OUTPUT'}")

    # Profil bilgilerini göster
    for p in range(engine.num_optimization_profiles):
        print(f"\nOptimization Profile {p}:")
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                try:
                    min_shape = engine.get_profile_shape(p, name)[0]
                    opt_shape = engine.get_profile_shape(p, name)[1]
                    max_shape = engine.get_profile_shape(p, name)[2]
                    print(f"  {name}: min={min_shape}, opt={opt_shape}, max={max_shape}")
                except Exception:
                    pass

    print("\n✅ Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a TensorRT engine (.plan) file")
    parser.add_argument("--path", type=str, required=True, help="Path to the TensorRT .plan file to inspect")
    args = parser.parse_args()

    inspect_plan(args.path)