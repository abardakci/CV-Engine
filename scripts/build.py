import tensorrt as trt
import argparse

def build_static(onnx_path, engine_path, fp16=True):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        parser.parse(f.read())

    config = builder.create_builder_config()
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    serialized_engine = builder.build_serialized_network(network, config)
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

def build_dynamic(onnx_path, engine_path, fp16=True):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        parser.parse(f.read())

    config = builder.create_builder_config()
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Otomatik input profili
    profile = builder.create_optimization_profile()
    profile.set_shape("input",
                    min=(1, 3, 540, 960),
                    opt=(1, 3, 540, 960),
                    max=(1, 3, 540, 960))
    
    config.add_optimization_profile(profile)
    serialized_engine = builder.build_serialized_network(network, config)
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

# Kullanım örneği
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    args = parser.parse_args()

    print(trt.__version__)
    name = args.model
    onnx_name = name + ".onnx"
    plan_name = name + ".plan"
    build_dynamic(onnx_name, plan_name)
