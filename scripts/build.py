import tensorrt as trt
import os

def build_trt(onnx_file_path, engine_output_path):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    
    # EXPLICIT_BATCH kullanımı
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse ONNX file")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return None

    # Yapılandırma
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)  # FP16 modunu etkinleştir
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB workspace

    # Statik input olduğu için profile eklemiyoruz

    # Engine'i oluştur ve kaydet
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("ERROR: Failed to build the engine.")
        return None

    with open(engine_output_path, 'wb') as f:
        f.write(serialized_engine)

    print(f"Engine saved to {engine_output_path}")
    return serialized_engine

# Kullanım örneği
if __name__ == "__main__":
    print(trt.__version__)
    # name = "yolov8n"
    # onnx_name = name + ".onnx"
    # plan_name = name + ".plan"
    # build_trt(onnx_name, plan_name)
