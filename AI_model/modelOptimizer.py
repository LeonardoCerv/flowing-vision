import openvino as ov

onnx_path = "OriginalModel.onnx"
ov_model = ov.convert_model(onnx_path)

xml_path = "OpenVinoModel.xml"
bin_path = "OpenVinoModel.bin"
ov.save_model(ov_model, xml_path)

print("Model exported succesfully to:")
print(f"XML: {xml_path}")
print(f"BIN: {bin_path}")