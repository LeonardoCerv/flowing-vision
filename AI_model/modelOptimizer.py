import openvino as ov
import os
import sys

def main():
    onnx_path = "OriginalModel.onnx"
    xml_path = "OpenVinoModel.xml"
    bin_path = "OpenVinoModel.bin"
    
    # Check if input file exists
    if not os.path.exists(onnx_path):
        print(f"Error: {onnx_path} not found! run modelConversion.py first.")
        sys.exit(1)
    
    try:
        ov_model = ov.convert_model(onnx_path)
        ov.save_model(ov_model, xml_path)
        
        print("Model exported successfully to:")
        print(f"XML: {xml_path}")
        print(f"BIN: {bin_path}")
        
    except Exception as e:
        print(f"Error during model optimization: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()