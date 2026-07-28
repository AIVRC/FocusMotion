
import cv2
import os
import sys

# Add current directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from extract_clothes import ComplexAreaIdentifier, CONFIG, Params

def test_extract():
    # Setup paths
    img_path = "/home/yanghaotian/server_data/yanghaotian/data/applied_dataset/ref/00009_0001.png"
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "test_result.png")

    # Initialize identifier
    config = CONFIG
    params = Params().normalize()
    identifier = ComplexAreaIdentifier(config)
    identifier._initialize(params)

    # Read image
    print(f"Reading image from {img_path}")
    frame = cv2.imread(img_path)
    if frame is None:
        print("Failed to read image")
        return

    # Process frame
    print("Processing frame...")
    success = identifier.process_frame(frame, save_path)
    
    if success:
        print("Processing successful")
        # Check outputs
        cloth_cut_path = save_path.replace('.png', '_cloth_cut.png')
        if os.path.exists(cloth_cut_path):
            print(f"Cloth cut saved to {cloth_cut_path}")
            # Check channels
            img = cv2.imread(cloth_cut_path, cv2.IMREAD_UNCHANGED)
            print(f"Image shape: {img.shape}")
            if img.shape[2] == 4:
                print("Image has alpha channel (Transparent)")
            else:
                print("Image is RGB (No transparency)")
        else:
            print("Cloth cut file not found")
    else:
        print("Processing failed (maybe no person/cloth detected)")

if __name__ == "__main__":
    test_extract()
