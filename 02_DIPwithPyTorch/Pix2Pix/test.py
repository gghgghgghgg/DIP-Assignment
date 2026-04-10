import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from facades_dataset import FacadesDataset
from FCN_network import FullyConvNetwork

def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor to a NumPy array suitable for OpenCV.
    """
    image = tensor.cpu().detach().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = (image + 1) / 2
    image = (image * 255).astype(np.uint8)
    return image

def main():
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize dataloader
    # Ensure you have generated test_list.txt before running
    list_file = 'test_list.txt'
    if not os.path.exists(list_file):
        print(f"Error: {list_file} not found. Please generate the test list first.")
        return
        
    test_dataset = FacadesDataset(list_file=list_file)
    # Batch size 1 is easier for saving individual images
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Initialize model
    model = FullyConvNetwork().to(device)

    # Automatically find the latest checkpoint
    checkpoint_dir = 'checkpoints'
    if not os.path.exists(checkpoint_dir):
        print(f"Error: Directory '{checkpoint_dir}' does not exist.")
        return
        
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoints:
        print(f"Error: No weight files found in '{checkpoint_dir}'.")
        return
        
    # Sort checkpoints by epoch number assuming format 'pix2pix_model_epoch_XXX.pth'
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])

    print(f"Loading weights from {latest_checkpoint}...")
    model.load_state_dict(torch.load(latest_checkpoint, map_location=device))
    model.eval()

    # Make output directory
    out_dir = 'test_results'
    os.makedirs(out_dir, exist_ok=True)

    print(f"Starting testing on {len(test_dataset)} images...")

    with torch.no_grad():
        for i, (image_rgb, image_semantic) in enumerate(test_loader):
            image_rgb = image_rgb.to(device)
            image_semantic = image_semantic.to(device)

            # Forward pass
            outputs = model(image_rgb)

            # Convert tensors to images
            input_img_np = tensor_to_image(image_rgb[0])
            target_img_np = tensor_to_image(image_semantic[0])
            output_img_np = tensor_to_image(outputs[0])

            # Concatenate the images horizontally: [Input | Ground Truth | Output]
            comparison = np.hstack((input_img_np, target_img_np, output_img_np))

            # Save the result
            save_path = os.path.join(out_dir, f'result_{i + 1:04d}.png')
            cv2.imwrite(save_path, comparison)

            if (i + 1) % 10 == 0 or (i + 1) == len(test_dataset):
                print(f"Processed {i + 1}/{len(test_dataset)} images.")

    print(f"Testing completed! All results are saved in the '{out_dir}' folder.")

if __name__ == '__main__':
    main()
