"""
Plotting utilities for ABC-Net inference visualization.
Extracted from img2smiles.py for modular use.
"""

import matplotlib.pyplot as plt
import numpy as np
import os


def plot_inference_results(
    img,
    atom_target_img,
    atoms_position_list_final,
    atoms_type_list_final,
    bond_target_img,
    bonds_position_list_final,
    bonds_property_list_final,
    bonds_delta_list_final,
    save_path,
    dpi=1000
):
    """
    Plot inference results with original image, atoms, and bonds.
    
    Args:
        img: Original input image [H, W]
        atom_target_img: Atom target heatmap [H, W]
        atoms_position_list_final: List of atom positions [[x, y], ...]
        atoms_type_list_final: List of atom types
        bond_target_img: Bond target heatmap [H, W]
        bonds_position_list_final: List of bond positions [[x, y], ...]
        bonds_property_list_final: List of bond types
        bonds_delta_list_final: List of bond direction vectors [[dx, dy], ...]
        save_path: Path to save the plot
        dpi: DPI for saved image
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(15, 5))
    
    # Plot original image
    plt.subplot(131)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Plot atoms
    plt.subplot(132)
    plt.imshow(atom_target_img.detach().cpu().numpy(), cmap='hot')
    plt.title('Detected Atoms')
    plt.axis('off')
    for m, position in enumerate(atoms_position_list_final):
        x, y = position
        position = [y, x]
        plt.annotate(atoms_type_list_final[m], xy=position, fontsize=6, 
                    color='white', weight='bold')
    
    # Plot bonds
    ax = plt.subplot(133)
    plt.imshow(bond_target_img.detach().cpu().numpy(), cmap='hot')
    plt.title('Detected Bonds')
    plt.axis('off')
    for m, position in enumerate(bonds_position_list_final):
        x, y = position
        position = [y, x]
        ax.annotate(str(bonds_property_list_final[m]), xy=position, fontsize=6,
                   color='white', weight='bold')
        x, y = position
        delta_y, delta_x = bonds_delta_list_final[m]
        ax.plot([x - delta_x, x + delta_x], [y - delta_y, y + delta_y], 
                color='cyan', linewidth=1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()


def plot_debug_info(
    imgs,
    atom_targets_pred,
    bond_targets_pred,
    save_path,
    sample_idx=0,
    dpi=1000
):
    """
    Plot debug information with original image and prediction heatmaps.
    
    Args:
        imgs: Batch of input images [B, 1, H, W]
        atom_targets_pred: Atom target predictions [B, 1, H, W]
        bond_targets_pred: Bond target predictions [B, 1, H, W]
        save_path: Path to save the plot
        sample_idx: Index of sample to plot from batch
        dpi: DPI for saved image
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(15, 5))
    
    # Plot original image
    plt.subplot(131)
    plt.imshow(imgs.cpu().numpy()[sample_idx, 0], cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Plot atom predictions
    plt.subplot(132)
    plt.imshow(atom_targets_pred.cpu().numpy()[sample_idx, 0], cmap='hot')
    plt.title('Atom Predictions')
    plt.axis('off')
    
    # Plot bond predictions
    plt.subplot(133)
    plt.imshow(bond_targets_pred.cpu().numpy()[sample_idx, 0], cmap='hot')
    plt.title('Bond Predictions')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()


def plot_3d_surface(z, save_path, dpi=1000):
    """
    Plot 3D surface of a 2D array.
    
    Args:
        z: 2D array to plot as surface
        save_path: Path to save the plot
        dpi: DPI for saved image
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    h, l = z.shape
    x, y = np.meshgrid(np.arange(0, h), np.arange(0, l))
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title('3D Surface Plot')
    
    plt.savefig(save_path, dpi=dpi)
    plt.close()
