import torch
from modules.voxel import Voxelization

def test_voxelization():
    # Define parameters for Voxelization
    voxel_size = [0.1, 0.1, 0.2]  # [x, y, z]
    point_cloud_range = [0, -10, -3, 10, 10, 3]  # [x_min, y_min, z_min, x_max, y_max, z_max]
    max_num_points = 5  # Maximum points per voxel
    max_voxels = (100, 200)  # (training, testing) max voxels
    deterministic = True

    # Create Voxelization instance
    voxel_layer = Voxelization(
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        max_num_points=max_num_points,
        max_voxels=max_voxels,
        deterministic=deterministic
    )

    # Generate random point cloud (N, ndim)
    # Points in the range of the point_cloud_range
    num_points = 1000  # Total number of points
    ndim = 4  # Number of dimensions (e.g., x, y, z, reflectivity)
    points = torch.rand((num_points, ndim)) * torch.tensor([10, 20, 6, 1]) - torch.tensor([0, 10, 3, 0])

    # Forward pass through the voxelization layer
    voxel_layer.train()  # Set to training mode
    voxels, coors, num_points_per_voxel = voxel_layer(points)

    # Print results
    print(f"Voxels shape: {voxels.shape}")  # (M, max_num_points, ndim)
    print(f"Coordinates shape: {coors.shape}")  # (M, 3)
    print(f"Num points per voxel shape: {num_points_per_voxel.shape}")  # (M,)

    # Validate results
    assert voxels.dim() == 3
    assert coors.dim() == 2
    assert num_points_per_voxel.dim() == 1

    print("Voxelization test passed!")

if __name__ == "__main__":
    test_voxelization()
