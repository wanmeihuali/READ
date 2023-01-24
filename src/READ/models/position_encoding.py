import torch

class PositionEncoding(torch.nn.Module):
    """
    Convert SE3 matrix(camera postion into a 3D Rotation Encoding)
    """
    def __init__(self):
        super().__init__()

    def forward(self, view_matrices: torch.Tensor) -> torch.Tensor:
        """
        Using the idea from "https://zhengyiluo.github.io/assets/pdf/Rotation_DL.pdf"
        use 6DOF Continuous Rotation Encoding, which is a 6D vector formed by the first two
        columns of the rotation matrix.
        Args:
            view_matrices: (B, 4, 4) SE3 matrix
        Returns:
            (B, 6) 3D rotation encoding
        """
        # (B, 3, 3)
        batch_size = view_matrices.shape[0]
        rotation_matrix = view_matrices[:, :3, :3]
        # (B, 3, 2)
        rotation_encoding = rotation_matrix[:, :, :2]
        # (B, 6)
        rotation_encoding = rotation_encoding.view(batch_size, -1)
        return rotation_encoding

        