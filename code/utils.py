import torch

# def calc_back_sub_weights(self):
#     inputs = torch.ones(self.in_channels * self.img_dim_in * self.img_dim_in)
#     reshape_conv = ReshapeConv(self.in_img_dim, self.out_img_dim, self.in_channels, self.out_channels, self.concrete_layer)
#     return jacobian(reshape_conv, inputs)

# def calc_back_sub_bias(self):
#     bias = torch.ones(1, self.out_channels, self.out_img_dim, self.out_img_dim)
#     return torch.flatten(bias * self.b.unsqueeze(0).unsqueeze(2).unsqueeze(3)).size()


class ReshapeConv(torch.nn.Module):
    def __init__(self, in_img_dim, out_img_dim, in_channels, out_channels, layer):
        super(ReshapeConv, self).__init__()
        self.in_img_dim = in_img_dim
        self.out_img_dim = out_img_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layer = layer

    def forward(self, x):
        out = self.layer(x.view(1, self.in_channels, self.in_img_dim, self.in_img_dim))
        return torch.flatten(out)


# def toeplitz_convmatrix2d(kernel, image_shape):
#     # kernel: (out_channels, in_channels, kernel_height, kernel_width, ...)
#     # image: (in_channels, image_height, image_width, ...)
#     assert image_shape[0] == kernel.shape[1]
#     assert len(image_shape[1:]) == len(kernel.shape[2:])
#     result_dims = torch.tensor(image_shape[1:]) - torch.tensor(kernel.shape[2:]) + 1
#     m = torch.zeros((
#         kernel.shape[0], 
#         *result_dims, 
#         *image_shape
#     ))
#     for i in range(m.shape[1]):
#         for j in range(m.shape[2]):
#             m[:,i,j,:,i:i+kernel.shape[2],j:j+kernel.shape[3]] = kernel
#     return m.flatten(0, len(kernel.shape[2:])).flatten(1)