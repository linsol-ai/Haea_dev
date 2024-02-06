import torch
from torch.nn.functional import gumbel_softmax
from torch import nn, einsum
from einops import rearrange
from math import log2, sqrt

from models.layers import ResidualBlock

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner


class DVAE(torch.nn.Module):
    """The discrete variational autoencoder.

    TODO: input image normalization
    TODO: bottleneck style encoder (like in the original DALL-E)
    TODO: reinmax
    Based on: https://github.com/lucidrains/DALLE-pytorch/blob/main/dalle_pytorch/dalle_pytorch.py

    References:
    (1) Zero-Shot Text-to-Image Generation
    (2) Neural Discrete Representation Learning
    """

    def __init__(
        self,
        *,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        temperature: float = 0.9,
        codebook_size: int = 512,
        codebook_vector_dim: int = 512,
    ) -> None:
        """Init the DVAE.

        Args:
            encoder: The encoder used to encode the image. The dimensionality of its output
                should match the dimensionality of the decoder's input. The shape of the encoder's
                output should be: `(batch_size, codebook_size, width, height)`.
            decoder: The decoder used to decode the image given a matrix of latent variables.
                The dimensionality of its input should match the dimensionality
                of the encoder's input.
            temperature: The temperature parameter for Gumbel Softmax sampling.
            codebook_size: Number of vectors in the codebook.
            codebook_vector_dim: Dimensionality of the codebook's vector.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.temperature = temperature
        self.codebook = torch.nn.Embedding(codebook_size, codebook_vector_dim)
        self.codebook_size = codebook_size
        self.codebook_vector_dim = codebook_vector_dim

    def forward(
        self, x: torch.Tensor, temperature: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.encoder(x)
        if temperature is None:
            temperature = self.temperature

        # (batch, width, height, self.codebook_size)
        one_hot = gumbel_softmax(logits, dim=-1, tau=temperature, hard=False)
        sampled = einsum('b n h w, n d -> b d h w', one_hot, self.codebook.weight)
        decoded = self.decoder(sampled)

        return decoded, logits


    def decode(self, img_seq, img_size) -> torch.Tensor:
        w, h = img_size
        image_embeds = self.codebook(img_seq)
        print(image_embeds.shape, w, h)
        b, n, d = image_embeds.shape
        image_embeds = rearrange(image_embeds, 'b (w h) d -> b d w h', h = h, w = w)
        images = self.decoder(image_embeds)
        return images
    

    @torch.no_grad()
    @eval_decorator
    def get_codebook_indices(self, images):
        logits = self.encoder(images)
        codebook_indices = logits.argmax(dim = 1).flatten(1)
        return codebook_indices, logits



class Conv2DEncoder(torch.nn.Module):
    """An image encoder based on 2D convolutions.

    Based on: https://github.com/lucidrains/DALLE-pytorch/blob/main/dalle_pytorch/dalle_pytorch.py
    """

    def __init__(
        self,
        *,
        input_channels: int = 3,
        output_channels: int = 512,
        hidden_size: int = 64,
        num_layers: int = 3,
        num_resnet_blocks: int = 1,
    ):
        """Init the encoder.

        Args:
            input_channels: Number of input channels.
            output_channels: Number of the output channels.
            hidden_size: Number of channels in the intermediate hidden layers.
            num_layers: Number of hidden layers.
            num_resnet_blocks: Number of resnet blocks added after each layer.
        """
        super().__init__()
        self.num_layers = num_layers

        layers_list: list[torch.nn.Module] = [
            torch.nn.Conv2d(input_channels, hidden_size, kernel_size=1),
            torch.nn.ReLU()
        ]
        for _ in range(num_layers):
            layers_list.extend(
                [
                torch.nn.Conv2d(hidden_size, hidden_size, kernel_size=4, stride=2, padding=1),
                torch.nn.GELU()
                ]
            )
        
        for _ in range(num_resnet_blocks):
            layers_list.append(
                ResidualBlock(hidden_size)
            )


        layers_list.append(torch.nn.Conv2d(hidden_size, output_channels, kernel_size=1))
        self.layers = torch.nn.Sequential(*layers_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode an image.

        Args:
            x: The input image of shape `(batch, input_channels, in_width, in_height)`

        Returns:
            The encoder image of shape `(batch, output_channels, out_width, out_height)`
        """
        return self.layers(x)

    def calculate_size(self, size):
        w, h = size
        div = 2 ** (self.num_layers)
        return w // div, h // div


class Conv2DDecoder(torch.nn.Module):
    """An image decoder based on 2D transposed convolutions.

    Based on: https://github.com/lucidrains/DALLE-pytorch/blob/main/dalle_pytorch/dalle_pytorch.py
    """

    def __init__(
        self,
        *,
        input_channels: int = 512,
        output_channels: int = 3,
        hidden_size: int = 64,
        num_layers: int = 3,
        num_resnet_blocks: int = 1,
    ):
        """Init the encoder.

        Args:
            input_channels: Number of input channels (dimensionality of the codebook).
            output_channels: Number of output channels.
            hidden_size: Number of channels in the intermediate hidden layers.
            num_layers: Number of hidden layers.
            num_resnet_blocks: Number of resnet blocks added after each layer.
        """
        super().__init__()
        layers_list: list[torch.nn.Module] = [
            torch.nn.Conv2d(input_channels, hidden_size, kernel_size=1),
            torch.nn.ReLU()
        ]
        for _ in range(num_layers):
            layers_list.extend(
                [
                    torch.nn.ConvTranspose2d(
                        hidden_size, hidden_size, kernel_size=4, stride=2, padding=1
                    ),
                    torch.nn.GELU()
                ]
            )
        
        for _ in range(num_resnet_blocks):
            layers_list.append(
                ResidualBlock(hidden_size)
            )

        layers_list.append(torch.nn.Conv2d(hidden_size, output_channels, kernel_size=1))
        self.layers = torch.nn.Sequential(*layers_list)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Encode an image given latent variables.

        Args:
            z: The input image of shape `(batch, input_channels, in_width, in_height)`.

        Returns:
            The encoder image of shape `(batch, output_channels, out_width, out_height)`.
        """
        return self.layers(z)
