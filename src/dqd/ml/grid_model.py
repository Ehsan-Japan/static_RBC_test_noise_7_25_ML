"""
grid_model.py — RayToLinesNet: sparse ray peaks -> dense transition-line map.

Input  : (batch, 2, H, W)  ch0 peaks, ch1 visited
Output : (batch, H, W)     per-pixel transition LOGIT (sigmoid -> probability)

A small U-Net.  The task is sparse-to-dense: a few dozen peak pixels have to
become continuous lines across the whole diagram, so the network needs both a
wide view (which slope family is this, where do the parallel lines repeat) and
pixel-precise output (a line one pixel wide).  Down-and-up with skips is the
standard answer to exactly that pair of demands.

Two properties matter for the budget study:

  * fully convolutional — no flatten, no fixed-size layer.  Any H, W works,
    and every measurement budget uses the SAME network with the SAME number
    of parameters.  Accuracy differences across the sweep are then caused by
    the measurement, not by model capacity.

  * upsampling resizes to the skip tensor's actual size, so odd grids
    (100 -> 50 -> 25 -> 12) reassemble exactly.

WIDTH is the number of filters in each convolution — the model's capacity
dial, and the only thing here you could sensibly tune.  It is set once,
below, and nothing else has to know about it.  32 is a reasonable size for
100 x 100 diagrams; drop it to 16 if training overfits.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 32


def _block(cin: int, cout: int) -> nn.Sequential:
    """Two 3x3 convolutions — the standard U-Net rung."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.GELU(),
    )


class RayToLinesNet(nn.Module):
    def __init__(self, in_channels: int = 2, depth: int = 3):
        super().__init__()
        widths = [WIDTH * 2 ** i for i in range(depth + 1)]      # 32,64,128,256

        self.down = nn.ModuleList()
        cin = in_channels
        for w in widths[:-1]:
            self.down.append(_block(cin, w))
            cin = w
        self.bottom = _block(cin, widths[-1])
        cin = widths[-1]                                     # bottom's output

        self.up = nn.ModuleList()
        for w in reversed(widths[:-1]):
            self.up.append(_block(cin + w, w))                   # concat skip
            cin = w
        self.head = nn.Conv2d(cin, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for blk in self.down:
            x = blk(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottom(x)
        for blk, skip in zip(self.up, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = blk(torch.cat([x, skip], dim=1))
        return self.head(x).squeeze(1)                            # (B, H, W)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
