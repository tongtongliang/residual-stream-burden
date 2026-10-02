# Residual-Stream Noise Sensitivity Results

## Depth-mean statistics

For JiT-B, DeCO-B, PixelDiT-B, and DiP-B the entries are means over 24 points `Attn0, MLP0, ..., Attn11, MLP11`. HyperDiT-B is averaged over its 16 semantic-stream points `Attn0, MLP0, ..., Attn7, MLP7`; its registers and fine stream are excluded. `Mean F` is the mean of pointwise fractions, not a ratio reconstructed from the mean `W` and `B`.

| Model | q | Mean absolute W | Mean absolute B | Mean absolute T | Mean ratio F |
|---|---:|---:|---:|---:|---:|
| JiT-B clean | 0.250000 | 3.131853 | 45.857432 | 48.989285 | 0.055988 |
| JiT-B velocity | 0.250000 | 49.876395 | 243.256161 | 293.132556 | 0.208562 |
| DeCO-B velocity | 0.250000 | 0.536969 | 10.078375 | 10.615344 | 0.049127 |
| PixelDiT-B velocity | 0.250000 | 3.973808 | 86.927753 | 90.901561 | 0.036059 |
| DiP-B velocity | 0.250000 | 4.614037 | 20.742822 | 25.356859 | 0.164650 |
| HyperDiT-B velocity | 0.250000 | 4.822691 | 32.388588 | 37.211278 | 0.097653 |
| JiT-B clean | 0.500000 | 6.438843 | 33.423464 | 39.862307 | 0.171397 |
| JiT-B velocity | 0.500000 | 297.493848 | 761.316096 | 1058.809944 | 0.380354 |
| DeCO-B velocity | 0.500000 | 1.162454 | 5.634689 | 6.797144 | 0.168560 |
| PixelDiT-B velocity | 0.500000 | 8.023565 | 59.962804 | 67.986369 | 0.107488 |
| DiP-B velocity | 0.500000 | 3.031442 | 10.202926 | 13.234368 | 0.220171 |
| HyperDiT-B velocity | 0.500000 | 11.166258 | 34.253038 | 45.419296 | 0.231856 |
| JiT-B clean | 0.750000 | 9.128767 | 21.760752 | 30.889520 | 0.398262 |
| JiT-B velocity | 0.750000 | 909.438021 | 1324.556230 | 2233.994251 | 0.522780 |
| DeCO-B velocity | 0.750000 | 1.860597 | 2.954027 | 4.814624 | 0.408932 |
| PixelDiT-B velocity | 0.750000 | 11.019508 | 36.561072 | 47.580580 | 0.259408 |
| DiP-B velocity | 0.750000 | 2.325957 | 4.768201 | 7.094158 | 0.359738 |
| HyperDiT-B velocity | 0.750000 | 17.423636 | 29.876045 | 47.299681 | 0.440535 |

## Bounded interpretation

JiT-B velocity retains substantially larger absolute noise-conditioned variance than JiT-B clean at every measured q, with the gap growing toward high input noise. HyperDiT-B has mean F `0.097653`, `0.231856`, and `0.440535` at q `0.25`, `0.50`, and `0.75`: consistently above JiT-B clean but below JiT-B velocity. Its absolute W is `4.822691`, `11.166258`, and `17.423636`, far below JiT-B velocity and much closer to the other semantic-flow models. PixelDiT-B retains the lowest F at all three q values. These observations establish differential residual sensitivity, not by themselves that the retained variation damages optimization or semantics.
