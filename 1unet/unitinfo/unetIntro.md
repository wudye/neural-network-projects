# UNET Architecture
    The bottleneck acts as a bridge between Contracting (Analysis) and Expanding (Synthesis).
    Encoder side: Focuses on what is in the image (classification).
    Decoder side: Focuses on where it is (localization).
    The Bottleneck transforms the "what" features into a format that the decoder can use to start placing them back into a full-sized map.

# each encoder and decoder block uses two successive 3 * 3 convolutions
    1. two 3 * 3 convolutions = 5 * 5 convolution
    parameters = 3 * 3 * in_channels * out_channels + 3 * 3 * out_channels * out_channels = 18 * in_channels * out_channels
    parameters = 5 * 5 * in_channels * out_channels = 25 * in_channels * out_channels
    18 : 25 = 72%  reduction in parameters (also why most deep learning models use small convolutions)
    use two 3 * 3 convolutions instead of one 5 * 5 can also see the same receptive field(5 * 5 area) but reduce the 
     number of parameters and also add more non-linearity to the model (more layers and more activation functions)
    2. Convolution is the "Thinking" and GELU is the "Decision."
# Interpolate  vs ConvTranspose2d
    
    Scenario	                        Recommendation
    Small Dataset / Weak GPU	        interpolate (Keep it simple and avoid overfitting).
    Complex Medical/Satellite Images	ConvTranspose2d (The model needs to learn specific textures).
    Modern Best Practice	            interpolate + Conv2d.

# ConvTranspose2d 
    output_size = (input_size - 1) × stride - 2 × padding + dilation × (kernel_size - 1) + output_padding + 1
    ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)
   normal convolution: input -> output (2 * 2  -> 1 * 1)
   ConvTranspose2d: input -> output (1 * 1 -> 2 * 2)
   input(2 * 2) = [a, b],        weight(2*2) = [w1, w2, w3, w4]
                  [c, d]
   (convTranspose2d)output(4 * 4) = [a * w1 , a * w2 , b * w1 , b * w2],
                                    [ a * w3, a* w4,   b * w3 , b * w4],
                                    [c * w1 ,  c * w2 , d * w1 ,d * w2],
                                    [c * w3 , c * w4 , d * w3 + d * w4]
   (normal) output(1 * 1) = [a * w1 + b * w2 + c * w3 + d * w4] a value


# interpolate 
    input 2* 2 = [[ A, B ], 
                [ C, D ]]
     -> output(4 * 4)= [[ A, A, B, B ],
                         [[ A, A, B, B ],
                         [[ C, C, D, D ],
                         [[ C, C, D, D ]]
     input= [10, 20],
            [30, 40] -> align_corners=True, mode='bilinear'
    Row	Col 1 (0%)	Col 2 (33%)	Col 3 (66%)	Col 4 (100%)
    Row 1 (0%)	10.0	13.3	16.6	20.0
    Row 2 (33%)	16.6	19.9	23.3	26.6
    Row 3 (66%)	23.3	26.6	29.9	33.3
    Row 4 (100%)	30.0	33.3	36.6	40.0