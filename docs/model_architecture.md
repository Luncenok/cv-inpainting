# Model Architecture

## Skip Connections
Skip connections, also known as residual connections, are direct paths between encoder and decoder layers that bypass the bottleneck. They serve several important purposes:
1. **Preserve Fine Details**: Allow high-resolution features from encoder layers to directly reach decoder layers
2. **Prevent Information Loss**: Help maintain spatial information that might be lost during downsampling
3. **Ease Gradient Flow**: Make training deeper networks easier by providing direct paths for gradient backpropagation
4. **Feature Reuse**: Enable the decoder to combine high-level semantic features with low-level detail features

In our architecture, skip connections connect each encoder block to its corresponding decoder block at the same resolution level.

## Architecture Diagram
```mermaid
graph TD
    subgraph Generator
        Input[Input:<br/>RGB + Mask] --> E1[Encoder 1<br/>64ch]
        E1 --> E2[Encoder 2<br/>128ch]
        E2 --> E3[Encoder 3<br/>256ch]
        E3 --> E4[Encoder 4<br/>512ch]
        
        E4 --> B1[Bottleneck<br/>ConvBlock]
        B1 --> ATT[Attention<br/>Block]
        ATT --> B2[Bottleneck<br/>ConvBlock]
        
        B2 --> D4[Decoder 4<br/>256ch]
        E3 --Skip--> D4
        
        D4 --> D3[Decoder 3<br/>128ch]
        E2 --Skip--> D3
        
        D3 --> D2[Decoder 2<br/>64ch]
        E1 --Skip--> D2
        
        D2 --> D1[Decoder 1<br/>32ch]
        Input --Skip--> D1
        
        D1 --> Output[Output:<br/>RGB]
    end

    subgraph Discriminator
        DInput[Input:<br/>RGB + Mask] --> DC1[Conv 1<br/>64ch]
        DC1 --> DC2[Conv 2<br/>128ch]
        DC2 --> DC3[Conv 3<br/>256ch]
        DC3 --> DC4[Conv 4<br/>512ch]
        DC4 --> DOutput[Output:<br/>Real/Fake]
    end

    subgraph Loss_Functions
        Output --> L1[L1 Loss]
        RealImg[Ground Truth<br/>RGB] --> L1
        
        Output --> PL[Perceptual<br/>Loss VGG16]
        RealImg --> PL
        
        DOutput --> ADV[Adversarial<br/>Loss BCE/MSE]
        IsReal[Real/Fake<br/>Label] --> ADV
        
        L1 --> TotalG[Total Generator<br/>Loss]
        PL --> TotalG
        ADV --> TotalG
        
        ADV --> TotalD[Total Discriminator<br/>Loss]
    end

    classDef block fill:#f9f,stroke:#333,stroke-width:2px
    classDef loss fill:#ffd,stroke:#333,stroke-width:2px
    class E1,E2,E3,E4,B1,ATT,B2,D4,D3,D2,D1,DC1,DC2,DC3,DC4 block
    class L1,PL,ADV,TotalG,TotalD loss

```

## Loss Functions
1. **L1 Loss**: Pixel-wise absolute difference between generated and ground truth images
   - Weight (λ₁): 100
   - Purpose: Ensure color and structural accuracy

2. **Perceptual Loss**: Feature-level difference using VGG16 network
   - Weight (λ₂): 10
   - Purpose: Capture high-level semantic similarities

3. **Adversarial Loss**: Binary Cross Entropy (BCE) or Mean Squared Error (MSE)
   - Purpose: Train generator to produce realistic images that can fool the discriminator
   - Used in both generator and discriminator training

Total Generator Loss = Adversarial Loss + 100 * L1 Loss + 10 * Perceptual Loss
Total Discriminator Loss = Average of real and fake adversarial losses
