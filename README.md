# MultiCascade-DS-3DUnet
This repo contains the implementation of the Multi-Cascade DS (Denoising-Segmentation) network we used for solving the Classification in Cryo-Electron Tomograms challenge of SHREC 2021 https://www.shrec.net/cryo-et/.<br> The final paper presenting our method and related ones can be found in https://diglib.eg.org/handle/10.2312/3dor20211307 . 
## Topology
THe Multi-Cascade DS network is based on the popular 3D U-Net and consists of two decoding pathways. These perform denoising on the input data (3D tomogram in our case) and volumetric segmentation to produce the 3D label map, respectively. The whole architecture is illustrated in the next figure.
![MC-DS](https://github.com/tommyshelby4/MultiCascade-DS-3DUnet/assets/58388534/ead80895-1034-4ca4-8285-bafa35ebe713)
