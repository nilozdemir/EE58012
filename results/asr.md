Clean accuracy 94.82%; ASR (%) over 9482 correctly classified test images; eps=2/255, PGD steps=100, step=0.05/255, Grad-CAM layer=layer3, cam score=auto, mask steps=100, mask lr=0.05, seed=0.

| Mask | untargeted rho=25% | untargeted rho=50% | targeted rho=25% | targeted rho=50% |
|---|---|---|---|---|
| Grad-CAM mask (A) | 30.6 | 58.8 | 6.3 | 25.2 |
| Random mask (A) | 24.6 | 57.3 | 6.2 | 27.0 |
| Learned, Grad-CAM init (B) | 36.2 | 69.8 | 12.3 | 42.9 |
| Learned, random init (B) | 31.9 | 69.2 | 11.7 | 43.9 |
