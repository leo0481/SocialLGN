# SocialLGN: Light Graph Convolution Network for Social Recommendation

# Dataset
We provide two datasets: [LastFM](https://grouplens.org/datasets/hetrec-2011/) and [Ciao](https://www.cse.msu.edu/~tangjili/datasetcode/truststudy.htm).

# Example to run the codes
1. Environment: I have tested this code with python3.8 Pytorch=1.7.1 CUDA=11.0
2. Run SocialLGN

    `python main.py --model=SocialLGN --dataset=lastfm --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --topks="[10,20]" --recdim=64 --bpr_batch=2048`
