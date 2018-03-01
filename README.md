# Large Scale GeoVisual Search

Implementing image visual similarity searching and indexing in NAIP GIS satellite imagery of lower California using Deep CNN. Trained ResNet architecture on UC Merced Land Use Dataset with data augmentation to learn geographical imagery features. Truncated last layer to get output features that are converted into 512-bit binary vector that compactly represents the input image from NAIP satellite image after forward pass, This vector is used to calculate hamming distance for similarity based search.


Further aim to parallelize searching across multiple nodes in AWS cluster using Hadoop/Spark.
