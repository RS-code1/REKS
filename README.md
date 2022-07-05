# A Generic Reinforced Explainable Framework with Knowledge Graph for Session-based Recommendation

## Requirements
pytorch

## Datasets
Three Amazon datasets (Beauty, Cellphones, Baby) are available in "jmcauley.ucsd.edu/data/amazon/links.html".

## Run the codes
1. Preprocess the data:

# python data_process.py

2. Construct the knowledge graph:

### python preprocess.py

3. Generate data for TransE method:

python transe_data.py 

4. Train knowledge graph representations by TransE method:

python train_transe_model.py

5. Train our REKS model:

python train_agent.py 

## References
[1] Y. Xian, Z. Fu, S. Muthukrishnan, G. De Melo, and Y. Zhang, “Reinforcement knowledge graph reasoning for explainable recommendation,” in Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval, 2019, pp. 285–294.
