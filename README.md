# Influence of parallel data on multilingual representation space of language models
This repository contains the source code for training the language models and running the evaluation experiments in the thesis "Influence of parallel data on multilingual representation space of language models". In the thesis, we train four transformer language models of 1.4 billion parameters on multilingual corpora with varying proportions of parallel data: 0%, 1%, 2%, and 5%. We then evaluate the
level of cross-lingual alignment in the models using principal component projections, projection weighted canonical correlation analysis (PWCCA), language-specific neuron analysis, and cross-lingual control vectors.

Thesis available for download at: [Link coming soon]()

## Additional resources
Link to the videos for the changes in the projections during the full training runs: https://www.youtube.com/playlist?list=PLnjJuKzLUmfRGiVAbCsScvZau0KlcvZ-e

## Notes
The source code for training the language models is based on [the source code for training the OLMo models](https://github.com/allenai/OLMo) by AI2 and modified for the training setup and infrastructure used in the thesis. The models are trained using the LUMI HPC cluster, powered by AMD Instinct™ MI250X accelerators.

The cross-lingual control vector code in the evaluation experiments is based on [the source code of Master's thesis "Controlling the Text Generation of a Large Language Model in Multilingual Setting using Latent Space Steering"](https://github.com/shiftleino/multilingual-latent-steering).

## Acknowledgements
The thesis was supported by the Technology Industries of Finland Centennial Foundation.

## Citation
Coming soon...
