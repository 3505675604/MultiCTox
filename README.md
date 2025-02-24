MultiCTox :Template for Submission of Manuscripts to American Chemical Society Journals



## Requirements
- python==3.9.13
- numpy==1.19.5
- pandas==1.5.3
- scikit_learn==1.2.1
- torch==1.13.1
- torch-geometric == 2.0.4

##Model set download address:https://huggingface.co/seyonec/PubChem10M_SMILES_BPE_450k

##The folder data contains the data set hERG-60/hERG-70,Cav1.2-60/Cav1.2-70,Nav1.5-60/Nav1.5-70.

##In the main function we marked how to modify the data set.

##Our parameter modifications can be made in config.py.

## Usage

`python main.py`
