![image](https://github.com/Lingjing-Zhang/ChemNTP/assets/150929272/22877af8-628f-4896-8dd6-0a9b451f84a0)

ChemNTP
======
The chemicals released into the environment can enter the human body through various exposure pathways, ultimately leading to harmful outcomes. In fact, many environmental chemicals detected have been confirmed to possess neurotoxicity, posing a significant threat to human health. Unfortunately, the molecular mechanisms of neurotoxicity of many environmental chemicals, especially the in vivo targets, remain unclear. To address this issue, a method, chemical neurotoxicity target prediction (ChemNTP) model, for predicting chemical neurotoxicity targets that integrates the structural characterization of environmental chemicals and biological targets was proposed. This method overcomed the limitations of traditional prediction methods, which are only applicable to single target and mechanism, to achieve rapid screening of 199 potential neurotoxic targets and key molecular initiating events (MIEs). The deep learning model constructed shows superior predictive performance compared to other machine learning algorithms, with an area under the receiver operating characteristic curve (AUCROC) of 0.922 for the validation set. Additionally, the attention mechanism of the ChemNTP model can recognise key residues of targets and key functional atoms of chemicals, revealing the structural basis of complex interactions.

User Guide
======
ChemNTP predict
-------
Users can predict the interactions between chemicals and proteins by running the predict.py in the ChemNTP_predict package.

Attention analysis
-------
Users can identify the key residues of targets and key functional atoms of chemicals in the compound-protein interaction, by by running compound_attention_ache.py and protein_attention_ache.py to get the attention weights of the protein residues and chemical atoms.

Hyperparameter tunning
-------
Running BayesianOptimization.py for hyperparameter tunning.

Chemical-protein interaction Data
-------
All modeling data used in this paper is freely available at https://github.com/Lingjing-Zhang/ChemNTP/releases.

Executable Software with GUl
-------
We provide a free user-friendly graphical interface that allows non-computational professionals to use ChemNTP for rapid screening of 199 neurotoxic targets of environmental chemicals. This is also the first application in the environmental field using deep learning methods to predict potential pollutants' targets while obtaining molecular mechanisms. The software package download and more usage instructions can be accessed at https://github.com/Lingjing-Zhang/ChemNTP/releases.
The following is a schematic diagram of how to use the software.
![软件流程](https://github.com/Lingjing-Zhang/ChemNTP/assets/150929272/74e5bfed-310e-42c8-acb6-526239238b1e)

