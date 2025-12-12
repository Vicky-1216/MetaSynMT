# MetaSynMT
Prediction of Synergistic and Safe Drug Combinations for Parasitic Diseases Via Meta-path Information Aggregation and Multi-task Learning
<img src="./figure.png" width="900">
we propose a novel multi-task learning model, MetaSynMT, based on a meta-path aggregation mechanism to predict synergistic and safe drug combinations.
This mechanism effectively captures drug features by enhancing both structural and high-order semantic representations.
In addition to the primary task of synergy prediction, a side effect prediction task is introduced as an auxiliary task to jointly identify drug combinations with strong synergy and low toxicity.

## Requirements
## Python requirements
⦁	Python >= 3.9
⦁	Pytorch >= 1.10
⦁	dgl >= 11.0
⦁ sklearn >= 1.0.2
⦁ pandas >= 1.3.5

## Main Data:
CCLE_expression.csv: Downloaded original DepMap gene expression data.
drugcomb_alldruginfo_dict.pickle: Collected information of involved drugs.
drugcomb_synergy_score.csv: Collected drug-drug-cell line synergy score samples.
twosides_side_effect.csv: Collected drug-drug adverse effect samples.
drug_target_interaction.csv: Collected drug-target interaction samples.
target_target_interaction.csv: Collected target-target interaction samples.

## Usage

Run the model by executing the Synergytrain.py file.
