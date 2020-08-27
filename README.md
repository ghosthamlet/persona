
## Open domain persona-aware dialogue generation

This codebase implemented four papers about open domain persona-aware dialogue generation, Code quality is low, just can be used for experiments:

1. Improvement of a dedicated model for open domain persona-aware dialogue generation (https://github.com/ghosthamlet/persona/blob/master/paper/persona.pdf): <br />
AttentionRoutingPlus/<br />
(Please contact zhengyinhe1@163.com for the PersonalDialog dataset)

2. A Pre-training Based Personalized Dialogue Generation Model with Persona-sparse Data (http://arxiv.org/abs/1911.04700): <br />
AttentionRouting/<br />
(Please contact zhengyinhe1@163.com for the PersonalDialog dataset)

3. Personalized Dialogue Generation with Diversified Traits (http://arxiv.org/abs/1901.09672): <br />
PersonalityTraitFusion/<br />
(Please contact zhengyinhe1@163.com for the PersonalDialog dataset)

4. Assigning personality/identity to a chatting machine for coherent conversation generation (http://arxiv.org/abs/1706.02861): <br />
AssignPersonality/<br />
(Personality Assignment dataset: http://coai.cs.tsinghua.edu.cn/hml/dataset/#AssignPersonality)

The first two papers code tested on ubuntu16.10 and python3.6.10.

The last two papers code still have problems for training , don't use them. 


## Install

`git clone https://github.com/ghosthamlet/persona.git`

cd into the corresponding paper code directory, then:

`pip install -r requirements.txt`


## Data prepare

for datasets of the first two papers, segment dialog sequences to characters separated by space, randomly extract 100000, 20000, 20000 sessions for three datasets without replacement: train_char.txt, valid_char.txt and test_char.txt, place them into datas directory of corresponding paper code.


## Training

in the corresponding paper code directory:

`python trainer.py --config_file configs/default.yaml`


## Evaluation 

in the corresponding paper code directory:

`python evaler.py --config_file configs/evaler.yaml`
