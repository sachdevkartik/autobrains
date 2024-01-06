# __Imitation Learning__
  
PyTorch-based library for imitation learning using Behaviour cloning for self-driving cars. Provides the provision of using :
    
- RNN architecture (GRU, LSTM etc.)
- Vision backbone (CNN, Vision Transformer)

# __Installation__
To install locally, using pip:
```bash
git clone git@github.com:sachdevkartik/autobrains.git
cd autobrains

sudo apt-get update && apt-get install python3-pip
pip3 install .
```

Note: Installation using docker is not tested yet due to space issues in my workstation

# __Training__

Modify the configuration of the chosen model and training scheme from ```config``` folder. Run the script:

```bash
# training with cnn
python3 scripts/main.py  --model cnn --model_config ../config/baseline2.yaml --common_config ../config/common.yaml

# training with Vision Transformer : Convolutional Vision Transformer (CvT)
python3 scripts/main.py --model cvt --model_config ../config/cvt.yaml --common_config ../config/common.yaml

# training with Vision Transformer : LeViT
python3 scripts/main.py --model levit --model_config ../config/levit.yaml --common_config ../config/common.yaml
```

Please use the following models and the corresponding config files:

| Name          | Config file | 
| ------------- | --- | 
| cnn      | ../config/baseline2.yaml  | 
| cvt    | ../config/cvt.yaml  | 
| levit   | ../config/levit.yaml  | 


# __Visualize__

## Visualize Trained Model
To visualize the output of the model, make sure that logger directory and respestive files are present in the root directory. For example:

```bash
python3 scripts/visualize_data.py --config ../logger/2024-01-06-07-43-10/config.yaml
```

## Visualize Dataset
To visualize the dataset, run the following:

```bash
python3 scripts/visualize_data.py 
```
The file loads and visualize one of the instance for e.g. ``20230910-094935`` of the dataset.

TODOs:

    [x] logger
    [x] save checkpoint
    [x] save config
    [x] plot: 
        [x] loss 
        [x] trajectory (full & segments)
    [] docker
    [x] setup.py
        [x] requirements
    [] Readme.md
    [] Profiling





# References
https://github.com/HumanCompatibleAI/imitation?tab=readme-ov-file

https://stackoverflow.com/questions/582336/how-do-i-profile-a-python-script