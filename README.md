# GMEC-LLMs
[Enhanced Generative Framework with LLMs for Multimodal Emotion-Cause Pair Extraction in Conversations](https://ieeexplore.ieee.org/abstract/document/10891643)

![image](https://github.com/user-attachments/assets/87df521d-2438-45df-8ff7-9d2425a7ecc0)

## Tables of Contents
- [Environment](#Environment)
- [Visual Caption](#VisualCaption)
- [Dataset](#Dataset)
- [GMEC-train](#GMEC)
- [LLMs Enhanced](#enhance)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [License](#license)

## Environment  <a name="Environment"></a>

* [Python](https://www.python.org/downloads/) >= 3.9
* [Cuda](https://developer.nvidia.com/cuda-toolkit) >= 11.3
* [Pytorch](https://pytorch.org/get-started/locally/) >= 12.0
* The details can be find in [environment.yml](environment.yml)


## Visual Caption <a name="VisualCaption"></a>
We use the [MinGPT-4](https://github.com/ai-liam/NLP-MiniGPT-4) to extact the visual caption.
1. download the MinGPT-4 and its 13B checkpoint.
2. Add the image_demo files into the folder. We revise the [image_demo.py](MinGPT-4/image_demo.py) to adapt our task
3. Extract the Caption about the Image, and save to json file.

## Dataset <a name="Dataset"></a>

## GMEC-train <a name="GMEC"></a>
### step 1: Extract image features.
```
bash scripts/extract_img_features.sh  # extract image features
python tools/dense_img_feats.py # dense the features into one file.
```
### step 2: train the GMEC models
About val **ver** in train.sh:  
1. **Emotion** means the MER task of recognizing six emotion categories;  
2. **Emotion_neu** means the MER task of recognizing two emotion categories;  
3. **cause** means the MCE task of extracting causes;  
4. **preEmo_precause_pair(pe_pc)** means the emotion and its cause are predicted simultaneously;  

About the parameter Context_cut in config/train.yaml:
1. **realtime** means there is no follow-up utterances after emotion
2. **static** means there is compelete context.

Training the GMEC model for different tasks  
```
bash scripts/train.sh
```
### step 3: Evaluating the results
Evaluate the results
```
bash evaluate_model.sh
```

Combining the emotion results and emotion-cause pairs to evaluate the task of MECPE-C
```
# select the function of **_evaluate_for_emo_pepc()** 
python evaluation/mecpec_evaluate.py 
```

## LLMs Enhanced <a name="enhance"></a>
### Generate the  heuristics 
 ```
bash heuristics_gen.sh
```

### prompting 
Add the openai key to the file.
```
bash prompt.sh
```


### evaluating
```
# select the function of **evaluate_for_prompt_emo_pepc()** 
python evaluation/mecpec_evaluate.py 
```



The code will be open later. If you have any question, you can email xcju@stu.suda.edu.cn
