<div align="center">

# Marten<img src="assert/Marten.png" width="35"/>: Visual Question Answering with Mask Generation for Multi-modal Document Understanding(CVPR 2025)

</div>


## 📖 Introduction

**Paper**: (🚀🚀🚀 **Accepted by CVPR2025** 🚀🚀🚀): 

<div align="center">
<img src="assert/pipeline.png">
</div>


## 📄 MTMask6M
![Datasets](assert/dataset.png)
- **MTMask6M:**
  - Coming soon

- **Original Document Data Sources:**
  - [DocStruct4M](https://huggingface.co/datasets/mPLUG/DocStruct4M)
  - [DocGenome](https://huggingface.co/datasets/U4R/DocGenome)
  - [IIT-CDIP]()


## 📚 Usage

### 📦 Installation

Ensure you have Python 3.8 or higher installed in your environment.

```bash
git clone https://github.com/Token-family/Marten.git
cd Marten
pip install -r requirements.txt
```


### 🛠️ Creating Your Own Dataset

#### Step 1: Obtain Word-level Bounding Boxes
Use OCR engines([PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR),[CRAFT](https://github.com/clovaai/CRAFT-pytorch)) to generate word-level bounding boxes in the format:
```
/path/to/image\t[[x1_1,y1_1,x2_1,y2_1,x3_1,y3_1,x4_1,y4_1], ... ,[x1_n,y1_n,x2_n,y2_n,x3_n,y3_n,x4_n,y4_n]]
```

#### Step 2: Generate Masks
```bash
python mask_utils/mask_generation.py
```

<!-- **Mask generation visualization** -->


#### Step 3: Data Format
Reference [InternVL2](https://github.com/OpenGVLab/InternVL) for complete format specifications:
```json
{
    "id": 1,
    "image": "/path/to/image",
    "mask_path": "/path/to/mask",
    "conversations":[
        {
            "from": "human",
            "value": "<image>\nRecognize all text:",
        },
        {
            "from": "gpt",
            "value":"Fill in the visual text content here",
        }
    ]
}

```

### 🚀 Training

Follow [InternVL2](https://internvl.github.io/blog/2024-07-02-InternVL-2.0/) methodology:

**Pre-training**

```bash
bash ./shell/marten_pretrain.sh
```

**Fine-tuning** 

```bash
bash ./shell/marten_finetune.sh
```

**Training with MGM**

If you want to integrate the MGM module into your own model structure, you can refer to the code.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
...
from transformers.modeling_utils import PreTrainedModel
from ..MGM_module import MGM

def dice_loss(pred, target, smooth=1e-6):
    """
    计算二分类问题的 Dice Loss
    :param pred: 预测结果, 形状为 [N, 1, H, W]
    :param target: 真实标签, 形状为 [N, 1, H, W]
    :param smooth: 平滑项，防止除零
    :return: Dice Loss 值
    """
    pred = torch.sigmoid(pred)
    
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    dice = (2. * intersection + smooth) / (union + smooth)
    
    dice_loss = 1 - dice
    
    return dice_loss


class CustomModel(PreTrainedModel):
    
    def __init__(self, config, ..., use_MGM=False):
        
        ...
        
        llm_hidden_size = config.llm_config.hidden_size
        self.use_MGM = use_MGM

        if self.use_MGM:
            self.MGM_Decoder = MGM(llm_hidden_size, hidden_size=512, dev_convs_nums=4, out_channels=1, layer_num=4)
            self.MGM_Decoder._initialize_weights()
            self.MGM_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5]))    
            self.MGM_aug_loss = dice_loss 
            self.select_llm_layer_idx = -4

    def forward(
        self,
        ...
        pixel_values_mask: Optional[bool] = None,
        ...
    ):

        ...

        """
        image_llm_hidden_features: Image-related features in the hidden layer of LLM
        text_llm_hidden_features: Text-related features in the hidden layer of LLM
        output_size: equal to image size
        image_patch_size: The size of patch of image
        image_token_num: The number of image tokens
        loss: LLM original loss

        """

        if self.use_MGM and pixel_values_masks is not None:
            
            # The other parameters are customized for InternVL dynamic slicing. If you use other VFMs, you can delete them.
            MGM_output = self.MGM_Decoder(image_llm_hidden_features, text_llm_hidden_features, output_size, image_patch_size, image_token_num)  
            loss += self.MGM_loss(MGM_output, pixel_values_mask, )
            loss += self.dice_loss(MGM_output, pixel_values_mask.float().long())

```

### 🔍 Evaluate

```bash
bash ./shell/marten/eval.sh
```

## 📌 TODO List

- [ ] Release training / evaluation code for Marten series
- [ ] Release code for mask generation
- [ ] Release dataset of MTMask6M


## 🙏 Acknowledgement
Marten is built with reference to the code of the following projects: [InternVL2](https://github.com/OpenGVLab/InternVL)

## 📜 Citation
```bibtex

```

