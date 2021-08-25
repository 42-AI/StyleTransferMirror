# StyleTransferMirror
The goal of this project is to have a mirror that display the styled transfer image of the reflection in real time.

This work will be based on this paper :

[*Arbitrary Style Transfer with Style-Attentional Networks*](https://arxiv.org/pdf/1812.02342.pdf)

---

To train with the same data as the pretrain network, or to train more,
```
sh download_utils.sh
```
and download the pretrain weight at this adresse :
https://drive.google.com/file/d/1_pkMt3NSJftkoYUekNJkPV1IJuQZdfj6/view?usp=sharing

Put this weight on the `model_save` folder.

Then unzip `wikiart.zip` and `train2017.zip`. 

Move the images to the folders, respectively, `style` and `content`

Finally use the `train.ipynb`