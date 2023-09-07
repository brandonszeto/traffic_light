# Fine-tuning FasterRCNN to recognize traffic lights

A deep learning model trained on a FasterRCNN backbone and fine-tuned on the LISA dataset. Source code for [this](https://brandonszeto.com/p/fine-tuning-fasterrcnn-to-recognize-traffic-lights/) project. Working notebook is also hosted on [Google Colab](https://colab.research.google.com/drive/1G-HGxmRyeuBpEZBamrkpe6OHVTD56wt9?usp=sharing).

#### Items to try in the future
- Experiment with different architectures (YOLO, SSD, etc.)
- Manually collect and annotate some data myself to expand training set
- Optimize inference and training runtimes using various techniques (gradient accumulation, batch normalization, less loops in raw python, etc.)
