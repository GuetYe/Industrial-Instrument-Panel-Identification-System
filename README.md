# An Industrial Instrument Panel Identification System Design for an Indoor Environment Based on an Improved YOLOv7 and UAV

 Industrial production often requires collecting and reading visual information on instrument panels. Traditional manual inspection and identification methods require considerable work, resulting in inefficient and inaccurate inspections. It is important to design an industrial instrument panel inspection and recognition system that can be applied in real time. In this paper, an industrial instrument panel recognition system is designed by combining an unmanned aerial vehicle (UAV) platform and deep learning techniques. For mobile UAVs, an ultrawideband (UWB) positioning module is used to implement the real-time UAV positioning and its fixed-point cruising function at each instrument panel location point. As the UAV moves around the instrument panel, a modified YOLOv7 model, compressed through a combination of pruning and quantization, is loaded to perform initial detection and recognition of the instrument panel image to ensure real-time transmission of the captured raw image to the ground side. On the ground-side server, the original YOLOv7 model is deployed to perform instrument panel detection on the initially captured images, filter out redundant background information, and then extract the tick marks and pointer information on the instrument panel through the U-Net semantic segmentation model, after which the tick marks and pointer information are processed using traditional image-processing methods. Finally, the instrument panel is identified through the angle method. This method can alleviate the accuracy loss caused by a lightweight model on the mobile side, thus ensuring the quality of accuracy. Finally, a web client is designed using Flask technology to display the uploaded instrument panel images and recognition results via a web terminal. A series of test experiments and their results show that the designed industrial instrument panel detection and recognition system is reliable and efficient in reading industrial instrument panel information in real time.