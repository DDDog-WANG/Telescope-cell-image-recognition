# research introduction 



- For master research, now, I am doing the classification system for telescope image of cells. My Lab is AIST Collaboration Laboratory, so I do my research with some biologist of AIST to get the telescope data. For now, I have used mask-rcnn to do segmentation that get each 1 cell from the whole image, and then creat classification model to recognize that cell is healthy or sick.

  - Data example

    <img src="./Datasample/1_conf_CTCF.png" alt="1_conf_CTCF" style="zoom:80%;" />

    

    

  - Segmentation

    - Methods: I use Mask Rcnn that has published on https://github.com/matterport/Mask_RCNN

      and I refer some recent publications that used this to segment cell image that use H3K9me3 protain.

      <img src="maskrcnn.png" alt="maskrcnn" style="zoom:50%;" />

    - Due to no mask for segmentation training, only pretrained model can be used. And result is like this.

      <img src="./Datasample/segmentation.jpeg" alt="segmentation" style="zoom:50%;" />

    - Then take the boxes position to get each cell images.

    

  - Classification
  - By using bag of visual words (BoVW) methods.
    - By using CNN.
  - By using resnet transfer learning.



 	





 

