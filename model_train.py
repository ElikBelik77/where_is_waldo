
from keras_retinanet.bin import train
train.main(
    ["--steps", "1500","--epochs", "20" ,"--snapshot", "resnet50_csv_30.h5", "--freeze-backbone", "--random-transform",
     "--image-min-side", "1800", "--image-max-side", "3000",  "csv", "csv/annotations.csv", "csv/classes.csv",
     "--val-annotations", "csv/val_annotations.csv" ,])