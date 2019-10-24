import predictModel as pM
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('image', type=str, default='flowers/test/1/image_06743.jpg',
                        help='Specify directory containing the image')

parser.add_argument('--checkpoint', type=str, default='checkpoint.pth',
                        help='Specify the checkpoint to upload the model')

parser.add_argument('--topk', type=int, default=5,
                        help='Specify the checkpoint to upload the model')

parser.add_argument('--cat_names', type=str, default='cat_to_name.json',
                        help='Specify the category mapping file')

parser.add_argument('--gpu', type=bool, default=True,
                        help='Specify whether or not to use the GPU')

commandLineArgs = parser.parse_args()

pM.predict(commandLineArgs.image, commandLineArgs.checkpoint, commandLineArgs.topk, commandLineArgs.cat_names, commandLineArgs.gpu)