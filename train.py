import argparse
import trainModelClass as tmc

parser = argparse.ArgumentParser()

parser.add_argument('dir', type=str, default='flowers',
                        help='Specify directory containing images')

parser.add_argument('--lr', type=float, default=0.01,
                        help='Specify the learning rate')

parser.add_argument('--hiddenUnits', type=int, default=500,
                        help='Specify the number of hidden units')

parser.add_argument('--epochs', type=int, default=20,
                        help='Specify the number of epochs')


parser.add_argument('--arch', type=str, default='densenet121',
                        help='Specify the model you would like to use')

parser.add_argument('--gpu', type=bool, default=True,
                        help='Specify whether or not to use the GPU')

parser.add_argument('--saveDir', type=str, default='',
                        help='Specify the directory to save your model')

commandLineArgs = parser.parse_args()

tmc.trainModel(data_dir=commandLineArgs.dir, learningRate=commandLineArgs.lr, hidden_units=commandLineArgs.hiddenUnits, epochs=commandLineArgs.epochs, arch=commandLineArgs.arch, gpu=commandLineArgs.gpu, saveDir=commandLineArgs.saveDir)
