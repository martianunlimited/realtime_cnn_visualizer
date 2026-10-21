import os
import argparse
import asyncio
import time
import math
from collections import namedtuple, OrderedDict
import itertools
import scipy
import pandas as pd
import tensorflow.compat.v1 as tf
import cv2
import numpy as np

from gradcam import gradCam, gradCamToHeatMap
from guidedBackprop import registerConvBackprops, register_fc_backprops
from networks import get_network
from maps import mapsToGrid
from utils import get_outputs_from_graph, get_outputs_from_model, getConvOutput
from timed import timeit,Timer,FPS

parser = argparse.ArgumentParser()
parser.add_argument('--network', default="VGG16",
                    help="Network to visualise: One of built in keras applications (VGG16,ResNet50 ...) or path to .h5 file")
args = parser.parse_args()
class_labels=pd.read_csv('caltech.csv').iloc[:,1].values
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
tf.disable_eager_execution()

graph = tf.get_default_graph()
sess = tf.Session(config=config)
sess.as_default()
tf.keras.backend.set_session(sess)

nn, ph = get_network(args.network)

print(nn.summary())

# conv_outputs = get_outputs_from_graph(type='Conv2D')
conv_outputs = get_outputs_from_model(nn,layer_type='Conv2D')
assert conv_outputs, "Provided network has no Convolutional layers, hence I have no idea what to visualize"



conv_grids = OrderedDict( (name, mapsToGrid(output[0])) for name, output in conv_outputs.items())



convBackprops = registerConvBackprops(conv_outputs,nn.input)

fc_outputs = get_outputs_from_model(nn,layer_type="Dense")
fc_backprops = register_fc_backprops(fc_outputs,nn.input)



sess.run(tf.variables_initializer([convBackprops[name][1] for name in convBackprops ]))
sess.run(tf.variables_initializer([fc_backprops[name].selection for name in fc_backprops ]))

if fc_outputs:
    # GradCam is possible if there are fully connected layers
    gradCamA = getConvOutput(nn,-1)
    softmaxin = nn.output.op.inputs[0]
    camT = gradCam(softmaxin,gradCamA)

# TODO: make qt part work in thread
# TODO: fix this, (bad fast way to exit from programm)
close_main_loop = [False]

def rescale_img(image):
    img = np.uint8((1. - (image - np.min(image)) * 1. / (np.max(image) - np.min(image))) * 255)
    return img

def values2Map(values, num_cols=20):
    size = len(values)
    vals_filled = np.append(values, [0] * ((num_cols - len(values) % num_cols) % num_cols))
    value_map = vals_filled.reshape(-1, num_cols)
    scaled_map = (value_map-value_map.min()) / (value_map.max()-value_map.min())
    img = cv2.applyColorMap(np.uint8(scaled_map*255), cv2.COLORMAP_JET)
    return img, size

def assignWhenChanged(var,value):
    # Assingning variable takes much time
    var_value = sess.run(var)
    if var_value != value:
        print(f" Variable value changed {value}!= {var_value}")
        sess.run(var.assign(value))

async def main(ui=None, options={}):
    assert ui
    ui.fillLayers(conv_grids.keys(), fc_outputs.keys())

    cap= cv2.VideoCapture(0)
    fps = FPS()
    framenum=0
    state=True
    logits=np.zeros(256)
    while state:
            framenum=framenum+1
            state,frame=cap.read()
            #print(frame.shape)
            if ui.paused:
                frame = old_frame
            else:
                old_frame = frame
            currentGridName = 'conv1_conv' 

            timer = Timer("processing",silent=True)
            ui.loadRealImage(frame)
            timer.tick("image loaded")
            map_raw_idx = ui.convMap.raw_idx
            #dense_raw_idx = ui.denseMap.raw_idx

            frame = cv2.resize(frame,(224,224))
            frameToShow = frame.copy()
            frame = np.array([frame])
            timer.tick("frame prepared")
            gridTensor,(columns,rows), mapStack1= conv_grids['conv1_conv']
            gridTensor,(columns,rows), mapStack2= conv_grids['conv2_block2_1_conv']
            gridTensor,(columns,rows), mapStack3= conv_grids['conv3_block2_2_conv']
            gridTensor,(columns,rows), mapStack4= conv_grids['conv4_block1_2_conv']
            gridTensor,(columns,rows), mapStack5= conv_grids['conv4_block6_1_conv']
            gridTensor,(columns,rows), mapStack6= conv_grids['conv5_block1_3_conv']
            gridTensor,(columns,rows), mapStack7= conv_grids['conv5_block3_3_conv']
            neuronBackpropT,map_neuron_selection_T = convBackprops[currentGridName]
            timer.tick("setting graph vars")
            #if map_raw_idx < len(mapStack):
            #    assignWhenChanged(map_neuron_selection_T, map_raw_idx)

            timer.tick("running main session")
            fetches = {
                #"grid": gridTensor,
                "map11": mapStack1[14],
                "map12": mapStack1[17],
                "map13": mapStack1[25],
                "map14": mapStack1[22],
                "map15": mapStack1[45],
                "map21": mapStack2[14],
                "map22": mapStack2[17],
                "map23": mapStack2[25],
                "map24": mapStack2[22],
                "map25": mapStack2[45],
                "map31": mapStack3[14],
                "map32": mapStack3[17],
                "map33": mapStack3[25],
                "map34": mapStack3[22],
                "map35": mapStack3[45],
                "map41": mapStack4[14],
                "map42": mapStack4[17],
                "map43": mapStack4[25],
                "map44": mapStack4[22],
                "map45": mapStack4[45],
                "map51": mapStack5[14],
                "map52": mapStack5[17],
                "map53": mapStack5[25],
                "map54": mapStack5[22],
                "map55": mapStack5[45],
                "map61": mapStack6[14],
                "map62": mapStack6[17],
                "map63": mapStack6[25],
                "map64": mapStack6[22],
                "map65": mapStack6[45],
                "map71": mapStack7[14],
                "map72": mapStack7[17],
                "map73": mapStack7[25],
                "map74": mapStack7[22],
                "map75": mapStack7[45],
                # "cam": camT,
                #"neuronBackprop": neuronBackpropT,
                # "dense": fc_outputs[currentDense]
            }
            # if the network has fully connected layers their inspection
            # as well as GradCam algorithm is possible
            if fc_outputs:
                currentDense = 'predictions' 
                fetches.update({
                    "dense":  fc_outputs[currentDense],
                    "cam": camT
                    })

            #if "dense" in fetches:
            #    assignWhenChanged(fc_backprops[currentDense].selection, dense_raw_idx)

            fetched = sess.run(fetches,
                                feed_dict={ph:frame})
            timer.tick("Session passed")
            if "cam" in fetched:
                heatmap, coloredMap = gradCamToHeatMap(fetched["cam"],frameToShow)
                ui.setGradCam(coloredMap)
            if "dense" in fetched:
                activationMap, cell_numbers = values2Map(fetched["dense"][0])
                ui.loadActivationMap(activationMap)
                #ui.loadActivationScrollMap(activationMap, cell_numbers)
                #if dense_raw_idx < cell_numbers:
                #    ui.setDenseValue(fetched["dense"][0][dense_raw_idx])
                logits=(fetched["dense"][0]-logits)*0.04+logits
                probas=scipy.special.softmax(logits)
                class_ids=np.argsort(probas)[::-1][0:5]
                #print(class_labels[class_ids])
                #print(probas[class_ids])
                ui.setPredTop(f'{class_labels[class_ids[0]]:20} {probas[class_ids[0]]:.4f}')
                ui.setPredAll(class_labels[class_ids],probas[class_ids])

            #if "grid" in fetched:
            #    ui.loadMap(rescale_img(fetched["grid"]), (rows,columns))
            if "map11" in fetched:
                ui.loadSubMap(fetched["map11"],0,0)
                ui.loadSubMap(fetched["map12"],0,1)
                ui.loadSubMap(fetched["map13"],0,2)
                ui.loadSubMap(fetched["map14"],0,3)
                ui.loadSubMap(fetched["map15"],0,4)
                ui.loadSubMap(fetched["map21"],1,0)
                ui.loadSubMap(fetched["map22"],1,1)
                ui.loadSubMap(fetched["map23"],1,2)
                ui.loadSubMap(fetched["map24"],1,3)
                ui.loadSubMap(fetched["map25"],1,4)
                ui.loadSubMap(fetched["map31"],2,0)
                ui.loadSubMap(fetched["map32"],2,1)
                ui.loadSubMap(fetched["map33"],2,2)
                ui.loadSubMap(fetched["map34"],2,3)
                ui.loadSubMap(fetched["map35"],2,4)
                ui.loadSubMap(fetched["map41"],3,0)
                ui.loadSubMap(fetched["map42"],3,1)
                ui.loadSubMap(fetched["map43"],3,2)
                ui.loadSubMap(fetched["map44"],3,3)
                ui.loadSubMap(fetched["map45"],3,4)
                ui.loadSubMap(fetched["map51"],4,0)
                ui.loadSubMap(fetched["map52"],4,1)
                ui.loadSubMap(fetched["map53"],4,2)
                ui.loadSubMap(fetched["map54"],4,3)
                ui.loadSubMap(fetched["map55"],4,4)
                ui.loadSubMap(fetched["map61"],5,0)
                ui.loadSubMap(fetched["map62"],5,1)
                ui.loadSubMap(fetched["map63"],5,2)
                ui.loadSubMap(fetched["map64"],5,3)
                ui.loadSubMap(fetched["map65"],5,4)
                ui.loadSubMap(fetched["map71"],6,0)
                ui.loadSubMap(fetched["map72"],6,1)
                ui.loadSubMap(fetched["map73"],6,2)
                ui.loadSubMap(fetched["map74"],6,3)
                ui.loadSubMap(fetched["map75"],6,4)
            #if "map2" in fetched:
            #    ui.loadSubMap(rescale_img(fetched["map2"]),0,1)
            #cv2.imshow("neuron-backprop",fetched["neuronBackprop"][0])

            #print(f"Frame Number:{framenum:5d}, FPS:{fps():2.1f}", end="\r")
            #cv2.imshow("neuron-backprop-fc",fc_backprop[0])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # TODO: add check for number of cells here

            QApplication.processEvents()

            if close_main_loop[0]:
                break



    sys.exit(0)

import sys
import signal
from ui_v2 import Ui
from PyQt5.QtWidgets import QApplication
# from PyQt5.QtCore import QThread

def sigint_handler(*args):
    """Handler for the SIGINT signal."""
    # sys.stderr.write('\r')
    # if QMessageBox.question(None, '', "Are you sure you want to quit?",
    #                         QMessageBox.Yes | QMessageBox.No,
    #                         QMessageBox.No) == QMessageBox.Yes:
    close_main_loop[0] = True


if __name__ == '__main__':
    loop  = asyncio.get_event_loop()
    signal.signal(signal.SIGINT, sigint_handler)

    app = QApplication(sys.argv)
    ui = Ui()
    ui.show()

    loop.run_until_complete(main(ui=ui))

# writer = tf.summary.FileWriter("outputgraph", sess.graph)
# writer.close()
