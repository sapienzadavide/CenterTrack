from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchvision.models as models
import torch
import torch.nn as nn
import os

import urllib.request
from PIL import Image
from torchvision import transforms
import numpy as np
import struct 

from opts import opts

def create_folders(path_debug, path_layers):
    if not os.path.exists(path_debug):
        os.makedirs(path_debug)
    if not os.path.exists(path_layers):
        os.makedirs(path_layers)

def bin_write(f, data):
    data = data.flatten()
    fmt = 'f'*len(data)
    bin = struct.pack(fmt, *data)
    f.write(bin)

def hook(module, input, output):
    setattr(module, "_value_hook", output)
    setattr(module, "_invalue_hook", input)

def load_ex_image(model, exp_wo_dim):
    # Download an example image from the pytorch website
    # url, filename = (
    #     "https://github.com/pytorch/hub/blob/master/images/dog.jpg", "dog.jpg")
    url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    try:
        urllib.request.URLopener().retrieve(url, filename)
    except:
        urllib.request.urlretrieve(url, filename)
    
    # sample execution (requires torchvision)
    input_image = Image.open(filename)
    
    preprocess = transforms.Compose([
        transforms.Resize(exp_wo_dim),
        transforms.CenterCrop(exp_wo_dim),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    
    # create a mini-batch as expected by the model
    input_batch = input_tensor.unsqueeze(0)

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
    #     model.to('cuda')
    
    return model, input_batch

def save_debug(layer, t, path_debug, flag):
    # Save the output of the layer
    try:
        output = layer._value_hook
        if flag:
            input = layer._invalue_hook
    except:
        print("no value_hook")
        return

    o = output
    try:
        if type(o) == list and type(o[0]) == dict:
            o = o[0]['hm'].cpu().data.numpy()
        else:
            o = o.cpu().data.numpy()
    except: 
        try:
            o = o[0].cpu().data.numpy()
        except:
            return
    o = np.array(o, dtype=np.float32)
    o.tofile(path_debug+"/" + t + ".bin", format="f")
    if(flag):
        o = input
        try:
            if type(o) == list and type(o[0]) == dict:
                o = o[0]['hm'].cpu().data.numpy()
            else:
                o = o.cpu().data.numpy()
        except: 
            try:
                o = o[0].cpu().data.numpy()
            except:
                return
        o = np.array(o, dtype=np.float32)
        o.tofile(path_debug+"/input_" + t + ".bin", format="f")
        print("debug  ",o.shape)


def exp_input(model, input_batch, pre_images, pre_hms, path_debug):
    # Export the input batch 
    model(input_batch, pre_images, pre_hms)
    i = input_batch.cpu().data.numpy()
    i = np.array(i, dtype=np.float32)
    i.tofile(path_debug+"/input.bin", format="f")
    print("input: ", i.shape)
    pre_img = pre_images.cpu().data.numpy()
    pre_img = np.array(pre_img, dtype=np.float32)
    pre_img.tofile(path_debug+"/pre_imgages.bin", format="f")
    print("pre_imgages: ", pre_img.shape)
    pre_h = pre_hms.cpu().data.numpy()
    pre_h = np.array(pre_h, dtype=np.float32)
    pre_h.tofile(path_debug+"/pre_hms.bin", format="f")
    print("pre_hms: ", pre_h.shape)
    

def exp_wb_output(model, path_debug, path_layers):
    raise NotImplementedError("EXPORTER NOT IMPLEMENTED YET")

def exp_wb_output_resdcn(model, path_debug, path_layers):
    f = None
    flag = False
    for n, m in model.named_modules():    
        in_output = m._value_hook
        if ' of DCN' in str(m.type):
            flag = True
            if 'weight' in m._parameters:
                w = m._parameters['weight'].data.cpu().numpy()
                print("    DCN weights shape:", np.shape(w))
            if 'bias' in m._parameters:
                b = m._parameters['bias'].data.cpu().numpy()
                print("    DCN bias shape:", np.shape(b))
            t = '-'.join(n.split('.'))    
            file_name = path_layers+"/" + t + ".bin"
            print("open file f1: ", file_name)
            f1 = open(file_name, mode='wb')
        
        print(m.type, "##################")

        o = in_output
        if type(o) == list:
            o = o[0]['hm'].cpu().data.numpy()
        else:
            o = o.cpu().data.numpy()
        o = np.array(o, dtype=np.float32)

        t = '-'.join(n.split('.'))    
        o.tofile(path_debug+"/" + t + ".bin", format="f")
        print('------- ', n, ' ------') 
        print("debug  ",o.shape)
        
        if not(' of Conv2d' in str(m.type) or ' of ConvTranspose2d' in str(m.type) or ' of Linear' in str(m.type) or ' of BatchNorm2d' in str(m.type) or ' of DCN' in str(m.type)):
            continue

        if ' of Conv2d' in str(m.type) or ' of ConvTranspose2d' in str(m.type) or ' of Linear' in str(m.type):
            file_name = path_layers+"/" + t + ".bin"
            print("open file f: ", file_name)
            f = open(file_name, mode='wb')
        
        w = np.array([])
        b = np.array([])
        if 'weight' in m._parameters and m._parameters['weight'] is not None:
            # w = m._parameters['weight'].cpu().data.numpy()
            w = m._parameters['weight'].data.cpu().numpy()
            
            w = np.array(w, dtype=np.float32)
            print("    weights shape:", np.shape(w))
            
        if 'bias' in m._parameters and m._parameters['bias'] is not None:
            b = m._parameters['bias'].data.cpu().numpy()
            b = np.array(b, dtype=np.float32)
            print("    bias shape:", np.shape(b))
            
        if 'BatchNorm2d' in str(m.type):
            b = m._parameters['bias'].data.cpu().numpy()
            b = np.array(b, dtype=np.float32)
            s = m._parameters['weight'].data.cpu().numpy()
            s = np.array(s, dtype=np.float32)
            rm = m.running_mean.data.cpu().numpy()
            rm = np.array(rm, dtype=np.float32)
            rv = m.running_var.data.cpu().numpy()
            rv = np.array(rv, dtype=np.float32)
            if flag:
                bin_write(f1, b)
                bin_write(f1, s)
                bin_write(f1, rm)
                bin_write(f1, rv)
            else:
                bin_write(f, b)
                bin_write(f, s)
                bin_write(f, rm)
                bin_write(f, rv)
            print("    b shape:", np.shape(b))    
            print("    s shape:", np.shape(s))
            print("    rm shape:", np.shape(rm))
            print("    rv shape:", np.shape(rv))
            
        elif ' of Conv2d' in str(m.type) and flag:
            if w.size > 0 and w is not None:
                bin_write(f, w)
            if b.size > 0 and b is not None:
                bin_write(f, b)
        else:
            if w.size > 0 and w is not None:
                if flag:   
                    print("w in f1")        
                    bin_write(f1, w)
                else:
                    print("w in f")
                    bin_write(f, w)
                
            else:
                print("Error: w.size = 0")
                return
            
            if b.size > 0 and b is not None:
                if flag:         
                    print("b in f1")  
                    bin_write(f1, b)
                else:
                    print("b in f")
                    bin_write(f, b)
                
        if flag and ' of Conv2d' in str(m.type):
            f.close()
            print("f close file")
            f = None

        if ' of BatchNorm2d' in str(m.type) or ' of Linear' in str(m.type):
            if flag:
                f1.close()
                print("f1 close file")
                f1 = None
                flag = False
            else:
                f.close()
                print("close file")
                f = None

def exp_wb_output_dla(model, path_debug, path_layers):
    f = None
    flag = False
    d2write = {}
    d2write['name'] = None
    d2write['dcn'] = False
    d2write['w'] = None
    d2write['b1'] = None
    d2write['b2'] = None
    d2write['s'] = None
    d2write['rm'] = None
    d2write['rv'] = None

    flag = True
    save_input=False
    for n, m in model.named_modules():
        m.eval()
        save_input=False
        if "base.fc" in n:
            continue
        if "base" in n:
            flag=False
        t = '-'.join(n.split('.')) 
        if "base-level0-0" in t:
            save_input=True
        save_debug(m, t, path_debug, save_input)  
        print(m.type, "##################")

        if (' of DCN' in str(m.type) or ' of Conv2d' in str(m.type) or ' of ConvTranspose2d' in str(m.type) or ' of Linear' in str(m.type) or ' of BatchNorm2d' in str(m.type)):
            
            if (' of ConvTranspose2d' in str(m.type)) or (flag and ' of Conv2d' in str(m.type)):
                print("open file f1: ", path_layers+"/" + t + ".bin")
                f1 = open(path_layers+"/" + t + ".bin", mode='wb')
                w1 = m._parameters['weight'].data.cpu().numpy()
                w1 = np.array(w1, dtype=np.float32)
                # if (' of ConvTranspose2d' in str(m.type)):
                #     np.set_printoptions(threshold=np.inf)
                #     print(w1.flatten())
                bin_write(f1, w1)
                print("    weights1 shape:", np.shape(w1))
                if 'bias' in m._parameters and m._parameters['bias'] is not None:
                    b1 = m._parameters['bias'].data.cpu().numpy()
                    b1 = np.array(b1, dtype=np.float32)        
                else:
                    b1 = np.zeros(np.shape(w1)[0])
                    b1 = np.array(b1, dtype=np.float32)        
                
                bin_write(f1, b1)
                print("    bias1 shape:", np.shape(b1))
                f1.close()
                print("f1 close file")
                continue
                
            if ' of DCN' in str(m.type) or ' of Conv2d' in str(m.type) or ' of Linear' in str(m.type):
                # print(d2write['name'])
                if d2write['name'] is not None:
                    print("Error: no name in dictionary")
                    return
                d2write['name'] = path_layers+"/" + t + ".bin"

            if ' of DCN' in str(m.type):
                d2write['dcn'] = True
                flag = True
     
            if 'BatchNorm2d' in str(m.type):
                b = m._parameters['bias'].data.cpu().numpy()
                b = np.array(b, dtype=np.float32)
                d2write['b2'] = b
                s = m._parameters['weight'].data.cpu().numpy()
                s = np.array(s, dtype=np.float32)
                d2write['s'] = s
                rm = m.running_mean.data.cpu().numpy()
                rm = np.array(rm, dtype=np.float32)
                d2write['rm'] = rm
                rv = m.running_var.data.cpu().numpy()
                rv = np.array(rv, dtype=np.float32)
                d2write['rv'] = rv
    
            else:
                if 'weight' in m._parameters and m._parameters['weight'] is not None:
                    w = m._parameters['weight'].data.cpu().numpy()
                    w = np.array(w, dtype=np.float32)
                    d2write['w'] = w
                    print("    weights shape:", np.shape(w))
                    
                if 'bias' in m._parameters and m._parameters['bias'] is not None:
                    b = m._parameters['bias'].data.cpu().numpy()
                    b = np.array(b, dtype=np.float32)
                    d2write['b1'] = b
                    print("    bias shape:", np.shape(b))
                    
        # if there are all info
        if d2write['name'] is not None and d2write['w'] is not None and d2write['s'] is not None and d2write['rm'] is not None and d2write['rv'] is not None:
            print("open file f: ", d2write['name'])
            f = open(d2write['name'], mode='wb')
            bin_write(f, d2write['w'])
            if d2write['dcn']:
                bin_write(f, d2write['b1'])
            bin_write(f, d2write['b2'])
            bin_write(f, d2write['s'])
            bin_write(f, d2write['rm'])
            bin_write(f, d2write['rv'])

            # reset dictionary
            d2write['name'] = None
            d2write['dcn'] = False
            d2write['w'] = None
            d2write['b1'] = None
            d2write['b2'] = None
            d2write['s'] = None
            d2write['rm'] = None
            d2write['rv'] = None
            f.close()
            print("f close file")
            f = None

_exporter_factory = {
  'res': exp_wb_output, # default Resnet with deconv
  'dlav0': exp_wb_output, # default DLAup
  'dla': exp_wb_output_dla,
  'resdcn': exp_wb_output,
  'hourglass': exp_wb_output,
}

def weights_outputs_exporter(model, input_batch, pre_images, pre_hms, exp_wo_dim):
    opt = opts().init()
    arch = opt.arch
    print("arch: ",arch)
    arch = arch[:arch.find('_')] if '_' in arch else arch
    
    get_exporter = _exporter_factory[arch]
    
    path_debug = 'debug_'+arch+'_'+str(exp_wo_dim)
    path_layers = 'layers_'+arch+'_'+str(exp_wo_dim)
    # create folders debug and layers if do not exist
    create_folders(path_debug, path_layers)

    # add output attribute to the layers
    for n, m in model.named_modules():
        m.register_forward_hook(hook)
        
    # export input bin
    exp_input(model, input_batch, pre_images, pre_hms, path_debug)

    # export weight and output bin
    try:
        get_exporter(model, path_debug, path_layers)
    except NotImplementedError as e:
        print(str(e))
        
    # save layers list of the network
    # print(list(model.children()))
    with open(arch+"_centernet.txt", 'w') as f:
        for item in list(model.children()):
            f.write("%s\n" % item)
    print("exit exporter")