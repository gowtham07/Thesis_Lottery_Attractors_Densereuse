import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from tqdm import tqdm
# from logging_code import logger
from logging_code import logger
from pathlib import Path
import os
from Layers import layers


def train(model, loss, optimizer, dataloader, device, epoch, verbose,log_interval=10):
    model.train()
    total = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        train_loss = loss(output, target)
        total += train_loss.item() * data.size(0)
        train_loss.backward()
        optimizer.step()
        if (batch_idx % log_interval == 0):

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), train_loss.item()))
            logger.print_and_log('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), train_loss.item()))  
            # log.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, batch_idx * len(data), len(dataloader.dataset),
            #     100. * batch_idx / len(dataloader), train_loss.item()))
               
    return total / len(dataloader.dataset)



def train_bn(model, loss, optimizer, dataloader, device, epoch, verbose, save_alphas_for_all_layers,save_bns_for_bias,save_alphas_for_all_layers_linear,save_bns_for_bias_linear,train_mode,log_interval=10):
    
   
    model.train()
    total = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data,save_alphas_for_all_layers,save_bns_for_bias,save_alphas_for_all_layers_linear,save_bns_for_bias_linear,train_mode)
        train_loss = loss(output, target)
        total += train_loss.item() * data.size(0)
        train_loss.backward()
        optimizer.step()
        if (batch_idx % log_interval == 0):

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), train_loss.item()))
            logger.print_and_log('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), train_loss.item()))  
            # log.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, batch_idx * len(data), len(dataloader.dataset),
            #     100. * batch_idx / len(dataloader), train_loss.item()))
               
    return total / len(dataloader.dataset)

def eval(model, loss, dataloader, device, verbose):
    model.eval()
    total = 0
    correct1 = 0
    correct5 = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total += loss(output, target).item() * data.size(0)
            _, pred = output.topk(5, dim=1)
            correct = pred.eq(target.view(-1, 1).expand_as(pred))
            correct1 += correct[:,:1].sum().item()
            correct5 += correct[:,:5].sum().item()
    average_loss = total / len(dataloader.dataset)
    accuracy1 = 100. * correct1 / len(dataloader.dataset)
    accuracy5 = 100. * correct5 / len(dataloader.dataset)
    # if verbose:
    print('Evaluation: Average loss: {:.4f}, Top 1 Accuracy: {}/{} ({:.2f}%)'.format(
            average_loss, correct1, len(dataloader.dataset), accuracy1))
    logger.print_and_log('Evaluation: Average loss: {:.4f}, Top 1 Accuracy: {}/{} ({:.2f}%)'.format(
            average_loss, correct1, len(dataloader.dataset), accuracy1))
    # log.info('Evaluation: Average loss: {:.4f}, Top 1 Accuracy: {}/{} ({:.2f}%)'.format(
    #         average_loss, correct1, len(dataloader.dataset), accuracy1))       
    return average_loss, accuracy1, accuracy5

def eval_1(model, loss, dataloader, device, verbose,l,sparsity):
    model.eval()
    total = 0
    correct1 = 0
    correct5 = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total += loss(output, target).item() * data.size(0)
            _, pred = output.topk(5, dim=1)
            correct = pred.eq(target.view(-1, 1).expand_as(pred))
            correct1 += correct[:,:1].sum().item()
            correct5 += correct[:,:5].sum().item()
    average_loss = total / len(dataloader.dataset)
    accuracy1 = 100. * correct1 / len(dataloader.dataset)
    accuracy5 = 100. * correct5 / len(dataloader.dataset)
    # if verbose:
    print('Evaluation: Average loss: {:.4f}, Top 1 Accuracy: {}/{} ({:.2f}%) at level {} with sparisty {}'.format(
            average_loss, correct1, len(dataloader.dataset), accuracy1,l,sparsity))
    logger.print_and_log('Evaluation: Average loss: {:.4f}, Top 1 Accuracy: {}/{} ({:.2f}%) at level {} with sparisty {}'.format(
            average_loss, correct1, len(dataloader.dataset), accuracy1,l,sparsity)) 
    # log.info('Evaluation: Average loss: {:.4f}, Top 1 Accuracy: {}/{} ({:.2f}%) at level {} with sparisty {}'.format(
    #         average_loss, correct1, len(dataloader.dataset), accuracy1,l,sparsity))     
    return average_loss, accuracy1, accuracy5

def eval_1_bn(model, loss, dataloader, device, verbose,l,sparsity,train_mode):
    model.eval()
    total = 0
    correct1 = 0
    correct5 = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            a = 1
            b =1 
            y =1
            z = 1
            output = model(data,a,b,y,z,train_mode)
            total += loss(output, target).item() * data.size(0)
            _, pred = output.topk(5, dim=1)
            correct = pred.eq(target.view(-1, 1).expand_as(pred))
            correct1 += correct[:,:1].sum().item()
            correct5 += correct[:,:5].sum().item()
    average_loss = total / len(dataloader.dataset)
    accuracy1 = 100. * correct1 / len(dataloader.dataset)
    accuracy5 = 100. * correct5 / len(dataloader.dataset)
    # if verbose:
    print('Evaluation: Average loss for model_bn: {:.4f}, Top 1 Accuracy: {}/{} ({:.2f}%) at level {} with sparisty {}'.format(
            average_loss, correct1, len(dataloader.dataset), accuracy1,l,sparsity))
    logger.print_and_log('Evaluation: Average loss for model_bn: {:.4f}, Top 1 Accuracy: {}/{} ({:.2f}%) at level {} with sparisty {}'.format(
            average_loss, correct1, len(dataloader.dataset), accuracy1,l,sparsity)) 
    # log.info('Evaluation: Average loss: {:.4f}, Top 1 Accuracy: {}/{} ({:.2f}%) at level {} with sparisty {}'.format(
    #         average_loss, correct1, len(dataloader.dataset), accuracy1,l,sparsity))     
    return average_loss, accuracy1, accuracy5  


def eval_1_bn_full(model, loss, dataloader, device, verbose,train_mode):
    model.eval()
    total = 0
    correct1 = 0
    correct5 = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            y =1
            z = 1
            a = 1
            b = 1
            output = model(data,a,b,y,z,train_mode)
            total += loss(output, target).item() * data.size(0)
            _, pred = output.topk(5, dim=1)
            correct = pred.eq(target.view(-1, 1).expand_as(pred))
            correct1 += correct[:,:1].sum().item()
            correct5 += correct[:,:5].sum().item()
    average_loss = total / len(dataloader.dataset)
    accuracy1 = 100. * correct1 / len(dataloader.dataset)
    accuracy5 = 100. * correct5 / len(dataloader.dataset)
    # if verbose:
    print('Evaluation: Average loss: {:.4f}, Top 1 Accuracy: {}/{} ({:.2f}%)'.format(
            average_loss, correct1, len(dataloader.dataset), accuracy1))
    logger.print_and_log('Evaluation: Average loss: {:.4f}, Top 1 Accuracy: {}/{} ({:.2f}%)'.format(
            average_loss, correct1, len(dataloader.dataset), accuracy1))
    # log.info('Evaluation: Average loss: {:.4f}, Top 1 Accuracy: {}/{} ({:.2f}%)'.format(
    #         average_loss, correct1, len(dataloader.dataset), accuracy1))       
    return average_loss, accuracy1, accuracy5      

def train_eval_loop(model, loss, optimizer, scheduler,train_loader, test_loader, device, epochs, verbose,l,sparsity,level):
    #test_loss, accuracy1, accuracy5 = eval(model, loss, test_loader, device, verbose)
    # rows = [[np.nan, test_loss, accuracy1, accuracy5]] 
    bn_parameters =[]
    
    for epoch in tqdm(range(epochs)):
        train_mode = True
        train_loss = train(model, loss, optimizer, train_loader, device, epoch, verbose)
        
       
        scheduler.step()
        
        
        
    test_loss, accuracy1, accuracy5 = eval_1(model, loss, test_loader, device, verbose,l,sparsity)    
    
    return  1,1
     
    #columns = ['train_loss', 'test_loss', 'top1_accuracy', 'top5_accuracy']
    # return pd.DataFrame(rows, columns=columns)

def eval_aft_level_train(model, loss, test_loader, device, verbose,l,sparisty):

    test_loss, accuracy1, accuracy5 = eval_1(model, loss, test_loader, device, verbose,l,sparisty)


def post_prune_train_eval_loop(model, loss, optimizer, scheduler, train_loader, test_loader, device, epochs, verbose,l, sparsity):
     
    
    for epoch in tqdm(range(epochs)):
        train_loss = train(model, loss, optimizer, train_loader, device, epoch, verbose)
        
       
        scheduler.step()

        
    test_loss, accuracy1, accuracy5 = eval_1(model, loss, test_loader, device, verbose,l,sparsity)     
    

def get_bn_mean_var(model):
    bn_layers = []
    for layer in model.modules():
        if isinstance(layer,(layers.BatchNorm1d,layers.BatchNorm2d)): 
            bn_layers.append(layer.running_var.detach())
            bn_layers.append(layer.running_mean.detach())
            
    return  bn_layers  

def full_train(model, loss, optimizer, scheduler, train_loader, test_loader, device, epochs, verbose): 
    for epoch in tqdm(range(epochs)):
        train_loss = train(model, loss, optimizer, train_loader, device, epoch, verbose)
        #test_loss, accuracy1, accuracy5 = eval(model, loss, test_loader, device, verbose)
        # row = [train_loss, test_loss, accuracy1, accuracy5]
        scheduler.step()
        #rows.append(row)
    test_loss, accuracy1, accuracy5 = eval(model, loss, test_loader, device, verbose)   

def full_train_bn(model, loss, optimizer, scheduler, train_loader, test_loader, device, epochs, verbose): 
    for epoch in tqdm(range(epochs)):
        
        train_loss = train(model, loss, optimizer, train_loader, device, epoch, verbose)
        s = count_parameters(model)
        logger.print_and_log("params in full dense train"+str(s))
        #test_loss, accuracy1, accuracy5 = eval(model, loss, test_loader, device, verbose)
        # row = [train_loss, test_loss, accuracy1, accuracy5]
        scheduler.step()
        #rows.append(row)
    test_loss, accuracy1, accuracy5 = eval(model, loss, test_loader, device, verbose)

def count_parameters(model):
    # table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        if 'bn' in name:
            continue

        param = parameter.numel()
        param_zero = (parameter ==0.0).sum()
        param = param - param_zero
        # table.add_row([name, param])
        total_params+=param
    # print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params    