import torch
import numpy as np
import os.path
import torch.nn as nn
import numpy as np
from Layers import layers
import copy
class Pruner:
    def __init__(self, masked_parameters,score_mode):
        self.masked_parameters = list(masked_parameters)
        self.score_mode = score_mode
        self.scores = {}
        

    def scores_mode(self):
         score_mode = self.score_mode

    def score(self, model, loss, dataloader, device):
        raise NotImplementedError

    # def _global_mask(self, sparsity,res_dir):
    #     r"""Updates masks of model with scores by sparsity level globally.
    #     """
    #     # # Set score for masked parameters to -inf 
    #     # for mask, param in self.masked_parameters:
    #     #     score = self.scores[id(param)]
    #     #     score[mask == 0.0] = -np.inf

    #     # Threshold scores
    #     if os.path.exists("{}/prev_mask.pt".format(res_dir)):
    #         prev_mask = torch.load("{}/prev_mask.pt".format(res_dir))
    #         its = 0
    #         scoring = {}
    #         for keys in self.scores.keys():
               
    #            scoring[keys] = self.scores[keys] * prev_mask[its]
    #            its = its + 1 
    #         global_scores = torch.cat([torch.flatten(v) for v in scoring.values()])
    #     else:
    #         global_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])

    #     masks = []
    #     k = int((1.0 - sparsity) * global_scores.numel())
    #     if not k < 1:
    #         threshold, _ = torch.kthvalue(global_scores, k)
    #         for mask, param in self.masked_parameters:
    #             score = self.scores[id(param)] 
    #             zero = torch.tensor([0.]).to(mask.device)
    #             one = torch.tensor([1.]).to(mask.device)
    #             mask.copy_(torch.where(score <= threshold, zero, one))
    #             masks.append(mask)
    #     prev_mask = masks
    #     torch.save(prev_mask,"{}/prev_mask.pt".format(res_dir))        
    def _global_mask(self, sparsity):
        r"""Updates masks of model with scores by sparsity level globally.
        """
        # # Set score for masked parameters to -inf 
        # for mask, param in self.masked_parameters:
        #     score = self.scores[id(param)]
        #     score[mask == 0.0] = -np.inf

        # Threshold scores
        
        global_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        k = int((1.0 - sparsity) * global_scores.numel()) # sparsity should be made 20% for each iteration
        if not k < 1:
            threshold, _ = torch.kthvalue(global_scores, k)
            for mask, param in self.masked_parameters:
                score = self.scores[id(param)] 
                zero = torch.tensor([0.]).to(mask.device)
                one = torch.tensor([1.]).to(mask.device)
                mask.copy_(torch.where(score <= threshold, zero, one))


    def _local_mask(self, sparsity):
        r"""Updates masks of model with scores by sparsity level parameter-wise.
        """
        for mask, param in self.masked_parameters:
            score = self.scores[id(param)]
            k = int((1.0 - sparsity) * score.numel())
            if not k < 1:
                threshold, _ = torch.kthvalue(torch.flatten(score), k)
                zero = torch.tensor([0.]).to(mask.device)
                one = torch.tensor([1.]).to(mask.device)
                mask.copy_(torch.where(score <= threshold, zero, one))

    def mask(self, sparsity, scope,res_dir):
        r"""Updates masks of model with scores by sparsity according to scope.
        """
        if scope == 'global':
            #self._global_mask(sparsity,res_dir)
            self._global_mask(sparsity)
        if scope == 'local':
            self._local_mask(sparsity)

    @torch.no_grad()
    def apply_mask(self):
        r"""Applies mask to prunable parameters.
        """
        for mask, param in self.masked_parameters:
            param.mul_(mask)

    def alpha_mask(self, alpha):
        r"""Set all masks to alpha in model.
        """
        for mask, _ in self.masked_parameters:
            mask.fill_(alpha)

    # Based on https://github.com/facebookresearch/open_lth/blob/master/utils/tensor_utils.py#L43
    def shuffle(self):
        for mask, param in self.masked_parameters:
            shape = mask.shape
            perm = torch.randperm(mask.nelement())
            mask = mask.reshape(-1)[perm].reshape(shape)

    def invert(self):
        for v in self.scores.values():
            v.div_(v**2)

    def stats(self):
        r"""Returns remaining and total number of prunable parameters.
        """
        remaining_params, total_params = 0, 0 
        for mask, _ in self.masked_parameters:
             remaining_params += mask.detach().cpu().numpy().sum()
             total_params += mask.numel()
        return remaining_params, total_params

def get_bn_mean_var(model):
    bn_layers = []
    for layer in model.modules():
        if isinstance(layer,layers.BatchNorm1d): 
            bn_layers.append(layer.running_var.detach())
            bn_layers.append(layer.running_mean.detach())
    return bn_layers        
class Rand(Pruner):
    def __init__(self, masked_parameters):
        super(Rand, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.randn_like(p)



class Mag(Pruner):
    def __init__(self, masked_parameters,score_mode):
        super(Mag, self).__init__(masked_parameters,score_mode)
        
    

    def score(self, model, loss, dataloader, device):
            bn_params = []
            count = 0
            count_cont = 0
            count_conts = 0
            if self.score_mode ==0:
                for layer in model.modules():
                      if isinstance(layer,(layers.BatchNorm1d,layers.BatchNorm2d)): 
                        bn_params.extend(list(layer.parameters()))
                bn_params = [x.detach() for x in bn_params]
                bn_running =   get_bn_mean_var(model)
                #bn_running =  [tuple(bn_running[i:i+2]) for i in range(0, len(bn_running), 2)]
                for kk, p in self.masked_parameters:
                    if count ==0:
                        self.scores[id(p)] = torch.clone(p.data).detach().abs_()
                        count = count + 1
                    else:   
                        
                        gamma = bn_params[2*(count-1)]
                        sd = torch.sqrt(bn_running[2*(count-1)])
                        gamma = gamma/sd
                        if len(p.shape) == 4:
                         gamma = gamma[None,:,None,None]
                        if len(p.shape) ==2:
                            if count_cont == 0 and count_conts == 0:
                                count_conts = count_conts + 1
                                #gamma = gamma.repeat(1,64)
                                #gamma = 1
                                gamma = gamma[None,:] 
                            else:
                                gamma = gamma[None,:] 
                        count = count +1
                        p.data = p/gamma
                        

                        self.scores[id(p)] = torch.clone(p.data).detach().abs_()    
            else:
                for kk, p in self.masked_parameters:
                    self.scores[id(p)] = torch.clone(p.data).detach().abs_() 

                        # if i%2 !=0:
                        #     p.data = save_alphas_for_all_layers[i-1]*p.data
                        #     self.scores[id(p)] = torch.clone(p.data).detach().abs_()
                        #     i = i + 1
                        # else:   
                             
                        #     self.scores[id(p)] = torch.clone(p.data).detach().abs_()

        # def score(self, model, loss, dataloader, device):
        #     count_linear = linear_layers(model)
        #     bn_running =   get_bn_mean_var(model)
        #     bn_running =  [tuple(bn_running[i:i+2]) for i in range(0, len(bn_running), 2)]
        #     #alpha_scales = get_aplha_scales(bn_running,self.bn_params)
        #     self.bn_params = [tuple(self.bn_params[i:i+2]) for i in range(0, len(self.bn_params), 2)]
        #     masked_counts = -1
        #     mosks = [tuple(self.masked_parameters[i:i+2]) for i in range(0, len(self.masked_parameters), 2)]
        #     counts_linear = len(count_linear) - 1 # sub one for last layer
        #     alpha_prev = 1
        #     save_alphas_for_all_layers = []
        #     save_bns_for_bias = []
        #     for p in mosks:
        #         masked_counts = masked_counts + 1
        #         if masked_counts<len(bn_running):
                
        #                 #m = int(masked_counts/2)
        #                 if masked_counts ==0:
        #                     alpha = self.bn_params[masked_counts][0]/bn_running[masked_counts][0]
        #                     alpha_prev = alpha  
        #                     alpha_b = alpha
        #                     save_alphas_for_all_layers.append(alpha)
        #                     save_alphas_for_all_layers.append(alpha_b)
        #                 else:
        #                     alpha = self.bn_params[masked_counts][0]/bn_running[masked_counts][0]
        #                     alpha_b = alpha
        #                     #alpha_prevs = torch.linalg.pinv(alpha_prev.unsqueeze(1))
        #                     alpha_prevs = 1/alpha_prev
        #                     alpha_prevs = alpha_prevs.unsqueeze(0)
        #                     alpha_prev = alpha
        #                     alpha = alpha.unsqueeze(1) *alpha_prevs
        #                     save_alphas_for_all_layers.append(alpha)
        #                     save_alphas_for_all_layers.append(alpha_b)
        #                 save_bns_for_bias.append(bn_running[masked_counts][1])
        #                 save_bns_for_bias.append(self.bn_params[masked_counts][1])    
                        
        #         else:
                    
        #             alpha = 1/(alpha_prev.unsqueeze(0))
        #             save_alphas_for_all_layers.append(alpha)
        #             save_alphas_for_all_layers.append(alpha)
        #     i = 1        
        #     for _, p in self.masked_parameters:

        #         if i <=4:
        #             if i%2 !=0:
        #                 if i ==1:
        #                     p.data = save_alphas_for_all_layers[i-1].unsqueeze(1)*p.data
        #                     self.scores[id(p)] = torch.clone(p.data).detach().abs_() 
        #                     i = i+1
        #                 else:
        #                     p.data = save_alphas_for_all_layers[i-1]*p.data
        #                     self.scores[id(p)] = torch.clone(p.data).detach().abs_() 
        #                     i = i+1
        #             else:
        #                 a = 1/save_alphas_for_all_layers[i-1]
        #                 p.data = save_alphas_for_all_layers[i-1]*(p.data- save_bns_for_bias[i-2] + a*save_bns_for_bias[i-1])
        #                 self.scores[id(p)] = torch.clone(p.data).detach().abs_()
        #                 i = i+1
        #         else:
        #             if i%2 !=0:
        #                 p.data = save_alphas_for_all_layers[i-1]*p.data
        #                 self.scores[id(p)] = torch.clone(p.data).detach().abs_()
        #                 i = i + 1
        #             else:    
        #                 self.scores[id(p)] = torch.clone(p.data).detach().abs_()



                    

# Based on https://github.com/mi-lad/snip/blob/master/snip.py#L18
class SNIP(Pruner):
    def __init__(self, masked_parameters):
        super(SNIP, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):

        # allow masks to have gradient
        for m, _ in self.masked_parameters:
            m.requires_grad = True

        # compute gradient
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss(output, target).backward()

        # calculate score |g * theta|
        for m, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(m.grad).detach().abs_()
            p.grad.data.zero_()
            m.grad.data.zero_()
            m.requires_grad = False

        # normalize score
        all_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        norm = torch.sum(all_scores)
        for _, p in self.masked_parameters:
            self.scores[id(p)].div_(norm)


# Based on https://github.com/alecwangcq/GraSP/blob/master/pruner/GraSP.py#L49
class GraSP(Pruner):
    def __init__(self, masked_parameters):
        super(GraSP, self).__init__(masked_parameters)
        self.temp = 200
        self.eps = 1e-10

    def score(self, model, loss, dataloader, device):

        # first gradient vector without computational graph
        stopped_grads = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data) / self.temp
            L = loss(output, target)

            grads = torch.autograd.grad(L, [p for (_, p) in self.masked_parameters], create_graph=False)
            flatten_grads = torch.cat([g.reshape(-1) for g in grads if g is not None])
            stopped_grads += flatten_grads

        # second gradient vector with computational graph
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data) / self.temp
            L = loss(output, target)

            grads = torch.autograd.grad(L, [p for (_, p) in self.masked_parameters], create_graph=True)
            flatten_grads = torch.cat([g.reshape(-1) for g in grads if g is not None])
            
            gnorm = (stopped_grads * flatten_grads).sum()
            gnorm.backward()
        
        # calculate score Hg * theta (negate to remove top percent)
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.grad * p.data).detach()
            p.grad.data.zero_()

        # normalize score
        all_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        norm = torch.abs(torch.sum(all_scores)) + self.eps
        for _, p in self.masked_parameters:
            self.scores[id(p)].div_(norm)

def linear_layers(model):
    linear_layers = []
    l = list( model.modules())
    for name, layer in model.named_modules():
     if isinstance(layer, layers.Linear):
        linear_layers.append(layer)
    return linear_layers 

def get_bn_mean_var(model):
    bn_layers = []
    for layer in model.modules():
        if isinstance(layer,(layers.BatchNorm1d,layers.BatchNorm2d)): 
            bn_layers.append(layer.running_var.detach())
            bn_layers.append(layer.running_mean.detach())
            
    return  bn_layers

def get_aplha_scales(bn_params,bn_p):
    alpha_scales = []
    for i in range(int(len(bn_params)/2)):
        
        alpha = bn_p[2*i]/bn_params[2*i]
        alpha_scales.append(alpha)
    return alpha_scales


class SynFlow(Pruner):
    def __init__(self, masked_parameters):
        super(SynFlow, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):
      
        @torch.no_grad()
        def linearize(model):
            # model.double()
            signs = {}
            for name, param in model.state_dict().items():
                signs[name] = torch.sign(param)
                param.abs_()
            return signs

        @torch.no_grad()
        def nonlinearize(model, signs):
            # model.float()
            for name, param in model.state_dict().items():
                param.mul_(signs[name])
        
        signs = linearize(model)

        (data, _) = next(iter(dataloader))
        input_dim = list(data[0,:].shape)
        input = torch.ones([1] + input_dim).to(device)#, dtype=torch.float64).to(device)
        output = model(input)
        torch.sum(output).backward()
        
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.grad * p).detach().abs_()
            p.grad.data.zero_()

        nonlinearize(model, signs)

