import numpy as np
import scipy.ndimage as nd

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.logger import get_logger

logger = get_logger()


class SigmoidFocalLoss(nn.Module):
    def __init__(self, ignore_label, gamma=2.0, alpha=0.25,
                 reduction='mean'):
        super(SigmoidFocalLoss, self).__init__()
        self.ignore_label = ignore_label
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, target):
        b, h, w = target.size()
        pred = pred.view(b, -1, 1)
        pred_sigmoid = pred.sigmoid()
        target = target.view(b, -1).float()
        mask = (target.ne(self.ignore_label)).float()
        target = mask * target
        onehot = target.view(b, -1, 1)

        max_val = (-pred_sigmoid).clamp(min=0)

        pos_part = (1 - pred_sigmoid) ** self.gamma * (
                pred_sigmoid - pred_sigmoid * onehot)
        neg_part = pred_sigmoid ** self.gamma * (max_val + (
                (-max_val).exp() + (-pred_sigmoid - max_val).exp()).log())

        loss = -(self.alpha * pos_part + (1 - self.alpha) * neg_part).sum(
            dim=-1) * mask
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss


class ProbOhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_label, reduction='mean', thresh=0.6, min_kept=256,
                 down_ratio=1, use_weight=False):
        super(ProbOhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.down_ratio = down_ratio
        if use_weight:
            weight = torch.FloatTensor(
                [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
                 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                 1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       weight=weight,
                                                       ignore_index=ignore_label)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       ignore_index=ignore_label)

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        #val, ind = torch.max(target)
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_label)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            logger.info('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]
            #output, counts = torch.unique_consecutive(target, return_counts=True)
            #print('prob.shape: ',prob.shape,' mask_prob.shape: ',mask_prob.shape,' val: ',val)
            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long() # these are thought to be hard examples
                valid_mask = valid_mask * kept_mask
                # logger.info('Valid Mask: {}'.format(valid_mask.sum()))

        target = target.masked_fill_(~valid_mask, self.ignore_label)
        target = target.view(b, h, w)

        return self.criterion(pred, target)


'''
    This is generalized dice loss for organs, which means you can compute dice for more than one organ at a time
input:
    inputs: predicted segmentation map, tensor type, even Variable
    targets: real segmentation map, tensor type, even Variable

output:
    loss: scalar

    April, 2018, let's test, succeed
    By Dong Nie, dongnie@cs.unc.edu
'''
class GeneralizedDiceLoss4Organs(nn.Module):
    def __init__(self, organWeights=None, size_average=True):
        super(GeneralizedDiceLoss4Organs, self).__init__()
        self.organWeights = organWeights
        self.size_average = size_average

    def forward(self, inputs, targets, save=True):
        """
            Args:
                inputs:(n, c, h, w, d)
                targets:(n, h, w, d): 0,1,...,C-1
        """
        assert not targets.requires_grad
        assert inputs.dim() == 5 or inputs.dim() == 4, inputs.shape
        assert targets.dim() == 4 or targets.dim() == 3, targets.shape
        assert inputs.dim() == targets.dim() + 1, "{0} vs {1} ".format(inputs.dim(), targets.dim())
        assert inputs.size(0) == targets.size(0), "{0} vs {1} ".format(inputs.size(0), targets.size(0))
        assert inputs.size(2) == targets.size(1), "{0} vs {1} ".format(inputs.size(2), targets.size(1))
        assert inputs.size(3) == targets.size(2), "{0} vs {1} ".format(inputs.size(3), targets.size(2))
        if inputs.dim() == 5:
            assert inputs.size(4) == targets.size(3), "{0} vs {1} ".format(inputs.size(4), targets.size(3))

        # print 'inputs.shape, ', inputs.shape,' targets.shape, ',targets.shape,' np.unique(targets), ',np.unique(targets.data.cpu().numpy())

        eps = torch.cuda.FloatTensor(1).fill_(0.000001)
        one = torch.cuda.FloatTensor(1).fill_(1.0)
        two = torch.cuda.FloatTensor(1).fill_(2.0)

        inputSZ = inputs.size()  # it should be sth like NxCxHxW

        inputs = F.softmax(inputs, dim=1)
        # print 'max: ',np.amax(inputs.data.cpu().numpy()), 'min: ',np.amin(inputs.data.cpu().numpy())
        # preLabel = inputs.data.cpu().numpy().argmax(axis=1)
        # print 'preLabel.shape, ',preLabel.shape,' unique of prelabel: ', np.unique(preLabel), 'inputSZ:',inputSZ
        #
        # volout = sitk.GetImageFromArray(preLabel[0,...])
        # sitk.WriteImage(volout,'prePatch'+'.nii.gz')
        # volout = sitk.GetImageFromArray(targets.data.cpu().numpy()[0,...])
        # sitk.WriteImage(volout,'realPatch'+'.nii.gz')

        numOfCategories = inputSZ[1]
        assert numOfCategories == len(self.organWeights), 'organ weights is not matched with organs (bg should be included)'
        ####### Convert categorical to one-hot format

        results_one_hot = inputs

        # target1 = torch.unsqueeze(targets.data, 1)  # Nx1xHxW
        # targets_one_hot = torch.cuda.FloatTensor(inputSZ).zero_()  # NxCxHxW
        # targets_one_hot.scatter_(1, target1, 1)  # scatter along the 'numOfDims' dimension
        # jaccard_target = (targets == cls).float()

        ###### Now the prediction and target has become one-hot format
        ###### Compute the dice for each organ
        out = torch.cuda.FloatTensor(1).zero_()
        #     intersect = Variable(torch.cuda.FloatTensor([1]).zero_(), requires_grad = True)
        #     union = Variable(torch.cuda.FloatTensor([1]).zero_(), requires_grad = True)

        intersect = torch.cuda.FloatTensor(1).fill_(0.0)
        union = torch.cuda.FloatTensor(1).fill_(0.0)

        for organID in range(0, numOfCategories):
            target = (targets == organID).contiguous().view(-1, 1).squeeze(1).float() # for 2D or 3D

            #print('target.shape: ',target.shape, 'targets.shape: ',targets.shape)
            # target = targets_one_hot[:, organID, ...].contiguous().view(-1, 1).squeeze(1)  # for 2D or 3D
            result = results_one_hot[:, organID, ...].contiguous().view(-1, 1).squeeze(1)  # for 2D or 3D
            #             print 'unique(target): ',unique(target),' unique(result): ',unique(result)
            #         print 'torch.sum(target): ',torch.sum(target)
            if torch.sum(target) == 0:
                organWeight = torch.cuda.FloatTensor(1).fill_(0.0)
            else:
                organWeight = 1 / ((torch.sum(
                    target)) ** 2 + eps)  # this is necessary, otherwise, union can be too big due to too big organ weight if some organ doesnot appear
            #             print 'sum: %d'%torch.sum(target),' organWeight: %f'%organWeight
            #         intersect = torch.dot(result, target)
            intersect_vec = result * target
            # print 'organID: ',organID, ' intersect: ',torch.sum(intersect_vec).data[0],' organWeight: ',organWeight.data[0]
            intersect = intersect + organWeight * torch.sum(intersect_vec)
            #         print type(intersect)
            # binary values so sum the same as sum of squares
            result_sum = torch.sum(result)
            #         print type(result_sum)
            target_sum = torch.sum(target)
            # print 'organID: ', organID, ' union: ', (result_sum + target_sum).data[0], ' organWeight: ', organWeight.data[0]
            union = union + organWeight * (result_sum + target_sum) + (two * eps)

            # the target volume can be empty - so we still want to
            # end up with a score of 1 if the result is 0/0

        IoU = intersect / union
        # print 'IoU is ', IoU.data[0],' intersect is ',intersect.data[0],' union is ',union.data[0]
        # print 'IoU is ', IoU.data[0]
        #             out = torch.add(out, IoU.data*2)
        out = one - two * IoU
        # print 'dice is ', out.data[0]
        #     print type(out)
        return out


'''
Dice_loss
ouputs: NxCxHxW (should before softmax)
targets: NxHxW
'''
class Dice_loss:
    def __init__(self, num_classes=1, class_weights=None):
        super(Dice_loss, self).__init__()
        self.num_classes = num_classes
        self.class_weights = class_weights
    def __call__(self, outputs, targets):
        loss_dice = 0
        smooth = 1.
        outputs = F.softmax(outputs, dim=1)
        for cls in range(self.num_classes):
            jaccard_target = (targets == cls).float()
            jaccard_output = outputs[:, cls]
            intersection = (jaccard_output * jaccard_target).sum()
            if self.class_weights is not None:
                w = self.class_weights[cls]
            else:
                w = 1.
            union = jaccard_output.sum() + jaccard_target.sum()
#                loss -= torch.log((intersection + eps) / (union - intersection + eps)) * self.jaccard_weight
            loss_dice += w*(1- (2.*intersection + smooth) / (union  + smooth))
            # three kinds of loss formulas: (1) 1 - iou (2) -iou (3) -torch.log(iou)
        return loss_dice/self.num_classes
