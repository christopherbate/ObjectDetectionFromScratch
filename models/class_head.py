import torch
import torch.nn as nn


class ClassHead(torch.nn.Module):
    '''
    A series of 3 convolutional layers to regress
    anchor box detection probabilities.
    '''
    def __init__(self, in_features=64, num_class=1, last_bias=-1.1,
                 num_anchors=2,
                 batch_norm=False, **kwargs):
        super(ClassHead, self).__init__(**kwargs)
        kernel_size = (21,11)
        padding = (10,5)

        conv = [nn.Conv2d(in_features, 256, kernel_size=kernel_size,
                          stride=1, padding=padding),
                nn.Conv2d(256, 256,                          
                          kernel_size=(3, 3),
                          stride=1,
                          padding=1),
                nn.Conv2d(in_channels=256,
                          out_channels=int(num_anchors)*int(num_class),
                          kernel_size=3, stride=1, padding=1, bias=True)]

        self.params = nn.ModuleList(conv)
        self.use_bn = batch_norm
        nn.init.constant(self.params[-1].bias, last_bias)
        self.batch_norm = nn.ModuleList([nn.BatchNorm2d(256),
                                         nn.BatchNorm2d(256)])
        self.num_classes = num_class

    def forward(self, x):
        '''
        x: list of feature maps
        '''
        features = []
        res = x
        for idx, c in enumerate(self.params[0:-1]):
            res = c(res)
            if(self.use_bn):
                res = self.batch_norm[idx](res)
            res = torch.relu(res)
            # Keep some (not all) features for visualization
            features.append(res[:,:10,:,:])            

        # Last layer does not get activation, it
        # is the logit output.
        res = self.params[-1](res)
        res = res.reshape((x.shape[0], -1, self.num_classes))
        features.append(res)

        return features


class BoxPrediction(torch.nn.Module):
    '''
        num_features: the number of features in each feature tensor
        that this clas will receive as input
    '''

    def __init__(self,
                 num_anchors=(2, 2),
                 num_features=(64, 64),
                 num_class=1,
                 last_bias=-2.1,
                 batch_norm=True, **kwargs):
        super(BoxPrediction, self).__init__(**kwargs)
        heads = [ClassHead(in_features=num,
                           num_class=num_class,
                           last_bias=last_bias,
                           num_anchors=num_anchors[i],
                           batch_norm=True)
                 for i, num in enumerate(num_features)]
        self.heads = nn.ModuleList(heads)
        self.num_classes = num_class

        print("Created Box Heads: ", heads)
        print(num_features)

    def forward(self, feature_maps):
        logits = []
        features = []
        for idx, feature_map in enumerate(feature_maps):
            ft = self.heads[idx](feature_map)
            logits.append(ft[-1])
            features.append(torch.cat(ft[0:-1], dim=1))

        logits = torch.cat(logits, dim=1)
        data = {
            'logits': logits,
            'features': features
        }
        return data
