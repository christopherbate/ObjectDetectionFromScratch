import torch
import torch.nn as nn


class BoxPrediction(torch.nn.Module):
    '''
        num_features: the number of features in each feature tensor
        that this clas will receive as input
    '''

    def __init__(self,
                 num_class,
                 num_anchors,
                 num_features,
                 last_bias=-4.1,
                 batch_norm=True, **kwargs):
        super(BoxPrediction, self).__init__(**kwargs)
        heads = [ClassHead(in_features=num,
                           num_class=num_class,
                           last_bias=last_bias,
                           num_anchors=num_anchors[i],
                           batch_norm=batch_norm)
                 for i, num in enumerate(num_features)]
        self.heads = nn.ModuleList(heads)
        self.num_classes = num_class

        print("Created Box Heads: ", heads)
        print(num_features)

    def forward(self, feature_maps):
        logits = []
        for idx, feature_map in enumerate(feature_maps):
            layer_logits = self.heads[idx](feature_map)
            logits.append(layer_logits)

        logits = torch.cat(logits, dim=1)
        return logits


class ClassHead(torch.nn.Module):
    '''
    A series of 3 convolutional layers to regress
    anchor box detection probabilities.
    '''

    def __init__(self,
                 in_features,
                 num_class,
                 last_bias,
                 num_anchors,
                 feature_depth=128,
                 kernel_size=(3, 3),
                 batch_norm=True,
                 **kwargs):
        super(ClassHead, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.padding = (kernel_size[0]//2, kernel_size[1]//2)

        conv = [nn.Conv2d(in_features, feature_depth,
                          kernel_size=self.kernel_size,
                          stride=1, bias=False,
                          padding=self.padding),
                nn.Conv2d(feature_depth, feature_depth,
                          kernel_size=self.kernel_size,
                          stride=1, bias=False,
                          padding=self.padding),
                nn.Conv2d(in_channels=feature_depth,
                          out_channels=int(num_anchors)*int(num_class),
                          kernel_size=self.kernel_size, stride=1,
                          padding=self.padding, bias=True)]

        self.conv = nn.ModuleList(conv)
        self.relu = torch.nn.ReLU(inplace=True)

        nn.init.kaiming_normal(
            self.conv[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal(
            self.conv[1].weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal(
            self.conv[2].weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant(self.conv[-1].bias, last_bias)

        self.batch_norm = nn.ModuleList([nn.BatchNorm2d(feature_depth),
                                         nn.BatchNorm2d(feature_depth)]) if batch_norm else None
        self.num_classes = num_class

    def forward(self, x):
        '''
        x: list of feature maps
        '''

        '''
        Conv over all layers except the final layer, which
        does not get an activation or BN (as it is the logit)
        '''
        out = x
        for idx, c in enumerate(self.conv[0:-1]):
            out = c(out)

            if(self.batch_norm):
                out = self.batch_norm[idx](out)

            out = self.relu(out)

        # Last layer does not get activation, it
        # is the logit output.
        out = self.conv[-1](out)
        out = out.reshape((out.shape[0], -1, self.num_classes))

        return out
