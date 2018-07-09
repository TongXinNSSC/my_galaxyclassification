import os
import torch
def pretrain(model, state_dict):
    own_state = model.state_dict()
   # print(own_state.keys())
    for name, param in state_dict.items():
        #name = 'module.'+name
        name = name[7:]
       # print(name)
        if name in own_state:
          #  print('here')
            if isinstance(param, torch.nn.Parameter): # isinstance函数来判断一个对象是否是一个已知的类型
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
               # print('here')
            except Exception:
                print('While copying the parameter named {}, '
                      'whose dimensions in the model are {} and '
                      'whose dimensions in the checkpoint are {}.'
                      .format(name, own_state[name].size(), param.size()))
                print("But don't worry about it. Continue pretraining.")

def load_state(load_path, model, optimizer=None):
    if os.path.isfile(load_path):  #如果path是一个存在的文件，返回True。否则返回False。
        checkpoint = torch.load(load_path) # 加载整个模型
        #model.load_state_dict(checkpoint['state_dict'], strict=False)  # 仅加载模型参数
        # ckpt_keys = set(checkpoint['state_dict'].keys())
        # own_keys = set(model.state_dict().keys())
        # missing_keys = own_keys - ckpt_keys
        # for k in missing_keys:
        #     print('missing keys from checkpoint {}: {}'.format(load_path, k))
        pretrain(model, checkpoint['state_dict'])

        print("=> loaded model from checkpoint '{}'".format(load_path))  #format函数用于字符串的格式化
        if optimizer != None:
            best_prec1 = checkpoint['best_prec1']
            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> also loaded optimizer from checkpoint '{}' (epoch {})"
                  .format(load_path, start_epoch))
            return start_epoch
        start_epoch = 1
        return start_epoch
    else:
        print("=> no checkpoint found at '{}'".format(load_path))
        start_epoch = 1
        return start_epoch
