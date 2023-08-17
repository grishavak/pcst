import torch
import models


def load_model_dncnn(opt, device, model_path):
    model = models.build_model(opt).to(device)
    learned_params = torch.load(model_path, map_location=device)
    model.load_state_dict(learned_params)

    return model


def load_state_params_dncnn(opt, device, model_path):
    model = models.build_model(opt).to(device)
    learned_params = torch.load(model_path, map_location=device)
    model.load_state_dict(learned_params['model_params'])

    return model


def load_model_unet(opt, device, model_path):
    model = models.build_model(opt).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model_list = [model]
    for m, state in zip(model_list, state_dict["model"]):
        m.load_state_dict(state)

    return model


def load_state_params_unet(opt, device, model_path):
    model = models.build_model(opt).to(device)
    learned_params = torch.load(model_path, map_location=device)
    model.load_state_dict(learned_params['model_params'])

    return model


def load_model_custom(opt, device, model_path):
    model = None

    """ Add your code here """

    return model    


def load_state_params_custom(opt, device, model_path):
    model = None

    """ Add your code here """

    return model    
