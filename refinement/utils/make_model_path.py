import datetime
import os

def make_model_path(model_name, decoder_scale, batch_size, decoder_resolution=0, num_neighborhood=24):
    now = datetime.datetime.now()
    path_backbone = ''
    if model_name == 'DenseNet161':
        path_backbone = 'D161_b' + str(batch_size).zfill(2) + '_scale' + str(decoder_scale).zfill(4)

    model_dir = 'data-' \
                      + str(now.year).zfill(4) \
                      + str(now.month).zfill(2) \
                      + str(now.day).zfill(2) \
                      + str(now.hour).zfill(2) \
                      + str(now.minute).zfill(2)
    model_path = 'runs/' + path_backbone + '/' + model_dir

    if not os.path.isdir('runs'):
        os.mkdir('runs')
    if not os.path.isdir('runs/' + path_backbone):
        os.mkdir('runs/' + path_backbone)
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    return model_path